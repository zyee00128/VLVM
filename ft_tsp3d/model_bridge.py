#!/usr/bin/env python3
"""微调管道 · 模型桥接（模型侧 `ft_outputs` → 损失契约 pred；单 record）。

背景
----
模型侧（`vlfm/tsp3d_models`）与本管道（`ft_tsp3d/`）通过两类接口之一对接：

1. **直接契约**（`StubModel` / 独立实现的模型）：`model.forward_train(inputs)` 直接
   返回 `losses.py` 头部契约的扁平 pred（键 `boxes` / `scores` / `p_exist` / …）。
2. **`BeaUTyDETR` 结构化输出**：`model(inputs)`（训练模式；`inputs["return_outputs"]=True`）
   返回的 losses dict 中带 `ft_outputs`（= `TSPHead._base_ft_forward` 的逐样本列表结构，
   另附 `gate_logits` / `gate_prob` / `final_scores`）。本模块把该结构索引到单 record
   （训练 B=1），并映射为契约 pred：

```
ft_outputs（多数键为逐样本 list，长度 = batch）      契约 pred（本管道）
----------------------------------------------      ---------------------------
final_logits[0].sigmoid()                       →   scores            (K,)   裸 s_k
final_boxes[0]                                  →   boxes             (K,6)  canonical
gate_logits[0]（若有）                           →   p_exist           标量   σ(logit)
stage1_obj_logits[0] / stage1_boxes[0]          →   prop_obj_logits / prop_boxes     (L_s1)
stage2_match_logits[0] / stage2_boxes[0]        →   match_logits / refine_boxes      (L_s2)
tgp_voxel_logits[0] / tgp_voxel_points[0]       →   tgp_logits / tgp_cluster_id /
  （逐候选列表 → 按候选序号拼接）                        tgp_voxel_xyz   (L_tgp)
```

约定
----
* `scores` 为**裸** `s_k`（sigmoid 后概率）；`Conf = P_exist×s_k` 的乘子与 detach 由
  损失侧（`losses.compute_record_losses`）结算——不在此处相乘（模型侧附加的
  `final_scores` / `gate_prob` 均为 detach 诊断量，不使用）。
* `p_exist` 保留计算图（`sigmoid(gate_logits)`），梯度经 `L_gate` / `L_gate_dist` 回流。
* TGP 簇编号 = 候选（提议）序号 `j`；`tgp_voxel_xyz` 为发送侧 canonical 系坐标
  （模型侧 `coords = C[:, 1:] × voxel_size`，与点云 / GT 同系、含 `box_margin` 判定）。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch


def is_contract_pred(pred: Any) -> bool:
    """是否已是损失契约 pred（含 `boxes` 与 `scores`）。"""
    return isinstance(pred, dict) and ("boxes" in pred) and ("scores" in pred)


def _first(value: Any, index: int = 0) -> Optional[Any]:
    """取逐样本容器的第 `index` 项；非容器原样返回；越界/空返回 None。"""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None   # 空列表（如 M2 档的 tgp_voxel_logits=[[...]] 的通道级空）
        return value[index] if index < len(value) else None
    return value


def adapt_ft_outputs(ft: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
    """`ft_outputs`（逐样本列表结构）→ 单 record 契约 pred；已是契约则原样返回。

    参数：
        ft:    模型侧结构化输出（或 `StubModel` 契约 pred）。
        index: batch 内位置（训练 B=1，恒 0；保留参数便于将来批式收集）。
    """
    if is_contract_pred(ft):
        return ft
    if not isinstance(ft, dict):
        raise RuntimeError(
            "训练前向输出不可识别：期望契约 pred（含 boxes/scores）或模型侧 ft_outputs"
            f"（inputs['return_outputs']=True）；实际得到 {type(ft).__name__}。")

    pred: Dict[str, Any] = {}

    boxes = _first(ft.get("final_boxes"), index)
    logits = _first(ft.get("final_logits"), index)
    if boxes is None or logits is None:
        raise RuntimeError("ft_outputs 缺少 final_boxes / final_logits（模型侧契约不匹配）。")
    pred["boxes"] = boxes
    pred["scores"] = torch.sigmoid(logits)   # 裸 s_k；门控乘子由损失侧结算

    gate = ft.get("gate_logits")
    if gate is not None:
        gate = torch.as_tensor(gate).reshape(-1)
        pred["p_exist"] = torch.sigmoid(gate[index].reshape(()))

    prop_logits = _first(ft.get("stage1_obj_logits", ft.get("proposal_obj_logits")), index)
    prop_boxes = _first(ft.get("stage1_boxes", ft.get("proposal_boxes")), index)
    if prop_logits is not None and prop_boxes is not None:
        pred["prop_obj_logits"] = prop_logits
        pred["prop_boxes"] = prop_boxes

    match_logits = _first(ft.get("stage2_match_logits", ft.get("match_logits")), index)
    refine_boxes = _first(ft.get("stage2_boxes", ft.get("refined_boxes")), index)
    if match_logits is not None:
        pred["match_logits"] = match_logits
    if refine_boxes is not None:
        pred["refine_boxes"] = refine_boxes

    # ---- TGP：逐候选体素列表 → 按候选序号拼接（簇级等权在损失侧逐簇平均）----
    v_logits = _first(ft.get("tgp_voxel_logits"), index)
    v_points = _first(ft.get("tgp_voxel_points"), index)
    if v_logits and v_points:
        logits_l, ids_l, xyz_l = [], [], []
        for j, (lv, vp) in enumerate(zip(v_logits, v_points)):
            if lv is None or vp is None:
                continue
            lv = lv.reshape(-1)
            vp = vp.reshape(-1, 3)
            if lv.numel() == 0 or vp.numel() == 0:
                continue
            logits_l.append(lv)
            ids_l.append(torch.full((lv.numel(),), j, dtype=torch.long, device=lv.device))
            xyz_l.append(vp.to(dtype=lv.dtype))
        if logits_l:
            pred["tgp_logits"] = torch.cat(logits_l)
            pred["tgp_cluster_id"] = torch.cat(ids_l)
            pred["tgp_voxel_xyz"] = torch.cat(xyz_l)
    return pred
