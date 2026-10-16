#!/usr/bin/env python3
"""微调管道 · 模型桥接（模型侧 `ft_outputs` → 损失 pred；单 record）。

背景
----
模型侧（`vlfm/tsp3d_models`）与本管道（`ft_tsp3d/`）通过两类接口之一对接：

1. **直接契约**（`StubModel` / 独立实现的模型）：
   `model.forward_train(inputs)` 直接返回 `losses.py`
   头部契约的扁平 pred（键 `boxes` / `scores` / `p_exist`）。
2. **`BeaUTyDETR` 结构化输出**：
   `model(inputs)`（训练模式）直接返回只含`ft_outputs` 的包装字典
   （= `TSPHead._base_ft_forward` 的逐样本列表结构，Presence Gate 启用时另附 `gate_logits`）。
   本模块把该结构索引到单 record（训练 B=1），并映射为契约 pred：
    ```
    ft_outputs（多数键为逐样本 list，长度 = batch）      契约 pred（本管道）
    ----------------------------------------------      ---------------------------
    final_logits[0].sigmoid()                       →   scores            (K,)   裸 s_k
    final_boxes[0]                                  →   boxes             (K,6)  canonical
    gate_logits[0]（若有）                           →   p_exist           标量   σ(logit)
    ```

约定
----
* `scores` 为 raw `s_k`（sigmoid 后概率）；Gate 只作高召回 veto，开闸后不改分数。
* `p_exist` 保留计算图（`sigmoid(gate_logits)`），梯度经 `L_gate` 回流。
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
            return None   # 空列表
        return value[index] if index < len(value) else None
    return value


def adapt_ft_outputs(ft: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
    """
    `ft_outputs`（逐样本列表结构）→ 单 record 契约 pred。

    Args:
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
    pred["scores"] = torch.sigmoid(logits)   # raw s_k；Gate 开闸后保持原值
    pred["logits"] = logits.reshape(-1)

    gate = ft.get("gate_logits")
    if gate is not None:
        gate = torch.as_tensor(gate).reshape(-1)
        pred["p_exist"] = torch.sigmoid(gate[index].reshape(()))

    return pred
