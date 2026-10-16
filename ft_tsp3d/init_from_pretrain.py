#!/usr/bin/env python3
"""微调管道 · 预训练加载 + 冻结清单 + 参数组划分。

作用
----
1. `build_model`：装配 `BeaUTyDETR`（延迟 import；模型侧新头就绪后此处无需改动）。
2. `load_pretrained`：旧 ckpt（`data/tsp3d_models/tsp3d_scanrefer.pth`）部分加载，
    backbone / RoBERTa / `text_projector` / 旧 head 键直载，
    **新模块自动 missing、旧残留自动 unexpected**（`strict=False` + 分类报告）。
3. 必定冻结清单：`vision_backbone` stem + `layer1/2` + RoBERTa（LR=0）。
4. 参数组划分：
   * 组 1 新增任务头：1e-4（`head.bbox_conv` / `head.cls_conv` + **未匹配 g2/g3 的可训参数兜底**）；
   * 组 2 跨模态交互与发送侧适配层：5e-5（`text_projector` + `head.keep_conv/keep_trans/`
     `com_trans/com_cls/pos_embed/fuse/up_block/lateral_block/out_block`）；
   * 组 3 3D Backbone 顶层卷积：1e-5（`vision_backbone.layer3/layer4`）；
   * Presence Gate 单列参数组并使用组 1 峰值 LR，使旧核心可保持 checkpoint 末端 LR；
   * 每档再按 `wd=0` 拆组（`*norm*` / `*.bn.*` / `bias` / `pos_embed`），其余 `wd=1e-4`。
5. `freeze_all_eval_mode`：把参数全冻结的子模块置 eval
    （防冻结 backbone BN running-stats 漂移与冻结 transformer dropout 激活）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (str(_HERE), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# legacy 冻结前缀（保留 base 训练行为）
FROZEN_PREFIXES: Tuple[str, ...] = (
    "vision_backbone.conv1",
    "vision_backbone.norm1",
    "vision_backbone.maxpool",
    "vision_backbone.layer1.",
    "vision_backbone.layer2.",
    "text_encoder.",
)

# 逐层解冻实验使用“可训练白名单”，这样每档从同一 scanrefer checkpoint 初始化时，训练边界唯一且可审计。
SEMANTIC_HEAD_PREFIXES: Tuple[str, ...] = (
    "head.cls_conv",
    "text_projector.",
    "head.keep_conv",
    "head.keep_trans",
    "head.com_trans",
    "head.com_cls",
    "head.pos_embed",
    "head.fuse",
    "head.up_block_",
    "head.lateral_block_",
    "head.out_block_",
)

UNFREEZE_PROFILES: Dict[str, Tuple[str, ...]] = {
    # 仅最终置信度头；bbox、融合路径、两个 encoder 均冻结。
    "u0": ("head.cls_conv",),
    # 置信度头 + 文本投影与 legacy 跨模态/TGP/CBA/FPN 路径；bbox 仍冻结。
    "u1": SEMANTIC_HEAD_PREFIXES,
    # U1 + 最高层 3D 视觉语义。
    "u2": SEMANTIC_HEAD_PREFIXES + ("vision_backbone.layer4.",),
    # U2 + layer3；与当前 base 的 backbone 解冻深度一致，但 bbox 保持冻结以控制变量。
    "u3": SEMANTIC_HEAD_PREFIXES + (
        "vision_backbone.layer3.",
        "vision_backbone.layer4.",
    ),
}

# 组 2：跨模态交互与发送侧适配层。
# 特征融合使用 5e-5；最终匹配/框回归头仍使用组 1 的 1e-4。
G2_PREFIXES: Tuple[str, ...] = (
    "text_projector.",
    "head.keep_conv",
    "head.keep_trans",
    "head.com_trans",
    "head.com_cls",
    "head.pos_embed",
    "head.fuse",
    "head.up_block_",
    "head.lateral_block_",
    "head.out_block_",
)

# 组 3：3D Backbone 顶层卷积（低 LR）
G3_PREFIXES: Tuple[str, ...] = (
    "vision_backbone.layer3.",
    "vision_backbone.layer4.",
)

# 组 1 显式成员（其余未匹配 g2/g3 的可训参数 = 新增任务头，兜底入组 1）
G1_PREFIXES: Tuple[str, ...] = (
    "head.bbox_conv",
    "head.cls_conv",
)

DEFAULT_CKPT = str(_REPO / "data" / "tsp3d_models" / "tsp3d_scanrefer.pth")


def build_model(data_path: Optional[str] = None, d_model: int = 128, voxel_size: float = 0.01,
                ft_config: Optional[Dict[str, Any]] = None):
    """
    装配 `BeaUTyDETR`（延迟 import vlfm 模型定义；训练侧单向依赖，不回写推理包）。

    `ft_config` 透传模型侧微调开关（当前仅 `enable_gate`）。
    """
    from vlfm.tsp3d_models.bdetr import BeaUTyDETR

    kwargs: Dict[str, Any] = {"d_model": d_model, "voxel_size": voxel_size}
    if data_path:
        kwargs["data_path"] = data_path
    if ft_config:
        kwargs["ft_config"] = dict(ft_config)
    return BeaUTyDETR(**kwargs)


def load_pretrained(model: nn.Module, ckpt_path: str) -> Dict[str, Any]:
    """旧 ckpt 部分加载（`strict=False`）；返回分类报告（missing / unexpected / direct）。"""
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:  # torch < 1.13 无 weights_only 参数
        ckpt = torch.load(str(path), map_location="cpu")
    state = ckpt.get("model") if isinstance(ckpt, dict) else None
    if state is None and isinstance(ckpt, dict):
        state = ckpt.get("state_dict", ckpt)
    if not isinstance(state, dict):
        raise ValueError(f"unrecognized checkpoint format: {type(ckpt)}")
    state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    report = {
        "ckpt": str(path),
        "ckpt_keys": len(state),
        "missing": list(missing),      # 新模块（当前为门控头）随机初始化
        "unexpected": list(unexpected),  # 旧 head 残留键（丢弃）
        "n_missing": len(missing),
        "n_unexpected": len(unexpected),
    }
    return report


def apply_freezes(model: nn.Module, profile: str = "legacy") -> List[str]:
    """
    应用 legacy 或 U0-U3 冻结边界；返回被冻结参数名。

    U0-U3 先冻结全部参数，再按白名单解冻，
    避免模型构造函数或后续新增模块的默认 ``requires_grad`` 状态使不同档位静默串档。
    """
    profile = str(profile or "legacy").lower()
    if profile != "legacy" and profile != "gate" and profile not in UNFREEZE_PROFILES:
        raise ValueError(
            f"unknown unfreeze profile: {profile}; "
            f"expected legacy, gate or {sorted(UNFREEZE_PROFILES)}"
        )

    if profile == "gate":
        if getattr(model, "presence_gate", None) is None:
            raise RuntimeError("gate profile requires an enabled Presence Gate")
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("presence_gate.")
        trainable = [name for name, parameter in model.named_parameters()
                     if parameter.requires_grad]
        if not trainable or any(not name.startswith("presence_gate.") for name in trainable):
            raise RuntimeError(f"gate profile trainable boundary is invalid: {trainable[:8]}")
        return [name for name, parameter in model.named_parameters()
                if not parameter.requires_grad]

    if profile != "legacy":
        prefixes = UNFREEZE_PROFILES[profile]
        if getattr(model, "presence_gate", None) is not None:
            prefixes = prefixes + ("presence_gate.",)
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith(prefixes)
        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(f"unfreeze profile {profile} selected zero parameters")
        # 所有逐层档都必须冻结 bbox 与 RoBERTa
        forbidden = [
            name for name in trainable
            if name.startswith("head.bbox_conv") or name.startswith("text_encoder.")
        ]
        if forbidden:
            raise RuntimeError(f"profile {profile} leaked forbidden parameters: {forbidden[:8]}")
        return [name for name, p in model.named_parameters() if not p.requires_grad]

    # 旧 base/gate 路径保持原语义。
    frozen: List[str] = []
    for name, p in model.named_parameters():
        if name.startswith(FROZEN_PREFIXES):
            p.requires_grad = False
            frozen.append(name)
    return frozen


def freeze_all_eval_mode(model: nn.Module) -> int:
    """
    参数全冻结的模块置 eval()（含无参数 Dropout）；返回切换的模块数。

    必须检查递归参数：RoBERTa 的 Dropout 自身没有参数，旧实现只检查 ``recurse=False``，导致冻结编码器中的 37 个 Dropout 仍处于训练态。
    """
    n = 0
    for mod in model.modules():
        params = list(mod.parameters())
        if params and all(not p.requires_grad for p in params):
            mod.eval()
            n += 1
    return n


def split_param_groups(
    model: nn.Module,
    lr_g1: float = 1e-4,
    lr_g2: float = 5e-5,
    lr_g3: float = 1e-5,
    wd: float = 1e-4,
    g2_extra: Sequence[str] = (),
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """分组（G1/G2/G3/Gate × wd{base,0}）；返回参数组与名称清单。

    Gate 参数组只会出现在 gate-only 冻结核心档位；不支持与核心联合优化。
    """
    g2_prefixes = tuple(G2_PREFIXES) + tuple(g2_extra)

    def _is_wd0(name: str, p: nn.Parameter) -> bool:
        if p.ndim <= 1:  # bias / norm 类
            return True
        low = name.lower()
        return ("norm" in low) or (".bn." in low) or low.endswith(".bias") or ("pos_embed" in low)

    buckets: Dict[str, Dict[str, List[Tuple[str, nn.Parameter]]]] = {
        f"g{g}_{s}": {"params": [], "names": []}
        for g in (1, 2, 3) for s in ("wd", "nowd")
    }
    buckets.update({
        f"gate_{s}": {"params": [], "names": []}
        for s in ("wd", "nowd")
    })
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("presence_gate."):
            key_prefix = "gate"
        elif name.startswith(g2_prefixes):
            g = 2
            key_prefix = f"g{g}"
        elif name.startswith(G3_PREFIXES):
            g = 3
            key_prefix = f"g{g}"
        else:
            g = 1  # 显式 g1 + 新增头兜底
            key_prefix = f"g{g}"
        suffix = "nowd" if _is_wd0(name, p) else "wd"
        buckets[f"{key_prefix}_{suffix}"]["params"].append(p)
        buckets[f"{key_prefix}_{suffix}"]["names"].append(name)

    lrs = {1: lr_g1, 2: lr_g2, 3: lr_g3}
    groups: List[Dict[str, Any]] = []
    summary: Dict[str, List[str]] = {}
    for key in sorted(buckets):
        b = buckets[key]
        if not b["params"]:
            continue
        g = 1 if key.startswith("gate_") else int(key[1])
        groups.append({
            "params": b["params"],
            "lr": lrs[g],
            "weight_decay": 0.0 if key.endswith("nowd") else wd,
            "name": key,
            "base_lr": lrs[g],  # 调度用（warmup/cosine 的峰值锚点）
        })
        summary[key] = b["names"]
    return groups, summary


def describe(model: nn.Module, g2_extra: Sequence[str] = ()) -> Dict[str, Any]:
    """报告：冻结数 / 各组参数名与数量 / 新旧模块概览。"""
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    _groups, summary = split_param_groups(model, g2_extra=g2_extra)
    return {
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_names": frozen,
        "group_counts": {k: len(v) for k, v in summary.items()},
    }

def main() -> None:
    ap = argparse.ArgumentParser(description="预训练部分加载 + 冻结/参数组报告")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--g2-extra", default="", help="逗号分隔：额外并入组 2 的前缀")
    ap.add_argument(
        "--unfreeze-profile", default="legacy",
        choices=("legacy", "gate", *sorted(UNFREEZE_PROFILES)),
        help="冻结边界：legacy、gate-only 或逐层解冻 u0/u1/u2/u3",
    )
    ap.add_argument("--dump-json", default=None, help="把报告落盘到该路径")
    ap.add_argument("--dry-run", action="store_true", help="不实例化模型，只打印计划")
    args = ap.parse_args()

    g2_extra = [s.strip() for s in args.g2_extra.split(",") if s.strip()]
    if args.dry_run:
        print("profile:", args.unfreeze_profile)
        if args.unfreeze_profile == "gate":
            print("trainable_prefixes:", ("presence_gate.",))
        elif args.unfreeze_profile != "legacy":
            print("trainable_prefixes:", UNFREEZE_PROFILES[args.unfreeze_profile])
        print("freeze:", FROZEN_PREFIXES)
        print("g2:", tuple(G2_PREFIXES) + tuple(g2_extra))
        print("g3:", G3_PREFIXES)
        print("g1(显式+兜底):", G1_PREFIXES, "+ 其余可训参数")
        return

    model = build_model(data_path=args.data_path)
    report = load_pretrained(model, args.ckpt)
    frozen = apply_freezes(model, args.unfreeze_profile)
    info = describe(model, g2_extra)
    out = {
        "pretrain": {"ckpt": report["ckpt"], "ckpt_keys": report["ckpt_keys"],
                     "n_missing": report["n_missing"], "n_unexpected": report["n_unexpected"],
                     "missing_head": report["missing"][:20]},
        "frozen": len(frozen),
        "unfreeze_profile": args.unfreeze_profile,
        "trainable": info["trainable"],
        "total_params": info["total_params"],
        "group_counts": info["group_counts"],
    }
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print("missing（新模块，随机初始化）样例:", report["missing"][:8])
    print("unexpected（旧 head 残留，丢弃）样例:", report["unexpected"][:8])
    if args.dump_json:
        Path(args.dump_json).write_text(json.dumps({**out, "report": report}, ensure_ascii=False, indent=1))
        print("dumped:", args.dump_json)

if __name__ == "__main__":
    main()
