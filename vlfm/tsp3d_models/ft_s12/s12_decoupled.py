"""S1/S2 解耦分支（文本无关提议 + 实体/文本精修）—— 归档区，暂不接线。

本目录 `vlfm/tsp3d_models/ft_s12/` 是 **S1/S2 归档区**：存放已从模型中移出的代码，
模型装配（`multilevel_head` / `bdetr` / `vlm.tsp3d`）与 `ft_utils`工具包都**不导入本目录**；
保留原文以便重新接线，不是当前可运行路径。

目录内容
--------
* `s12_decoupled.py`（本文件）—— 提议 / 实体精修前向、结构化输出与配套损失。

背景
----
S1/S2 的设计思路是**解耦 TSP3D 模型**（文本无关提议 → 实体精修 → 匹配），
原涉及 `multilevel_head.py` 的提议/精修前向、结构化输出与配套损失、
`bdetr.py` 的 `enable_prop` / `enable_refine` 开关，
以及在线侧的 prop/refine 证据链。
模型侧现仅保留 base / gate / distill 路径，这些代码整体移入本文件保留。

与模型的接口（重新接线时需要）
------------------------------
把 `DecoupledS12Mixin` 混入 `TSPHead` 并补齐下列成员即可恢复原行为；
导出结构（`ft_outputs`）需与训练侧 `ft_tsp3d/model_bridge.py` 的映射表一致。

模型侧需提供：
  * 属性：`num_proposals`、`enable_prop`、`enable_refine`、`test_cfg`、
    `gate_threshold`、`voxel_size`、`text_dim`、`_last_ft_outputs`；
  * 层：`proposal_obj` / `proposal_reg` / `proposal_feat` / `entity_fuse` /
    `tgp_voxel` / `cba_shallow` / `entity_match` / `entity_reg` / `entity_tgp`
    （由 `_init_decoupled_layers` 创建，内部已调用 `_reset_decoupled_biases`）；
  * 推理/训练公共件：`_forward_single`、`_bbox_pred_to_bbox`（仍在模型内），
    批大小取 `..ft_utils.prediction.batch_size_of`；
  * 共享工具（**在用的模型侧代码**，随模型保留）：
    `..ft_utils.boxes.box_iou` / `.box_giou`、
    `..ft_utils.prediction.empty_prediction` / `.to_box_tensor` / `.item_at`、
    `..ft_utils.ft_modules.center_dim_to_corners` / `.focal_binary_loss` /
    `.masked_mean`。

原调用点（接回时按此改写模型）
------------------------------
1) `TSPHead.__init__`：接受 `enable_prop` / `enable_refine` 入参并保存为属性，
   在 `_init_layers` 末尾调用 `_init_decoupled_layers(out_channels)`；
2) `TSPHead.forward_test`：在 legacy 主体之前插回——
   `if self.enable_prop or self.enable_refine:` →
   `results, diagnostics, _ = self._s12_test(x[-1], text_feats, text_attention_mask,`
   `img_metas, gate_logits=gate_logits, gate_threshold=gate_threshold,`
   `shallow_visual=x[0] if len(x) > 1 else None)` → `return results, 0.0, diagnostics`；
3) `TSPHead.forward_train`：`enable_prop or enable_refine` 时改走
   `pred = self._s12_forward(...)` 并置 `self._last_ft_outputs = pred`；
4) `TSPHead._ft_losses`：按 `target / target_all / boxes / prop_boxes /
   prop_logits / match_logits` 逐样本调用 `_decoupled_loss_terms(...)`，
   把 `s1_terms` / `s2_terms` 并入返回字典的 `"s1_loss"` / `"s2_loss"`；
   baseline 分支再补 `losses["s1_loss"] = losses["s2_loss"] = 零张量`；
5) `BeaUTyDETR`：`ft_config` 重新接受 `enable_prop` / `enable_refine`
   （`mode` M2/M3/M4 推导），并透传给 `TSPHead`；
6) `vlm/tsp3d.py`：`_checkpoint_ft_config` 由 `args.ablation` / `args.losses`
   反推 prop/refine 位，`predict` 恢复 `candidate_scores` 证据链与候选字段
   `raw_score`、`diagnostics["ft"]`。

导出的逐样本契约（`_s12_forward` → `ft_outputs`）：
  `proposal_logits` / `proposal_boxes` / `proposal_points` / `proposal_features` /
  `proposal_indices` / `proposal_obj_logits` / `match_logits` / `refined_boxes` /
  `tgp_logits` / `tgp_voxel_logits` / `tgp_voxel_indices` / `tgp_voxel_points` /
  `final_logits` / `final_boxes` / `text_context`（另附别名
  `stage1_obj_logits` / `stage1_boxes` / `stage2_match_logits` / `stage2_boxes`）。

训练侧配套（按约定**原位保留**，未随本次归档移动）
----------------------------------------------
`ft_tsp3d/losses.py` 的 L_s1 / L_s2(TGP) 项、
`pipeline.py` 的 `ABLATIONS` M0–M4 与 `enable_prop` / `enable_refine` 装配、
`scripts/ft_tsp3d.sh` 的 s1/s2 档位。
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

import MinkowskiEngine as ME
from mmdet3d.structures.bbox_3d import DepthInstance3DBoxes

from ..ft_utils import prediction
from ..ft_utils.boxes import box_giou, box_iou
from ..ft_utils.ft_modules import center_dim_to_corners, focal_binary_loss, masked_mean


def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to giving probability."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


def combine_candidate_logits(proposal_logits, match_logits):
    """Fuse Stage-1 objectness and Stage-2 matching in logit space.

    Stage 2 depends on Stage 1 for candidate selection, but its ablation
    arm deliberately does not enable the explicit L_s1 loss.  Keeping the
    proposal logit in the final score gives selected proposals a gradient
    from the always-on base classification loss without silently adding
    L_s1 back to that arm.
    """
    return proposal_logits + match_logits


class DecoupledS12Mixin:
    """S1/S2 前向 / 推理 / 损失（移出模型，暂不接线）。

    混入 `TSPHead` 后需先调用 `_init_decoupled_layers(out_channels)` 建层，
    并把主干特征、文本特征与门控信息按原签名传入。见文件头接口说明。
    """

    def _init_decoupled_layers(self, out_channels):
        """原 `_init_layers` 中的 S1/S2 头模块（提议 / 实体 / TGP / CBA）。"""
        # Stage-3: text-free geometric proposal branch and entity/text refine
        # branch.  These heads operate on the last sparse backbone level and
        # therefore also work with the existing 128-channel checkpoints.
        self.proposal_obj = ME.MinkowskiConvolution(
            out_channels, 1, kernel_size=1, bias=True, dimension=3)
        self.proposal_reg = ME.MinkowskiConvolution(
            out_channels, 6, kernel_size=1, bias=True, dimension=3)
        self.proposal_feat = nn.Sequential(
            nn.Linear(out_channels, out_channels), nn.LayerNorm(out_channels), nn.GELU())
        self.entity_fuse = nn.Sequential(
            nn.Linear(out_channels + self.text_dim + 3, out_channels),
            nn.LayerNorm(out_channels), nn.GELU(), nn.Dropout(0.1))
        # Entity-level TGP: each voxel in a proposal receives a soft keep
        # probability from its visual feature and the query text.  The sigmoid
        # gate is used for feature weighting during training, so gradients do
        # not depend on MinkowskiPruning's non-differentiable hard mask.
        self.tgp_voxel = nn.Sequential(
            nn.Linear(out_channels + self.text_dim, out_channels),
            nn.LayerNorm(out_channels), nn.GELU(), nn.Linear(out_channels, 1))
        # TSPBackbone.layer1 has 64 channels.  CBA pools those high-resolution
        # features inside each proposal AABB before the final entity head.
        self.cba_shallow = nn.Sequential(
            nn.Linear(64, out_channels), nn.LayerNorm(out_channels), nn.GELU())
        self.entity_match = nn.Linear(out_channels, 1)
        self.entity_reg = nn.Linear(out_channels, 6)
        self.entity_tgp = nn.Linear(out_channels, 1)
        self._reset_decoupled_biases()

    def _reset_decoupled_biases(self):
        """Use conservative initial priors for newly introduced branches."""
        if getattr(self.proposal_obj, "bias", None) is not None:
            nn.init.constant_(self.proposal_obj.bias, bias_init_with_prob(0.20))
        nn.init.constant_(self.entity_match.bias, bias_init_with_prob(0.20))

    def _s12_forward(self, visual, text_feats, text_padding_mask=None,
                        shallow_visual=None):
        """Build text-free proposals followed by optional entity refinement.

        The return value is a dictionary of per-sample tensors.  Keeping the
        tensors in lists is deliberate: sparse scenes contain different voxel
        counts and each scene may have fewer than ``num_proposals`` candidates.
        """
        batch_size = prediction.batch_size_of(visual)
        obj = self.proposal_obj(visual).F.squeeze(-1)
        reg = self.proposal_reg(visual).F
        feats = visual.F
        coords = visual.C[:, 1:].to(dtype=feats.dtype) * float(self.voxel_size)
        batch = visual.C[:, 0].long()
        text_ctx = masked_mean(text_feats, text_padding_mask)

        out = {
            "proposal_logits": [], "proposal_boxes": [], "proposal_points": [],
            "proposal_features": [], "proposal_indices": [], "proposal_obj_logits": [],
            "match_logits": [], "refined_boxes": [], "tgp_logits": [],
            "tgp_voxel_logits": [], "tgp_voxel_indices": [], "tgp_voxel_points": [],
            "final_logits": [], "final_boxes": [], "text_context": text_ctx,
        }
        for bi in range(batch_size):
            idx = torch.nonzero(batch == bi, as_tuple=False).flatten()
            if idx.numel() == 0:
                empty = feats.new_empty((0,))
                out["proposal_logits"].append(empty)
                out["proposal_obj_logits"].append(empty)
                out["match_logits"].append(empty)
                out["tgp_logits"].append(empty)
                out["tgp_voxel_logits"].append([])
                out["tgp_voxel_indices"].append([])
                out["tgp_voxel_points"].append([])
                for key in ("proposal_boxes", "refined_boxes", "final_boxes"):
                    out[key].append(feats.new_empty((0, 6)))
                for key in ("proposal_points", "proposal_features", "proposal_indices"):
                    width = 3 if key == "proposal_points" else (feats.shape[-1] if key == "proposal_features" else 0)
                    out[key].append(feats.new_empty((0, width), dtype=torch.long if key == "proposal_indices" else feats.dtype))
                continue
            k = min(self.num_proposals, int(idx.numel()))
            values, local = obj[idx].topk(k, sorted=True)
            chosen = idx[local]
            point = coords[chosen]
            delta = reg[chosen]
            center = point + delta[:, :3]
            dims = torch.exp(delta[:, 3:6].clamp(min=-5.0, max=5.0)).clamp_min(1e-3)
            boxes = torch.cat([center, dims], dim=-1)
            pfeat = self.proposal_feat(feats[chosen])
            pos_feat = torch.tanh(point / 10.0)
            txt = text_ctx[bi].expand(k, -1)
            # Build a geometry-defined proposal cluster around each candidate.
            # The fallback nearest voxel keeps tiny/fragmented views valid.
            voxel_tgp_logits, voxel_indices, voxel_points = [], [], []
            pooled = []
            for j in range(k):
                half = dims[j] * 0.5 + 0.05
                inside = ((coords[idx] - center[j]).abs() <= half).all(dim=-1)
                members = idx[inside]
                if members.numel() == 0:
                    members = chosen[j:j + 1]
                if self.enable_refine:
                    member_text = text_ctx[bi].expand(members.numel(), -1)
                    v_logits = self.tgp_voxel(torch.cat([feats[members], member_text], dim=-1)).squeeze(-1)
                    weights = torch.sigmoid(v_logits)
                    pooled.append((feats[members] * weights.unsqueeze(-1)).sum(dim=0) /
                                  weights.sum().clamp_min(1e-4))
                    voxel_tgp_logits.append(v_logits)
                    voxel_indices.append(members)
                    voxel_points.append(coords[members])
                else:
                    # M2 deliberately has no TGP/CBA; use an unweighted
                    # proposal cluster for the plain text matcher.
                    pooled.append(feats[members].mean(dim=0))
            pfeat = self.proposal_feat(torch.stack(pooled, dim=0))

            # CBA: retrieve shallow high-resolution features within the same
            # candidate AABB.  If no shallow voxel falls inside, use the
            # nearest shallow voxel so every proposal has a differentiable path.
            cba = torch.zeros_like(pfeat)
            if self.enable_refine and shallow_visual is not None and len(shallow_visual):
                sf = shallow_visual.F
                scoords = shallow_visual.C[:, 1:].to(dtype=sf.dtype) * float(self.voxel_size)
                sbatch = shallow_visual.C[:, 0].long()
                cba_rows = []
                for j in range(k):
                    sidx = torch.nonzero(sbatch == bi, as_tuple=False).flatten()
                    if sidx.numel():
                        half = dims[j] * 0.5 + 0.05
                        inside = ((scoords[sidx] - center[j]).abs() <= half).all(dim=-1)
                        members = sidx[inside]
                        if members.numel() == 0:
                            members = sidx[((scoords[sidx] - center[j]) ** 2).sum(-1).argmin()].reshape(1)
                        cba_rows.append(self.cba_shallow(sf[members].mean(dim=0)))
                    else:
                        cba_rows.append(pfeat[j].new_zeros(pfeat.shape[-1]))
                cba = torch.stack(cba_rows, dim=0)
            pfeat = pfeat + cba
            entity = self.entity_fuse(torch.cat([pfeat, txt, pos_feat], dim=-1))
            # Candidate TGP score is retained as a diagnostic/legacy loss
            # target; voxel-level logits are the primary differentiable path.
            tgp = self.entity_tgp(entity).squeeze(-1) if self.enable_refine else entity.new_zeros((k,))
            # M2 already performs plain proposal/text matching; M3/M4 add the
            # refinement and TGP branches on top of the same match head.
            match = self.entity_match(entity).squeeze(-1) if (self.enable_prop or self.enable_refine) else values
            if self.enable_refine:
                refine = self.entity_reg(entity)
                rcenter = center + refine[:, :3]
                rdims = dims * torch.exp(refine[:, 3:6].clamp(min=-2.0, max=2.0))
                refined = torch.cat([rcenter, rdims.clamp_min(1e-3)], dim=-1)
            else:
                refined = boxes
            final_boxes = refined if self.enable_refine else boxes
            final_logits = combine_candidate_logits(values, match)
            out["proposal_logits"].append(values)
            out["proposal_obj_logits"].append(values)
            out["proposal_boxes"].append(boxes)
            out["proposal_points"].append(point)
            out["proposal_features"].append(entity)
            out["proposal_indices"].append(chosen)
            out["match_logits"].append(match)
            out["refined_boxes"].append(refined)
            out["tgp_logits"].append(tgp)
            out["tgp_voxel_logits"].append(voxel_tgp_logits)
            out["tgp_voxel_indices"].append(voxel_indices)
            out["tgp_voxel_points"].append(voxel_points)
            out["final_logits"].append(final_logits)
            out["final_boxes"].append(final_boxes)
        # Descriptive aliases make the structured output easy for the
        # external ``losses.py`` pipeline to consume without depending on
        # private field names.
        out["stage1_obj_logits"] = out["proposal_obj_logits"]
        out["stage1_boxes"] = out["proposal_boxes"]
        out["stage2_match_logits"] = out["match_logits"]
        out["stage2_boxes"] = out["refined_boxes"]
        return out

    def _s12_test(self, visual, text_feats, text_padding_mask, img_metas,
                     gate_logits=None, gate_threshold=None, shallow_visual=None):
        # Avoid constructing proposals at all when the complete batch is
        # rejected by the hard gate.  Mixed batches still share one sparse
        # tensor and are evaluated together, then filtered per sample below.
        threshold = self.gate_threshold if gate_threshold is None else float(gate_threshold)
        gate_prob = torch.sigmoid(gate_logits) if gate_logits is not None else None
        if gate_prob is not None and self.enable_prop and bool((gate_prob <= threshold).all()):
            batch_size = int(text_feats.shape[0])
            if img_metas is None:
                img_metas = [{"box_type_3d": DepthInstance3DBoxes} for _ in range(batch_size)]
            results = [prediction.empty_prediction(visual.F.device, visual.F.dtype,
                                               img_metas[i]["box_type_3d"])
                       for i in range(batch_size)]
            return results, {"ft": True, "gate_threshold": threshold,
                             "gate": [float(v.detach().cpu()) for v in gate_prob],
                             "proposal_counts": [0] * batch_size, "max_score": [0.0] * batch_size,
                             "candidate_scores": [[] for _ in range(batch_size)],
                             "fused_scores": [[] for _ in range(batch_size)]}, None

        pred = self._s12_forward(visual, text_feats, text_padding_mask, shallow_visual)
        batch_size = len(pred["final_boxes"])
        if img_metas is None:
            img_metas = [{"box_type_3d": DepthInstance3DBoxes} for _ in range(batch_size)]
        results, diagnostics = [], {"ft": True, "gate_threshold": threshold, "gate": []}
        for bi in range(batch_size):
            if gate_prob is not None:
                gp = float(gate_prob[bi].detach().cpu())
                diagnostics["gate"].append(gp)
                if self.enable_prop and gp <= threshold:
                    results.append(prediction.empty_prediction(
                        visual.F.device, visual.F.dtype,
                        img_metas[bi]["box_type_3d"]))
                    continue
            boxes = pred["final_boxes"][bi]
            logits = pred["final_logits"][bi]
            if boxes.numel() == 0:
                results.append(prediction.empty_prediction(
                    visual.F.device, visual.F.dtype,
                    img_metas[bi]["box_type_3d"]))
                continue
            # Delivery confidence is the product required by S3:
            # P_exist * s_k.  Gate probabilities are detached here because
            # this is the inference/settlement path; train-time losses use
            # the raw logits and supervise the gate independently.
            scores = logits.sigmoid()
            if gate_prob is not None:
                scores = scores * gate_prob[bi].detach()
            keep = scores >= float(self.test_cfg.get("score_thr", 0.01))
            if keep.any():
                boxes, scores = boxes[keep], scores[keep]
                order = scores.argsort(descending=True)
                nms_pre = int(self.test_cfg.get("nms_pre", self.num_proposals))
                if nms_pre > 0:
                    order = order[:nms_pre]
                boxes, scores = boxes[order], scores[order]
            else:
                boxes, scores = boxes[:0], scores[:0]
            labels = torch.zeros((len(scores),), dtype=torch.long, device=scores.device)
            # DepthInstance3DBoxes consumes center/dimension boxes when yaw is
            # disabled, matching the existing six-parameter TSP3D contract.
            box_obj = img_metas[bi]["box_type_3d"](
                boxes, box_dim=6, with_yaw=False, origin=(.5, .5, .5))
            results.append((box_obj, scores, labels))
        diagnostics["proposal_counts"] = [int(len(x)) for x in pred["final_boxes"]]
        diagnostics["candidate_scores"] = [
            x.sigmoid().detach().cpu().tolist() for x in pred["final_logits"]
        ]
        diagnostics["fused_scores"] = [
            (x.sigmoid() * (gate_prob[i].detach() if gate_prob is not None else 1.0)).detach().cpu().tolist()
            for i, x in enumerate(pred["final_logits"])
        ]
        diagnostics["max_score"] = [
            float((x.sigmoid() * (gate_prob[i].detach() if gate_prob is not None else 1.0)).max().detach().cpu())
            if len(x) else 0.0 for i, x in enumerate(pred["final_logits"])]
        return results, diagnostics, pred

    def _decoupled_loss_terms(self, pred, bi, boxes, target, target_all, dtype,
                              prop_boxes, prop_logits, match_logits):
        """逐样本 S1/S2 损失项；返回 `(s1_terms, s2_terms)`。"""
        s1_terms, s2_terms = [], []
        if prop_logits.numel():
            s1_target = torch.zeros_like(prop_logits)
            if target_all.numel():
                iou = box_iou(center_dim_to_corners(prop_boxes), center_dim_to_corners(target_all))
                best_iou, best_gt = iou.max(dim=1)
                s1_target = (best_iou >= 0.05).to(dtype)
            s1_terms.append(focal_binary_loss(prop_logits, s1_target, alpha=0.5, gamma=2.0))
            if target_all.numel() and bool(s1_target.any()):
                pos = torch.nonzero(s1_target > 0, as_tuple=False).flatten()
                matched = target_all[best_gt[pos]]
                s1_terms.append(nn.functional.smooth_l1_loss(
                    prop_boxes[pos, :3], matched[:, :3], reduction="mean"))
                s1_terms.append(nn.functional.smooth_l1_loss(
                    torch.log(prop_boxes[pos, 3:].clamp_min(1e-4)),
                    torch.log(matched[:, 3:].clamp_min(1e-4)), reduction="mean"))
                s1_terms.append(1.0 - box_giou(
                    center_dim_to_corners(prop_boxes[pos]),
                    center_dim_to_corners(matched)).diag().mean())
        if match_logits.numel():
            s2_target = torch.zeros_like(match_logits)
            if len(target):
                iou = box_iou(center_dim_to_corners(boxes), center_dim_to_corners(target))
                best_s2 = iou.max(dim=1).values.argmax()
                s2_target[best_s2] = 1.0
                chosen_target = target[iou[best_s2].argmax()]
                s2_terms.append(nn.functional.smooth_l1_loss(
                    boxes[best_s2, :3], chosen_target[:3], reduction="mean"))
                s2_terms.append(nn.functional.smooth_l1_loss(
                    torch.log(boxes[best_s2, 3:].clamp_min(1e-4)),
                    torch.log(chosen_target[3:].clamp_min(1e-4)), reduction="mean"))
                s2_terms.append(1.0 - box_giou(
                    center_dim_to_corners(boxes[best_s2:best_s2 + 1]),
                    center_dim_to_corners(chosen_target.reshape(1, 6))).mean())
            s2_terms.append(focal_binary_loss(match_logits, s2_target, alpha=0.25, gamma=2.0))
            if self.enable_refine:
                s2_terms.append(focal_binary_loss(pred["tgp_logits"][bi], s2_target, alpha=0.5, gamma=2.0))
                # TGP BCE uses the geometric target mask directly on proposal
                # voxels.  This is the differentiable counterpart of the
                # hard keep/zero operation used by the inference path.
                voxel_losses = []
                target_corners = center_dim_to_corners(target) if len(target) else None
                for v_logits, v_points in zip(pred.get("tgp_voxel_logits", [])[bi],
                                              pred.get("tgp_voxel_points", [])[bi]):
                    if v_logits.numel() == 0:
                        continue
                    if target_corners is None:
                        v_target = torch.zeros_like(v_logits)
                    else:
                        lo = target_corners[:, :3] - 0.05
                        hi = target_corners[:, 3:] + 0.05
                        inside = ((v_points[:, None, :] >= lo[None, :, :]) &
                                  (v_points[:, None, :] <= hi[None, :, :])).all(dim=-1)
                        v_target = inside.any(dim=1).to(dtype)
                    voxel_losses.append(focal_binary_loss(v_logits, v_target,
                                                          alpha=0.5, gamma=0.0))
                if voxel_losses:
                    s2_terms.append(torch.stack(voxel_losses).mean())
        return s1_terms, s2_terms
