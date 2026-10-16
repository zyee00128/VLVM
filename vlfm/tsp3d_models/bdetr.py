import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizerFast
import MinkowskiEngine as ME

from .mink_resnet import TSPBackbone
from .multilevel_head import TSPHead
from .ft_utils import PresenceGate
from mmdet3d.structures import bbox3d2result
import time


_DEFAULT_DATA_PATH = os.environ.get(
    "TSP3D_DATA_PATH", str(Path(__file__).resolve().parents[2] / "data" / "tsp3d_models")
)


def _build_presence_gate_with_isolated_rng(
        visual_dim: int, text_dim: int, seed: int) -> PresenceGate:
    """Initialize the CPU Gate without changing the caller's CPU/CUDA RNG state."""
    # PresenceGate is constructed on CPU and moved with the parent model later.
    # torch.manual_seed() would also reseed CUDA, while devices=[] only makes
    # fork_rng restore the CPU generator. Seed the CPU generator directly.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(int(seed))
        return PresenceGate(visual_dim=visual_dim, text_dim=text_dim)


class BeaUTyDETR(nn.Module):
    """
    3D language grounder.
    """

    def __init__(self, num_class=256, 
                num_obj_class=485, 
                input_feature_dim=3,
                d_model=128,
                data_path=_DEFAULT_DATA_PATH,
                voxel_size=0.01,
                enable_gate=False,
                num_proposals=8,
                gate_threshold=0.30,
                ft_config=None):
        """Initialize layers."""
        super().__init__()
        self.voxel_size = voxel_size

        # Allow the external fine-tuning pipeline to pass one compact config
        # without changing the legacy constructor call sites.
        ft_config = dict(ft_config or {})
        enable_gate = bool(ft_config.get("enable_gate", ft_config.get("gate", enable_gate)))
        num_proposals = int(ft_config.get("num_proposals", num_proposals))
        gate_threshold = float(ft_config.get("gate_threshold", gate_threshold))
        gate_init_seed = int(ft_config.get("gate_init_seed", 0))
        self.ft_config = {
            "enable_gate": enable_gate, "num_proposals": num_proposals,
            "gate_threshold": gate_threshold, "gate_init_seed": gate_init_seed,
        }

        # Visual encoder
        self.vision_backbone = TSPBackbone(in_channels=6)
        # Text encoder
        # t_type = f'{data_path}roberta-base/'
        t_type = data_path
        self.tokenizer = RobertaTokenizerFast.from_pretrained(t_type, local_files_only=True)
        self.text_encoder = RobertaModel.from_pretrained(t_type, local_files_only=True)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        # RoBERTa 是固定特征提取器：除 requires_grad=False 外，还必须永久关闭其 Dropout。重载 train() 会在每次 model.train() 后重申该约束。
        self.text_encoder.eval()
        self.text_projector = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, d_model),
            nn.LayerNorm(d_model, eps=1e-12),
            nn.Dropout(0.1)
        )       
        
        # self.neck = TR3DNeck()
        # Gate 只用于冻结核心的独立训练及推理。隔离其随机初始化，避免
        # 新增模块改变核心模块初始化及训练 Dropout 的 CPU/CUDA RNG 序列。
        if enable_gate:
            self.presence_gate = _build_presence_gate_with_isolated_rng(
                visual_dim=128, text_dim=d_model, seed=gate_init_seed)
        else:
            self.presence_gate = None
        self.head = TSPHead(
            voxel_size=self.voxel_size,
            num_proposals=num_proposals,
            gate_threshold=gate_threshold,
            text_dim=d_model,
        )

    # BRIEF forward.
    def forward(self, inputs, 
        gt_bboxes=None, gt_labels=None, 
        gt_all_bbox_new=None, auxi_bbox=None, 
        img_metas=None, epoch=None):
        """
        Forward pass.
        Args:
            inputs: dict
                {point_clouds, text}
                point_clouds (tensor): (B, Npoint, 3 + input_channels)
                text (list): ['text0', 'text1', ...], len(text) = B
        Returns:
            end_points: dict
        """
        # vision encoding
        points = inputs['point_clouds']
        start_time = time.time()
        coordinates, features = ME.utils.batch_sparse_collate(
                [(p[:, :3] / self.voxel_size, p[:, 0:] if p.shape[1] > 3 else p[:, :3])
                for p in points],device=points[0].device)
        x = ME.SparseTensor(coordinates=coordinates, features=features)
        x = self.vision_backbone(x)
        visual_time = time.time() - start_time
        
        # Text encoding
        start_time = time.time()
        tokenized = self.tokenizer.batch_encode_plus(
            inputs['text'], padding="longest", return_tensors="pt"
        ).to(inputs['point_clouds'][0].device)
        with torch.no_grad():
            encoded_text = self.text_encoder(**tokenized)
        if inputs.get("gate_only", False):
            projector_training = self.text_projector.training
            self.text_projector.eval()
            try:
                with torch.no_grad():
                    text_feats = self.text_projector(encoded_text.last_hidden_state)
            finally:
                self.text_projector.train(projector_training)
        else:
            text_feats = self.text_projector(encoded_text.last_hidden_state)
        text_attention_mask = tokenized.attention_mask.ne(1).bool()
        text_time = time.time() - start_time

        gate_only = bool(inputs.get("gate_only", False))
        gate_logits = None
        if self.presence_gate is not None and (gate_only or not self.training):
            # 训练时只允许 Gate-only 档位计算该分支；推理路径保持不变。
            cuda_devices = []
            if text_feats.device.type == "cuda":
                cuda_devices = [
                    text_feats.device.index
                    if text_feats.device.index is not None else torch.cuda.current_device()
                ]
            with torch.random.fork_rng(devices=cuda_devices):
                gate_logits = self.presence_gate(
                    x[-1], text_feats, text_attention_mask, detach_inputs=True)

        if gate_only:
            if gate_logits is None:
                raise RuntimeError("gate_only forward requires enable_gate=True")
            empty_boxes = x[-1].F.new_empty((0, 6))
            empty_logits = x[-1].F.new_empty((0,))
            return {"ft_outputs": {
                "final_boxes": [empty_boxes],
                "final_logits": [empty_logits],
                "gate_logits": gate_logits,
            }}

        # Extract dynamic control parameters
        sigma_sce = inputs.get('sigma_sce', None)
        tau = inputs.get('tau', None)

        if not self.training:
            start_time = time.time()
            bbox_list, head_time, diagnostics = self.head.forward_test(
                    x, text_feats, text_attention_mask, img_metas,
                    sigma_sce=sigma_sce, tau=tau, gate_logits=gate_logits,
                    gate_threshold=inputs.get("gate_threshold", self.ft_config["gate_threshold"])
            )
            if gate_logits is not None:
                diagnostics["gate_logits"] = gate_logits.detach().cpu()
                diagnostics["gate_prob"] = torch.sigmoid(gate_logits).detach().cpu()
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in bbox_list
            ]
            fusion_time = time.time() - start_time
            return bbox_results, {'loss':0.}, diagnostics, [visual_time,text_time,fusion_time-head_time,head_time]

        outputs = self.head.forward_train(x, text_feats, text_attention_mask)
        return {"ft_outputs": outputs}

    def train(self, mode: bool = True):
        """切换模型模式，同时保持完全冻结的 RoBERTa 永久处于 eval。"""
        super().train(mode)
        self.text_encoder.eval()
        return self
