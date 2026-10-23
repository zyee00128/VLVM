import torch
import time
import logging
from torch import nn

import MinkowskiEngine as ME
from mmdet3d.structures.bbox_3d import DepthInstance3DBoxes
from .trans_modules import (BiEncoder, BiEncoderLayer, PositionEmbeddingLearned)
from .ft_utils import (
    batch_size_of,
    empty_prediction,
)


class MinkowskiFeatureFusionBlock(nn.Module):
    """
    Block to fuse backbone features with text features in Minkowski space.
    """
    def __init__(self, backbone_channels, text_channels, 
                output_channels, dimension=3):
        super(MinkowskiFeatureFusionBlock, self).__init__()
        self.conv = ME.MinkowskiConvolution(
            backbone_channels + text_channels,
            output_channels,
            kernel_size=1,
            stride=1,
            dimension=dimension
        )
        self.norm = ME.MinkowskiBatchNorm(output_channels)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, backbone_feats, text_feats):
        # Extract batch indices from the coordinates of backbone features 
        # (Note: index 0 is batch index in ME)
        batch_indices = backbone_feats.C[:, 0].long()  
        # Repeat text features for each point in the corresponding batch
        repeated_text_feats = text_feats[batch_indices]
        # Combine the backbone and text features
        combined_features = torch.cat([backbone_feats.F, repeated_text_feats], dim=1)
        combined_feats = ME.SparseTensor(
            features=combined_features,
            coordinate_map_key=backbone_feats.coordinate_map_key,
            coordinate_manager=backbone_feats.coordinate_manager
        )

        # Convolution and normalization
        x = self.conv(combined_feats)
        x = self.norm(x)
        return self.relu(x)


class TSPHead(nn.Module):
    def __init__(self, voxel_size=.01, volume_threshold=27,
                prune_threshold=(0.3, 0.7), com_threshold=0.15, 
                test_cfg=None, 
                in_channels=(128, 128, 128), out_channels=128,
                n_reg_outs=6,  # Regression parameters: [dx_offset, dy_offset, dz_offset, log(w), log(h), log(l)]
                n_classes=1,
                num_proposals=8,
                gate_threshold=0.30,
                text_dim=128):
        super(TSPHead, self).__init__()
        self.voxel_size = voxel_size
        self.volume_threshold = volume_threshold
        self.prune_threshold = prune_threshold
        self.com_threshold = com_threshold

        self.num_proposals = max(1, int(num_proposals))
        self.gate_threshold = float(gate_threshold)
        self.text_dim = int(text_dim)
        
        self.test_cfg = test_cfg \
        if test_cfg is not None \
        else dict(nms_pre=1, iou_thr=.5, score_thr=.01) # Parse inference NMS settings
        # Limit maximum scale of reconstructed voxel sampling for bi-directional attention inference
        self.num_samples_com = 2400
        self._init_layers(in_channels, out_channels, 
                        n_reg_outs, n_classes)

    @staticmethod
    def make_block(in_channels, out_channels, kernel_size=3):
        """ 
        Build a standard sparse 3D convolution-normalization-activation pipeline block
        """
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels,
                                    kernel_size=kernel_size, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True))

    @staticmethod
    def make_up_block(in_channels, out_channels, generative=False):
        """ 
        Build a transposed convolution 3D upsampling block, 
        enabling 'generative' allows dynamic generation of new feature coordinates in unknown spaces 
        """
        conv = ME.MinkowskiGenerativeConvolutionTranspose if generative \
            else ME.MinkowskiConvolutionTranspose
        return nn.Sequential(
            conv(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True))

    def _init_layers(self, in_channels, out_channels, n_reg_outs, n_classes):
        """
        Instantiate all 3D convolutions, transformation layers, 
        and learnable positional embedding parameters inside the detection head 
        """
        self.bbox_conv = ME.MinkowskiConvolution(
            out_channels, n_reg_outs, kernel_size=1, bias=True, dimension=3)
        self.cls_conv = ME.MinkowskiConvolution(
            out_channels, n_classes, kernel_size=1, bias=True, dimension=3)
        self.keep_conv = nn.ModuleList([
            ME.MinkowskiConvolution(out_channels, 1, kernel_size=1, bias=True, dimension=3),
            ME.MinkowskiConvolution(out_channels, 1, kernel_size=1, bias=True, dimension=3)
        ])  # pruning score-evaluation conv list
        self.pos_embed = PositionEmbeddingLearned(3, 128)
        
        # Build a three-layer bi-directional transformer subnet 
        # to associate and align prompt word features and spatial 3D voxels during feature routing
        bi_layer0 = BiEncoderLayer(
            128, dropout=0.1, activation="relu",
            n_heads=8, dim_feedforward=128,
            self_attend_lang=True, self_attend_vis=True,
            use_butd_enc_attn=False
        )
        bi_layer1 = BiEncoderLayer(
            128, dropout=0.1, activation="relu",
            n_heads=8, dim_feedforward=128,
            self_attend_lang=True, self_attend_vis=True,
            use_butd_enc_attn=False
        )
        bi_layer2 = BiEncoderLayer(
            128, dropout=0.1, activation="relu",
            n_heads=8, dim_feedforward=128,
            self_attend_lang=True, self_attend_vis=True,
            use_butd_enc_attn=False
        )
        self.keep_trans = nn.ModuleList([BiEncoder(bi_layer0, 2), BiEncoder(bi_layer1, 2)])
        self.com_trans = BiEncoder(bi_layer2, 2)
        self.pruning = ME.MinkowskiPruning()
        self.com_cls = nn.Conv1d(128, 1, kernel_size=1, bias=True)

        # Dynamically assemble multi-scale feature pyramid layers
        for i in range(len(in_channels)):
            # Build the upsampling path
            if i > 0:
                setattr(
                    self, f'up_block_{i}',
                    self.make_up_block(in_channels[i], in_channels[i - 1], generative=True))
            # Build lateral local geometric refinement module
            setattr(
                self, f'lateral_block_{i}',
                self.make_block(in_channels[i], in_channels[i]))  
            # Define final aggregation calculation node of the geometric pyramid
            if i == 0:
                setattr(
                    self, f'out_block_{i}',
                    self.make_block(in_channels[i], out_channels))
                
        self.fuse = MinkowskiFeatureFusionBlock(out_channels, self.text_dim, out_channels)


    @staticmethod
    def _soft_reweight_sparse(x, logits, keep_complement=False):
        """Apply a differentiable counterpart of a binary sparse keep mask."""
        if logits is None:
            return x
        
        weights = logits.reshape(-1).sigmoid()
        if keep_complement:
            weights = 1.0 - weights
        return ME.SparseTensor(
            features=x.F * weights.unsqueeze(-1),
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )

    @staticmethod
    def _lateral_features_at(source, target):
        """
        Align an integer-grid lateral skip to ``target`` coordinates.

        Both tensors are sparse feature-pyramid levels on the same integer lattice.  
        Using ``features_at_coordinates`` here unnecessarily invokes MinkowskiEngine's continuous interpolation kernel, 
        which can return an ``(N_target, N_source)`` tensor for tiny maps instead of ``(N_target, C)``. 
        A kernel-size-one coordinate map is the native and exact operation: 
        matched coordinates copy their source feature and generatively created target coordinates receive zero.

        Structural mismatches are model errors and must not be hidden by a zero-feature fallback.
        """
        source_channels = int(source.F.shape[1])
        target_channels = int(target.F.shape[1])
        if source_channels != target_channels:
            raise RuntimeError(
                "lateral feature channel mismatch: "
                f"source={source_channels}, target={target_channels}"
            )

        source_stride = tuple(int(v) for v in source.tensor_stride)
        target_stride = tuple(int(v) for v in target.tensor_stride)
        if source_stride != target_stride:
            raise RuntimeError(
                "lateral tensor-stride mismatch: "
                f"source={source_stride}, target={target_stride}"
            )

        source_manager = source.coordinate_manager
        target_manager = target.coordinate_manager
        if source_manager._manager is not target_manager._manager:
            raise RuntimeError(
                "lateral tensors must share the same coordinate manager"
            )

        aligned = target.F.new_zeros((len(target), target_channels))
        kernel_map = source_manager.kernel_map(
            source.coordinate_map_key,
            target.coordinate_map_key,
            kernel_size=1,
            region_type=ME.RegionType.HYPER_CUBE,
        )
        for in_idx, out_idx in kernel_map.values():
            in_idx = in_idx.to(device=source.F.device, dtype=torch.long)
            out_idx = out_idx.to(device=target.F.device, dtype=torch.long)
            aligned = aligned.index_copy(0, out_idx, source.F[in_idx])
        return aligned

    def _base_ft_forward(self, backbone_feats, text_feats, text_padding_mask):
        """
        Train the legacy head through the same feature pyramid used at inference.

        Inference physically removes voxels with threshold masks.
        Training keeps the identical multi-scale/attention/completion topology 
        but replaces the two binary decisions with sigmoid feature weights, 
        preserving gradients for ``keep_conv`` and ``com_cls``.
        """
        inputs = backbone_feats[1:]
        x = inputs[-1]
        prune_logits = None

        for i in range(len(inputs) - 1, -1, -1):
            if i in (1, 0):
                # _prune_inference keeps 1-sigmoid(prune_logits).
                x = self._soft_reweight_sparse(x, prune_logits, keep_complement=True)
                x = getattr(self, f"up_block_{i + 1}")(x)
                skip_features = self._lateral_features_at(inputs[i], x)
                skip = ME.SparseTensor(
                    features=skip_features,
                    coordinate_map_key=x.coordinate_map_key,
                    coordinate_manager=x.coordinate_manager,
                )
                x = x + skip

            if i == 0:
                # sigmoid scores softly weight candidates rather than thresholding them out of the graph.
                sampled_features, sampled_coords = [], []
                for permutation in inputs[0].decomposition_permutations:
                    if len(permutation) > self.num_samples_com:
                        choice = torch.randperm(len(permutation), device=permutation.device)
                        choice = torch.sort(choice[:self.num_samples_com]).values
                        sampled_features.append(inputs[0].F[permutation][choice])
                        sampled_coords.append(inputs[0].C[permutation][choice])
                    else:
                        n_pad = self.num_samples_com - len(permutation)
                        sampled_features.append(torch.cat((
                            inputs[0].F[permutation],
                            inputs[0].F.new_zeros((n_pad, inputs[0].F.shape[1])),
                        ), dim=0))
                        sampled_coords.append(torch.cat((
                            inputs[0].C[permutation],
                            inputs[0].C.new_full((n_pad, inputs[0].C.shape[1]), -1),
                        ), dim=0))
                sampled_features = torch.stack(sampled_features)
                sampled_coords = torch.stack(sampled_coords)
                sampled_features, text_feats = self.com_trans(
                    vis_feats=sampled_features.contiguous(),
                    pos_feats=self.pos_embed(
                        sampled_coords[:, :, 1:] * self.voxel_size
                    ).transpose(1, 2).contiguous(),
                    padding_mask=sampled_coords[:, :, 0] == -1,
                    text_feats=text_feats,
                    text_padding_mask=text_padding_mask,
                )
                completion_logits = self.com_cls(
                    sampled_features.transpose(1, 2).contiguous()
                ).transpose(1, 2).contiguous()
                valid = sampled_coords[:, :, 0] != -1
                candidate_features = sampled_features[valid]
                candidate_coords = sampled_coords[valid]
                completion_logits = completion_logits[valid].reshape(-1)
                candidate_features = (
                    candidate_features * completion_logits.sigmoid().unsqueeze(-1)
                )
                # Soft training keeps the full coordinate map, so most completion candidates already exist in the reconstructed trunk.
                # Fuse their weighted features at existing positions instead of dropping the entire completion branch 
                # as the collision-avoiding hard inference path must do.
                completion = ME.SparseTensor(
                    features=candidate_features,
                    coordinates=candidate_coords,
                    coordinate_manager=x.coordinate_manager,
                    tensor_stride=x.tensor_stride,
                    device=x.device,
                )
                existing_completion = completion.features_at_coordinates(x.C.float())
                if existing_completion.shape == x.F.shape:
                    x = ME.SparseTensor(
                        features=x.F + existing_completion,
                        coordinate_map_key=x.coordinate_map_key,
                        coordinate_manager=x.coordinate_manager,
                    )
                duplicate = (
                    candidate_coords.unsqueeze(1) == x.C.unsqueeze(0)
                ).all(dim=-1).any(dim=1)
                candidate_features = candidate_features[~duplicate]
                candidate_coords = candidate_coords[~duplicate]
                if len(candidate_features):
                    trunk_features = x.features_at_coordinates(candidate_coords.float())
                    if trunk_features.shape == candidate_features.transpose(0, 1).shape:
                        trunk_features = trunk_features.transpose(0, 1).contiguous()
                    if trunk_features.shape == candidate_features.shape:
                        completion_features = trunk_features + candidate_features
                        x = ME.SparseTensor(
                            features=torch.cat((x.F, completion_features), dim=0),
                            coordinates=torch.cat((x.C, candidate_coords), dim=0),
                            coordinate_manager=x.coordinate_manager,
                            tensor_stride=x.tensor_stride,
                            device=x.device,
                        )

            if i > 0:
                sampled_coords, sampled_features = [], []
                max_len_x = max(
                    len(permutation) for permutation in x.decomposition_permutations)
                for permutation in x.decomposition_permutations:
                    padding_size = max_len_x - len(permutation)
                    padded_features = x.F[permutation]
                    padded_coords = x.C[permutation]
                    if padding_size:
                        padded_features = torch.cat(
                            [padded_features,
                             x.F.new_zeros((padding_size, x.F.shape[1]))], dim=0)
                        padded_coords = torch.cat(
                            [padded_coords,
                             x.C.new_full((padding_size, x.C.shape[1]), -1)], dim=0)
                    sampled_features.append(padded_features)
                    sampled_coords.append(padded_coords)
                sampled_features = torch.stack(sampled_features)
                sampled_coords = torch.stack(sampled_coords)
                sampled_features, text_feats = self.keep_trans[i - 1](
                    vis_feats=sampled_features.contiguous(),
                    pos_feats=self.pos_embed(
                        sampled_coords[:, :, 1:] * self.voxel_size
                    ).transpose(1, 2).contiguous(),
                    padding_mask=sampled_coords[:, :, 0] == -1,
                    text_feats=text_feats,
                    text_padding_mask=text_padding_mask,
                )
                valid = sampled_coords[:, :, 0] != -1
                x = ME.SparseTensor(
                    features=sampled_features[valid],
                    coordinates=sampled_coords[valid],
                    coordinate_manager=x.coordinate_manager,
                    tensor_stride=x.tensor_stride,
                    device=x.device,
                )
                prune_logits = self.keep_conv[i - 1](x).F

            x = getattr(self, f"lateral_block_{i}")(x)
            if i == 0:
                visual = getattr(self, f"out_block_{i}")(x)

        text_ctx = text_feats[:, 0]
        fused = self.fuse(visual, text_ctx)
        bbox_preds, cls_preds, points = self._forward_single(fused)
        batch_size = batch_size_of(visual)
        out = {
            "final_logits": [], "final_boxes": [],
        }
        for bi in range(batch_size):
            all_logits = cls_preds[bi].reshape(-1)
            all_boxes = self._bbox_pred_to_bbox(points[bi], bbox_preds[bi])
            if all_logits.numel():
                logits, indices = torch.topk(
                    all_logits, k=1, largest=True, sorted=True)
                boxes = all_boxes[indices]
            else:
                logits = visual.F.new_empty((0,))
                boxes = visual.F.new_empty((0, 6))
            out["final_logits"].append(logits)
            out["final_boxes"].append(boxes)
        return out

    def forward_train(self, x, text_feats, text_attention_mask):
        """返回外部微调 pipeline 消费的可微结构化预测。"""
        return self._base_ft_forward(x, text_feats, text_attention_mask)


    def _prune_inference(self, x, scores, layer_id, prune_threshold=None):
        """
        Prunes the tensor by score thresholding.

        Args:
            x (SparseTensor): Tensor to be pruned.
            scores (SparseTensor): Scores for thresholding.

        Returns:
            SparseTensor: Pruned tensor.
        """
        if scores is None:
            return x  # If scores are missing, skip pruning for this layer

        with torch.no_grad():
            prune_mask = scores.new_zeros((len(scores)), dtype=torch.bool)
            if prune_threshold is not None:
                threshold = prune_threshold
            else:
                threshold = self.prune_threshold[layer_id]
    
            for permutation in x.decomposition_permutations:
                score = scores[permutation].sigmoid()
                score = 1 - score
                mask = score > threshold
                mask = mask.reshape([len(score)])
                prune_mask[permutation[mask]] = True
        # Call the MinkowskiEngine backend physical level 
        # to remove the voxels designated for pruning from the hash map entirely
        if prune_mask.sum() != 0:
            x = self.pruning(x, prune_mask)
        else:
            x = None

        return x

    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        """
        Transform predicted bbox parameters to bbox.

        Args:
            points (Tensor): Final locations of shape (N, 3)
            bbox_pred (Tensor): Predicted bbox parameters of shape (N, 6) or (N, 8).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        x_center = points[:, 0] + bbox_pred[:, 0]
        y_center = points[:, 1] + bbox_pred[:, 1]
        z_center = points[:, 2] + bbox_pred[:, 2]
        base_bbox = torch.stack([
            x_center, y_center, z_center,
            bbox_pred[:, 3], bbox_pred[:, 4], bbox_pred[:, 5]],
            dim=-1)
        # axis-aligned case
        if bbox_pred.shape[1] == 6:
            return base_bbox
        
        # Decode complex 3D bounding boxes with yaw rotation angles
        scale = bbox_pred[:, 3] + bbox_pred[:, 4]  
        q = torch.exp(
            torch.sqrt(
                torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
        alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])

        return torch.stack((x_center, y_center, z_center, 
            scale / (1 + q),
            scale / (1 + q) * q,
            bbox_pred[:, 5] + bbox_pred[:, 4], 
            alpha), dim=-1)

    def _get_bboxes_single(self, bbox_preds, cls_preds, points, img_meta):
        """ 
        Merge predicted features and apply score coarse filtering, transformation,
        and NMS to generate final single-batch 3D boundaries
        """
        scores = torch.cat(cls_preds).sigmoid()
        bbox_preds = torch.cat(bbox_preds)
        points = torch.cat(points)
        max_scores, _ = scores.max(dim=1)

        # Coarse top-k (baseline keeps nms_pre=1 -> single candidate).
        n_coarse = int(self.test_cfg['nms_pre'])
        if len(scores) > n_coarse > 0:
            _, ids = max_scores.topk(n_coarse)
            bbox_preds = bbox_preds[ids]
            scores = scores[ids]
            points = points[ids]

        boxes = self._bbox_pred_to_bbox(points, bbox_preds)

        # Baseline single-candidate path: wrap the top-1 box.
        labels = boxes.new_zeros((len(boxes), ), dtype=torch.long)
        boxes = img_meta['box_type_3d'](boxes, box_dim=6, with_yaw=False, origin=(.5, .5, .5))
        return boxes, scores, labels

    def _get_bboxes(self, bbox_preds, cls_preds, points, img_metas):
        results = []
        for i in range(len(img_metas)):
            result = self._get_bboxes_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i])
            results.append(result)
        return results


    def _forward_single(self, x):
        reg_final = self.bbox_conv(x).features
        reg_distance = torch.exp(reg_final[:, 3:6])
        reg_angle = reg_final[:, 6:]
        bbox_pred = torch.cat((reg_final[:, :3], reg_distance, reg_angle), dim=1)
        scores = self.cls_conv(x)
        cls_pred = scores.features

        bbox_preds, cls_preds, points = [], [], []
        for permutation in x.decomposition_permutations:
            bbox_preds.append(bbox_pred[permutation])
            cls_preds.append(cls_pred[permutation])
            points.append(x.coordinates[permutation][:, 1:]* self.voxel_size)
        return bbox_preds, cls_preds, points


    def forward_test(self, x, 
                    text_feats, 
                    text_attention_mask, 
                    img_metas=None, 
                    pc=None, gt_bboxes=None,
                    sigma_sce=None, tau=None, gate_logits=None,
                    gate_threshold=None):
        """ 
        Pure inference forward pass main loop: 
        Implement multi-layer transposed convolution geometric upsampling, dynamic scene pruning, 
        and bi-directional feature compensation/completion for missing voxels
        """

        inputs = x[1:]
        x = inputs[-1]
        prune_inference = None  
        if img_metas is None:
            # Perform robustness completion for img_metas
            img_metas = [{'box_type_3d': DepthInstance3DBoxes}
                        for _ in range(len(inputs[0].decomposition_permutations))]

        prune_threshold_layer1 = sigma_sce if sigma_sce is not None else self.prune_threshold[1]
        prune_threshold_layer0 = (sigma_sce * (self.prune_threshold[0] / self.prune_threshold[1])) if sigma_sce is not None else self.prune_threshold[0]
        com_threshold = tau if tau is not None else self.com_threshold

        # ---- Pruning / completion behavior diagnostics ----
        diag = {
            "input_voxels": [len(inputs[k]) for k in range(len(inputs))],
            "prune_layer1_before": None, "prune_layer1_after": None, "pruned_all_layer1": False,
            "prune_layer0_before": None, "prune_layer0_after": None, "pruned_all_layer0": False,
            "com_sampled": None, "com_valid": None, "com_kept": None, "com_added": None, "tau": com_threshold,
            "out_voxels": None, "n_boxes": 0, "max_score": 0.0,
        }
        
        for i in range(len(inputs) - 1, -1, -1):
            if i == 1:
                # Execute Layer 1 language-guided pruning
                diag["prune_layer1_before"] = len(x)
                x = self._prune_inference(x, prune_inference, i, prune_threshold=prune_threshold_layer1) 
                if x is not None:
                    diag["prune_layer1_after"] = len(x)
                    # x = self.__getattr__(f'up_block_{i + 1}')(x)
                    x = getattr(self, f'up_block_{i + 1}')(x)
                    x_level_features = self._lateral_features_at(inputs[i], x)
                    x_level = ME.SparseTensor(features=x_level_features,
                                              coordinate_map_key=x.coordinate_map_key,
                                              coordinate_manager=x.coordinate_manager)
                    x = x + x_level
                else:
                    diag["pruned_all_layer1"] = True
                    diag["prune_layer1_after"] = 0
                    logging.warning("[TSPHead] All features pruned at Layer 1. Returning robust empty predictions.")
                    return [(img_meta['box_type_3d'](torch.zeros((0, 6), device=inputs[0].device), 
                            box_dim=6, with_yaw=False), 
                            torch.zeros((0,), device=inputs[0].device), 
                            torch.zeros((0,), dtype=torch.long, device=inputs[0].device))
                            for img_meta in img_metas], 0.0, diag
            
            elif i == 0:
                # Execute Layer 0 dynamic language-guided pruning
                diag["prune_layer0_before"] = len(x)
                x = self._prune_inference(x, prune_inference, i, prune_threshold=prune_threshold_layer0)                
                if x is not None:
                    diag["prune_layer0_after"] = len(x)
                    # x = self.__getattr__(f'up_block_{i + 1}')(x)
                    x = getattr(self, f'up_block_{i + 1}')(x)
                    x_level_features = self._lateral_features_at(inputs[i], x)
                    x_level = ME.SparseTensor(features=x_level_features,
                                              coordinate_map_key=x.coordinate_map_key,
                                              coordinate_manager=x.coordinate_manager)
                    x_ori = x + x_level
                else:
                    diag["pruned_all_layer0"] = True
                    diag["prune_layer0_after"] = 0
                    logging.warning("[TSPHead] All features pruned at Layer 0. Returning robust empty predictions.")
                    return [(img_meta['box_type_3d'](torch.zeros((0, 6), device=inputs[0].device), 
                            box_dim=6, with_yaw=False), 
                            torch.zeros((0,), device=inputs[0].device), 
                            torch.zeros((0,), dtype=torch.long, device=inputs[0].device)) 
                            for img_meta in img_metas], 0.0, diag

                # Shape Completion
                sampled_coords, sampled_features, original_indices = [], [], []
                # Restore multi-scale feature voxels that failed to reconstruct 
                # due to line-of-sight blind spots, strong light refraction, 
                # or partial physical occlusion
                for permutation in inputs[0].decomposition_permutations:
                    original_indices.extend(permutation.cpu().numpy())
                    if len(permutation) > self.num_samples_com:
                        choice = torch.randperm(len(permutation))[:self.num_samples_com]
                        choice = torch.sort(choice).values
                        sampled_features.append(inputs[0].features[permutation][choice])
                        sampled_coords.append(inputs[0].coordinates[permutation][choice])
                    else:
                        padding_size = self.num_samples_com - len(permutation)      
                        padded_features = torch.cat(
                            [inputs[0].features[permutation], torch.zeros((padding_size, inputs[0].features[permutation].shape[1]), 
                                                                  dtype=inputs[0].features.dtype).to(inputs[0].device)], dim=0)  # Pad the boundary with zeros
                        padded_coords = torch.cat(
                            [inputs[0].coordinates[permutation], -torch.ones((padding_size, inputs[0].coordinates[permutation].shape[1]),
                                                                     dtype=inputs[0].coordinates.dtype).to(inputs[0].device)], 
                                                                     dim=0)  # Pad invalid voxel coordinates with -1
                        sampled_features.append(padded_features)
                        sampled_coords.append(padded_coords)
                sampled_features = torch.stack(sampled_features)    # Merge feature list
                sampled_coords = torch.stack(sampled_coords)        # Merge coordinate list
                diag["com_sampled"] = int(sampled_features.shape[0])

                # Utilize cross-modal bi-directional attention awareness 
                # to align sparse coordinates with language features, 
                # generating candidate foreground representations
                sampled_features, text_feats = self.com_trans(
                    vis_feats=sampled_features.contiguous(),
                    pos_feats=self.pos_embed(sampled_coords[:, :, 1:] * self.voxel_size).transpose(1, 2).contiguous(),
                    padding_mask=sampled_coords[:, :, 0] == -1,
                    text_feats=text_feats,
                    text_padding_mask=text_attention_mask)

                # Compute semantic score logits for missing candidates
                com_pred = self.com_cls(sampled_features.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
                valid_mask = sampled_coords[:, :, 0] != -1
                diag["com_valid"] = int(valid_mask.sum().item())
                sampled_features = sampled_features[valid_mask]
                sampled_coords = sampled_coords[valid_mask]
                com_pred = com_pred[valid_mask].squeeze(-1)
                com_mask = com_pred.sigmoid() > com_threshold  # Extract candidates with language scores higher than the threshold
                diag["com_kept"] = int(com_mask.sum().item())
                sampled_features = sampled_features[com_mask]
                sampled_coords = sampled_coords[com_mask]
                # Eliminate duplicate voxel hash points that already exist in space to avoid collisions
                matches = (sampled_coords.unsqueeze(1) == x_ori.coordinates.unsqueeze(0)).all(dim=-1).any(dim=1)
                sampled_features = sampled_features[~matches]
                sampled_coords = sampled_coords[~matches]
                diag["com_added"] = int(sampled_features.shape[0])

                if sampled_features.shape[0] == 0:
                    # MinkowskiEngine returns a rank-1 empty tensor for an empty
                    # coordinate query, which cannot be added to (0, C) features.
                    x = x_ori
                else:
                    # Interpolate and extract original coordinates' potential physical features
                    x_com_features = x.features_at_coordinates(sampled_coords.float())
                    if (
                        x_com_features.shape != sampled_features.shape
                        and x_com_features.shape == sampled_features.transpose(0, 1).shape
                    ):
                        x_com_features = x_com_features.transpose(0, 1).contiguous()
                    if x_com_features.shape != sampled_features.shape:
                        # A query outside x's coordinate map can yield (0, C)
                        # even when one or more completion candidates remain.
                        # Keep the reconstructed trunk instead of creating an
                        # invalid sparse tensor with unequal row counts.
                        logging.warning(
                            "[TSPHead] Skipping incompatible completion features: "
                            f"queried={tuple(x_com_features.shape)}, "
                            f"sampled={tuple(sampled_features.shape)}"
                        )
                        x = x_ori
                    else:
                        # Supplement and recombine voxel information compensated by multi-modal alignment
                        x_com_features = x_com_features + sampled_features
                        # Concatenate and merge main feature trunk and the newly generated recombined voxel features at the tensor level
                        x = ME.SparseTensor(features=torch.cat((x_ori.features, x_com_features), dim=0),
                                            coordinates=torch.cat((x_ori.coordinates, sampled_coords), dim=0),
                                            coordinate_manager=x_ori.coordinate_manager,
                                            tensor_stride=x_ori.tensor_stride,
                                            device=x_ori.device)

            if i > 0:
                sampled_coords, sampled_features = [], []
                len_x = []
                for permutation in x.decomposition_permutations:
                    len_x.append(len(x.coordinates[permutation]))
                max_len_x = int(torch.tensor(len_x).max())
                # Implement highly compatible padding and alignment, 
                # enabling non-uniform batch sparse data to be merged into the keep_trans bi-directional layer for batch operations
                if len(len_x) > 1:
                    for permutation in x.decomposition_permutations:
                        if len(permutation) > max_len_x:
                            choice = torch.randperm(len(permutation))[:max_len_x]
                            choice = torch.sort(choice).values
                            sampled_features.append(x.features[permutation][choice])
                            sampled_coords.append(x.coordinates[permutation][choice])
                        else:
                            padding_size = max_len_x - len(permutation)      
                            padded_features = torch.cat(
                                [x.features[permutation], 
                                torch.zeros((padding_size, x.features[permutation].shape[1]), 
                                dtype=x.features.dtype).to(x.device)], dim=0) 
                            padded_coords = torch.cat(
                                [x.coordinates[permutation], 
                                -torch.ones((padding_size, x.coordinates[permutation].shape[1]),
                                dtype=x.coordinates.dtype).to(x.device)], dim=0)   
                            sampled_features.append(padded_features)
                            sampled_coords.append(padded_coords)
                else:
                    for permutation in x.decomposition_permutations:
                        sampled_features.append(x.features[permutation])
                        sampled_coords.append(x.coordinates[permutation])                        
                sampled_features = torch.stack(sampled_features)
                sampled_coords = torch.stack(sampled_coords)

                # Perform bi-directional feature attention decoupling calculations 
                # on voxel positions and language query features
                sampled_features, text_feats = self.keep_trans[i - 1](
                    vis_feats=sampled_features.contiguous(),
                    pos_feats=self.pos_embed(sampled_coords[:, :, 1:] * self.voxel_size).transpose(1, 2).contiguous(),
                    padding_mask=sampled_coords[:, :, 0] == -1,
                    text_feats=text_feats,
                    text_padding_mask=text_attention_mask)

                valid_mask = sampled_coords[:, :, 0] != -1
                sampled_features = sampled_features[valid_mask]
                sampled_coords = sampled_coords[valid_mask]
                # Re-encapsulate as a sparse tensor
                x = ME.SparseTensor(features=sampled_features, 
                                    coordinates=sampled_coords, 
                                    coordinate_manager=x.coordinate_manager, 
                                    tensor_stride=x.tensor_stride, device=x.device)
                # Compute the pruning value score map for this layer using value prediction convolution
                keep_scores = self.keep_conv[i - 1](x)
                keep_pred = keep_scores.features
                prune_inference = keep_pred

            x = getattr(self, f'lateral_block_{i}')(x)
            if i == 0:
                out = getattr(self, f'out_block_{i}')(x)

        start_time = time.time()
        out = self.fuse(out, text_feats[:, 0])
        diag["out_voxels"] = len(out)
        bbox_pred, cls_pred, point = self._forward_single(out)
        results = self._get_bboxes([bbox_pred], [cls_pred], [point], img_metas)

        # Retain the legacy TSP3D head but make the existence decision a true output short-circuit.
        # The gate is applied after the legacy computation for compatibility 
        # with the legacy single-candidate decoding path.
        if gate_logits is not None:
            threshold = self.gate_threshold if gate_threshold is None else float(gate_threshold)
            gate_prob = torch.sigmoid(gate_logits)
            diag["gate"] = [float(v.detach().cpu()) for v in gate_prob]
            for bi, gp in enumerate(gate_prob):
                if float(gp.detach()) < threshold:
                    results[bi] = empty_prediction(
                        inputs[0].F.device, inputs[0].F.dtype,
                        img_metas[bi]["box_type_3d"])

        if len(results) > 0:
            _, res_scores, _ = results[0]
            diag["n_boxes"] = int(res_scores.shape[0])
            if res_scores.shape[0] > 0:
                diag["max_score"] = float(res_scores.max().item())
        head_time = time.time() - start_time
        return results, head_time, diag

    def forward(self, x, text_feats, text_attention_mask, 
                img_metas=None, pc=None, gt_bboxes=None,
                sigma_sce=None, tau=None, gate_logits=None,
                gate_threshold=None):
        """ Module inference entry wrapper """
        return self.forward_test(x, text_feats, text_attention_mask, img_metas, pc, gt_bboxes,
                                 sigma_sce, tau, gate_logits=gate_logits,
                                 gate_threshold=gate_threshold)
)
