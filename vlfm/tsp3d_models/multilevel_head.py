import numpy as np
import torch
import time
import logging
from torch import nn

import MinkowskiEngine as ME
from mmcv.ops import nms3d, nms3d_normal
from mmdet3d.structures.bbox_3d import DepthInstance3DBoxes
from .trans_modules import (BiEncoder, BiEncoderLayer, PositionEmbeddingLearned)

def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to giving probability."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


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
                n_classes=1):
        super(TSPHead, self).__init__()
        self.voxel_size = voxel_size
        self.volume_threshold = volume_threshold
        self.prune_threshold = prune_threshold
        self.com_threshold = com_threshold
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
        ])
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

        # for i in range(len(in_channels)):
        #     if i > 0:
        #         self.__setattr__(
        #             f'up_block_{i}',
        #             self.make_up_block(in_channels[i], in_channels[i - 1], generative=True))
        #     self.__setattr__(
        #         f'lateral_block_{i}',
        #         self.make_block(in_channels[i], in_channels[i]))
        #     if i == 0:
        #         self.__setattr__(
        #             f'out_block_{i}',
        #             self.make_block(in_channels[i], out_channels))
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
                
        self.fuse = MinkowskiFeatureFusionBlock(128, 128, 128)

    def _prune_inference(self, x, scores, layer_id, prune_threshold=None):
        """Prunes the tensor by score thresholding.

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


    def _nms(self, bboxes, scores, img_meta):
            """Multi-class nms for a single scene.
            Args:
                bboxes (Tensor): Predicted boxes of shape (N_boxes, 6) or
                    (N_boxes, 7).
                scores (Tensor): Predicted scores of shape (N_boxes, N_classes).
                img_meta (dict): Scene meta data.
            Returns:
                Tensor: Predicted bboxes.
                Tensor: Predicted scores.
                Tensor: Predicted labels.
            """
            n_classes = scores.shape[1]
            yaw_flag = bboxes.shape[1] == 7
            nms_bboxes, nms_scores, nms_labels = [], [], []
            for i in range(n_classes):
                ids = scores[:, i] > self.test_cfg['score_thr']
                if not ids.any():
                    continue

                class_scores = scores[ids, i]
                class_bboxes = bboxes[ids]
                print(f"[NMS] class_bboxes type={type(class_bboxes).__name__}, shape={getattr(class_bboxes, 'shape', 'N/A')}")
                if yaw_flag:
                    nms_function = nms3d
                else:
                    print(f"[NMS] before cat: class_bboxes={class_bboxes}, slice={class_bboxes[:, :1]}")
                    class_bboxes = torch.cat(
                        (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                        dim=1)
                    nms_function = nms3d_normal

                print(f"[NMS] input bboxes={class_bboxes}, scores={class_scores}")
                nms_ids = nms_function(class_bboxes, class_scores,
                                    self.test_cfg['iou_thr'])
                print(f"[NMS] nms_ids={nms_ids}")
                nms_bboxes.append(class_bboxes[nms_ids])
                nms_scores.append(class_scores[nms_ids])
                nms_labels.append(
                    bboxes.new_full(
                        class_scores[nms_ids].shape, i, dtype=torch.long))

            if len(nms_bboxes):
                nms_bboxes = torch.cat(nms_bboxes, dim=0)
                nms_scores = torch.cat(nms_scores, dim=0)
                nms_labels = torch.cat(nms_labels, dim=0)
            else:
                nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
                nms_scores = bboxes.new_zeros((0, ))
                nms_labels = bboxes.new_zeros((0, ))

            if yaw_flag:
                box_dim = 7
                with_yaw = True
            else:
                box_dim = 6
                with_yaw = False
                nms_bboxes = nms_bboxes[:, :6]
            nms_bboxes = img_meta['box_type_3d'](
                nms_bboxes,
                box_dim=box_dim,
                with_yaw=with_yaw,
                origin=(.5, .5, .5))

            return nms_bboxes, nms_scores, nms_labels

    def init_weights(self):
        nn.init.normal_(self.bbox_conv.kernel, std=.01)
        nn.init.normal_(self.cls_conv.kernel, std=.01)
        nn.init.constant_(self.cls_conv.bias, bias_init_with_prob(.01))

        for i in range(len(self.keep_conv)):
            nn.init.normal_(self.keep_conv[i].kernel, std=.01)

        for n, m in self.named_modules():
            if ('bbox_conv' not in n) and ('cls_conv' not in n) \
                and ('keep_conv' not in n) and ('loss' not in n):
                if isinstance(m, ME.MinkowskiConvolution):
                    ME.utils.kaiming_normal_(
                        m.kernel, mode='fan_out', nonlinearity='relu')

                if isinstance(m, ME.MinkowskiBatchNorm):
                    nn.init.constant_(m.bn.weight, 1)
                    nn.init.constant_(m.bn.bias, 0)       


    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        """Transform predicted bbox parameters to bbox.
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
        Merge predicted features and apply score coarse filtering, transformation, and NMS to generate final single-batch 3D boundaries
        """
        scores = torch.cat(cls_preds).sigmoid()
        bbox_preds = torch.cat(bbox_preds)
        points = torch.cat(points)
        max_scores, _ = scores.max(dim=1)

        # Preliminarily retain top-N geometric candidates based on maximum class confidence
        if len(scores) > self.test_cfg['nms_pre'] > 0:
            _, ids = max_scores.topk(self.test_cfg['nms_pre'])  
            bbox_preds = bbox_preds[ids]
            scores = scores[ids]
            points = points[ids]

        boxes = self._bbox_pred_to_bbox(points, bbox_preds)
        # labels = boxes.new_zeros((1, ),dtype=int)
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
                    sigma_sce=None, tau=None):
        """ 
        Pure inference forward pass main loop: 
        Implement multi-layer transposed convolution geometric upsampling, dynamic scene pruning, 
        and bi-directional feature compensation/completion for missing voxels """

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
        
        for i in range(len(inputs) - 1, -1, -1):
            if i == 1:
                # Execute Layer 1 language-guided pruning
                x = self._prune_inference(x, prune_inference, i, prune_threshold=prune_threshold_layer1) 
                if x is not None:
                    # x = self.__getattr__(f'up_block_{i + 1}')(x)
                    x = getattr(self, f'up_block_{i + 1}')(x)
                    coords = x.coordinates.float()
                    x_level_features = inputs[i].features_at_coordinates(coords)
                    x_level = ME.SparseTensor(features=x_level_features,
                                              coordinate_map_key=x.coordinate_map_key,
                                              coordinate_manager=x.coordinate_manager)
                    x = x + x_level
                else:
                    logging.warning("[TSPHead] All features pruned at Layer 1. Returning robust empty predictions.")
                    return [(img_meta['box_type_3d'](torch.zeros((0, 6), device=inputs[0].device), 
                            box_dim=6, with_yaw=False), 
                            torch.zeros((0,), device=inputs[0].device), 
                            torch.zeros((0,), dtype=torch.long, device=inputs[0].device))
                            for img_meta in img_metas], 0.0
            
            elif i == 0:
                # Execute Layer 0 dynamic language-guided pruning
                x = self._prune_inference(x, prune_inference, i, prune_threshold=prune_threshold_layer0)                
                if x is not None:
                    # x = self.__getattr__(f'up_block_{i + 1}')(x)
                    x = getattr(self, f'up_block_{i + 1}')(x)
                    coords = x.coordinates.float()
                    x_level_features = inputs[i].features_at_coordinates(coords)
                    x_level = ME.SparseTensor(features=x_level_features,
                                              coordinate_map_key=x.coordinate_map_key,
                                              coordinate_manager=x.coordinate_manager)
                    x_ori = x + x_level
                else:
                    logging.warning("[TSPHead] All features pruned at Layer 0. Returning robust empty predictions.")
                    return [(img_meta['box_type_3d'](torch.zeros((0, 6), device=inputs[0].device), 
                            box_dim=6, with_yaw=False), 
                            torch.zeros((0,), device=inputs[0].device), 
                            torch.zeros((0,), dtype=torch.long, device=inputs[0].device)) 
                            for img_meta in img_metas], 0.0
        
                sampled_coords, sampled_features, original_indices = [], [], []
                # Restore multi-scale feature voxels that failed to reconstruct 
                # due to line-of-sight blind spots, strong light refraction, or partial physical occlusion
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
                
                # Utilize cross-modal bi-directional attention awareness 
                # to align sparse coordinates with language features, generating candidate foreground representations
                sampled_features, text_feats = self.com_trans(
                    vis_feats=sampled_features.contiguous(),
                    pos_feats=self.pos_embed(sampled_coords[:, :, 1:] * self.voxel_size).transpose(1, 2).contiguous(),
                    padding_mask=sampled_coords[:, :, 0] == -1,
                    text_feats=text_feats,
                    text_padding_mask=text_attention_mask)

                # Compute semantic score logits for missing candidates
                com_pred = self.com_cls(sampled_features.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
                valid_mask = sampled_coords[:, :, 0] != -1
                sampled_features = sampled_features[valid_mask]
                sampled_coords = sampled_coords[valid_mask]
                com_pred = com_pred[valid_mask].squeeze(-1)
                com_mask = com_pred.sigmoid() > com_threshold  # Extract candidates with language scores higher than the threshold
                sampled_features = sampled_features[com_mask]
                sampled_coords = sampled_coords[com_mask]
                # Eliminate duplicate voxel hash points that already exist in space to avoid collisions
                matches = (sampled_coords.unsqueeze(1) == x_ori.coordinates.unsqueeze(0)).all(dim=-1).any(dim=1)
                sampled_features = sampled_features[~matches]
                sampled_coords = sampled_coords[~matches]

                # Interpolate and extract original coordinates' potential physical features
                x_com_features = x.features_at_coordinates(sampled_coords.float())
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
        bbox_pred, cls_pred, point = self._forward_single(out)
        results = self._get_bboxes([bbox_pred], [cls_pred], [point], img_metas)
        head_time = time.time() - start_time
        return results, head_time

    def forward(self, x, text_feats, text_attention_mask, 
                img_metas=None, pc=None, gt_bboxes=None,
                sigma_sce=None, tau=None):
        """ Module inference entry wrapper """
        return self.forward_test(x, text_feats, text_attention_mask, img_metas, pc, gt_bboxes, sigma_sce, tau)
