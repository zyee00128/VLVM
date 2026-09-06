# ------------------------------------------------------------------------
# TSP3D TSPHead fine-tune patch (training-only).
# Methods ported verbatim from the original TSP3D training repo
# (github.com/GWxuan/TSP3D, models/multilevel_head.py) and monkey-patched onto
# the VLVM inference TSPHead at import time. The inference head file
# (multilevel_head.py) and its online behaviour are untouched.
#
# Import this module (e.g. in train.py) to enable TSPHead.forward_train.
# ------------------------------------------------------------------------
import numpy as np
import time
import torch
from torch import nn

import MinkowskiEngine as ME
from mmdet.models.losses import FocalLoss
from mmdet3d.structures.bbox_3d import rotation_3d_in_axis

from .axis_aligned_iou_loss import AxisAlignedIoULoss2
from .multilevel_head import TSPHead, bias_init_with_prob


def _forward_train_body(self, x,text_feats, text_attention_mask, gt_bboxes, gt_labels, gt_all_bbox_new, auxi_bbox, img_metas,pc=None):
    bboxes_level = []
    bboxes_state = []
    if self.assign_type == 'volume':
        for idx in range(len(img_metas)):
 
            bbox_all = gt_all_bbox_new[idx]
            bbox_level = torch.ones([bbox_all.shape[0], 1])
            bbox_state_all = torch.cat((bbox_level, bbox_all.gravity_center, bbox_all.tensor[:, 3:]), dim=1)

            bbox_gt = gt_bboxes[idx]
            bbox_state_gt = torch.cat((bbox_gt.gravity_center, bbox_gt.tensor[:, 3:]), dim=1)                
            bbox_auxi = auxi_bbox[idx]
            bbox_state_auxi = torch.cat((bbox_auxi.gravity_center, bbox_auxi.tensor[:, 3:]), dim=1)
            bbox_state_auxi_gt = torch.cat((bbox_state_gt, bbox_state_auxi), dim=0)
            bbox_level = torch.zeros([bbox_state_auxi_gt.shape[0], 1])
            bbox_state_auxi_gt = torch.cat((bbox_level, bbox_state_auxi_gt), dim=1)
            
            bbox_state = torch.cat((bbox_state_all, bbox_state_auxi_gt), dim=0)
            
            bboxes_level.append(bbox_state[:,[0]])
            bboxes_state.append(bbox_state)
    
    bbox_preds, cls_preds, points = [], [], []
    keep_gts = []
    keep_preds, prune_masks = [], []
    prune_mask = None
    inputs = x[1:]
    x = inputs[-1]
    for i in range(len(inputs) - 1, -1, -1): # 2,1,0
        if i ==1 :  #  1,0         
            prune_mask = self._get_keep_voxel(x, i + 2, bboxes_state, img_metas) 

            keep_gt = []
            for permutation in x.decomposition_permutations:
                keep_gt.append(prune_mask[permutation])
            keep_gts.append(keep_gt)
            x = self.__getattr__(f'up_block_{i + 1}')(x)
            coords = x.coordinates.float()
            x_level_features = inputs[i].features_at_coordinates(coords)  # select for partial addition
            x_level = ME.SparseTensor(features=x_level_features,
                                      coordinate_map_key=x.coordinate_map_key,
                                    coordinate_manager=x.coordinate_manager)
            x = x + x_level
            x = self._prune_training(x, prune_training_keep, i) 
        elif i == 0:
            prune_mask = self._get_keep_voxel(x, i + 2, bboxes_state, img_metas) 
            keep_gt = []
            for permutation in x.decomposition_permutations:
                keep_gt.append(prune_mask[permutation])
            keep_gts.append(keep_gt)
            x = self.__getattr__(f'up_block_{i + 1}')(x)
            prune_threshold_ = np.random.randint(self.random_prune_threshold[0], self.random_prune_threshold[1])
            self.pts_prune_threshold = (prune_threshold_,self.pts_prune_threshold[1])
            x = self._prune_training(x, prune_training_keep, i)
            coords = x.coordinates.float()
            x_level_features = inputs[i].features_at_coordinates(coords)  # select for partial addition
            x_level = ME.SparseTensor(features=x_level_features,
                                      coordinate_map_key=x.coordinate_map_key,
                                    coordinate_manager=x.coordinate_manager)
            x_ori = x + x_level
            
            
            sampled_coords,sampled_features, original_indices = [],[],[]
            
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
                                                              dtype=inputs[0].features.dtype).to(inputs[0].device)], dim=0) 
                    padded_coords = torch.cat(
                        [inputs[0].coordinates[permutation], -torch.ones((padding_size, inputs[0].coordinates[permutation].shape[1]),
                                                                 dtype=inputs[0].coordinates.dtype).to(inputs[0].device)], 
                                                                 dim=0)  
                    sampled_features.append(padded_features)
                    sampled_coords.append(padded_coords)
            sampled_features = torch.stack(sampled_features)
            sampled_coords = torch.stack(sampled_coords)
            sampled_features, text_feats = self.com_trans(
                vis_feats=sampled_features.contiguous(),
                pos_feats=self.pos_embed(sampled_coords[:,:,1:]*self.voxel_size).transpose(1, 2).contiguous(),
                padding_mask=sampled_coords[:, :,0] == -1,
                text_feats=text_feats,
                text_padding_mask=text_attention_mask)
            
            com_pred = self.com_cls(sampled_features.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
            valid_mask = sampled_coords[:, :,0] != -1
            com_pred_training = [com_pred[k][valid_mask[k]] for k in range(len(com_pred))]
            com_coords_training = [sampled_coords[k][valid_mask[k]][:,1:]*self.voxel_size for k in range(len(com_pred))]
            sampled_features = sampled_features[valid_mask]
            sampled_coords = sampled_coords[valid_mask]
            com_pred = com_pred[valid_mask].squeeze(-1)
            com_mask = com_pred.sigmoid() > self.com_threshold
            sampled_features = sampled_features[com_mask]
            sampled_coords = sampled_coords[com_mask]                
            matches = (sampled_coords.unsqueeze(1) == x_ori.coordinates.unsqueeze(0)).all(dim=-1).any(dim=1)
            sampled_features = sampled_features[~matches]
            sampled_coords = sampled_coords[~matches]                   
            
            x_com_features = x.features_at_coordinates(sampled_coords.float())     
            x_com_features = x_com_features + sampled_features           
            x = ME.SparseTensor(features=torch.cat((x_ori.features,x_com_features),dim=0), 
                                coordinates=torch.cat((x_ori.coordinates,sampled_coords),dim=0), 
                                coordinate_manager=x_ori.coordinate_manager, tensor_stride=x_ori.tensor_stride, device=x_ori.device)
        if i > 0: # 2,1
            sampled_coords,sampled_features, original_indices = [],[],[]
            prune_mask = torch.zeros(x.shape[0], dtype=torch.bool).to(x.device)
            for permutation in x.decomposition_permutations:
                original_indices.extend(permutation.cpu().numpy())
                if len(permutation) > self.num_samples[i-1]:
                    choice = torch.randperm(len(permutation))[:self.num_samples[i-1]]
                    choice = torch.sort(choice).values
                    sampled_features.append(x.features[permutation][choice])
                    sampled_coords.append(x.coordinates[permutation][choice])
                    prune_mask[permutation[choice]] = True
                else:
                    padding_size = self.num_samples[i-1] - len(permutation)      
                    padded_features = torch.cat(
                        [x.features[permutation], torch.zeros((padding_size, x.features[permutation].shape[1]), 
                                                              dtype=x.features.dtype).to(x.device)], dim=0) 
                    padded_coords = torch.cat(
                        [x.coordinates[permutation], -torch.ones((padding_size, x.coordinates[permutation].shape[1]),
                                                                 dtype=x.coordinates.dtype).to(x.device)], 
                                                                 dim=0)  
                    sampled_features.append(padded_features)
                    sampled_coords.append(padded_coords)
                    prune_mask[permutation] = True
            sampled_features = torch.stack(sampled_features)
            sampled_coords = torch.stack(sampled_coords)
            sampled_features, text_feats = self.keep_trans[i-1](
                vis_feats=sampled_features.contiguous(),
                pos_feats=self.pos_embed(sampled_coords[:,:,1:]*self.voxel_size).transpose(1, 2).contiguous(),
                padding_mask=sampled_coords[:, :,0] == -1,
                text_feats=text_feats,
                text_padding_mask=text_attention_mask)
            
            valid_mask = sampled_coords[:, :,0] != -1
            sampled_features = sampled_features[valid_mask]
            sampled_coords = sampled_coords[valid_mask]
            
            x = ME.SparseTensor(features=sampled_features, coordinates=sampled_coords, 
                                coordinate_manager=x.coordinate_manager, tensor_stride=x.tensor_stride, device=x.device)
            keep_scores = self.keep_conv[i-1](x) # 1 MLP
            prune_training_keep = ME.SparseTensor(
                                -keep_scores.features,
                                coordinate_map_key=keep_scores.coordinate_map_key,
                                coordinate_manager=keep_scores.coordinate_manager)
            
 
            keep_pred = keep_scores.features
            prune_inference = keep_pred
            keeps = []

            try:
                for permutation in x.decomposition_permutations:
                    keeps.append(keep_pred[permutation])
            except:
                raise RuntimeError('pdb hit in ft body')
            keep_preds.append(keeps)
            
        x = self.__getattr__(f'lateral_block_{i}')(x)
        if i == 0:
            out = self.__getattr__(f'out_block_{i}')(x)
    out = self.fuse(out, text_feats[:, 0])
    bbox_pred, cls_pred, point = self._forward_single(out)
    return [bbox_pred], [cls_pred], [point], keep_preds[::-1], keep_gts[::-1], bboxes_level, com_pred_training, com_coords_training


def make_down_block(in_channels, out_channels):
    return nn.Sequential(
        ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3,
                                stride=2, dimension=3),
        ME.MinkowskiBatchNorm(out_channels),
        ME.MinkowskiReLU(inplace=True))


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


def _prune_training(self, x, scores, layer_id):
    """Prunes the tensor by score thresholding.

    Args:
        x (SparseTensor): Tensor to be pruned.
        scores (SparseTensor): Scores for thresholding.

    Returns:
        SparseTensor: Pruned tensor.
    """

    with torch.no_grad():
        coordinates = x.C.float()
        interpolated_scores = scores.features_at_coordinates(coordinates)
        prune_mask = interpolated_scores.new_zeros(
            (len(interpolated_scores)), dtype=torch.bool)
        for permutation in x.decomposition_permutations:
            score = interpolated_scores[permutation]
            mask = score.new_zeros((len(score)), dtype=torch.bool)
            topk = min(len(score), self.pts_prune_threshold[layer_id])
            ids = torch.topk(score.squeeze(1), topk, sorted=False).indices
            mask[ids] = True
            prune_mask[permutation[mask]] = True
    x = self.pruning(x, prune_mask)
    return x


@torch.no_grad()
def _get_keep_voxel(self, input, cur_level, bboxes_state, input_metas):
    bboxes = []
    for size in range(len(input_metas)):
        bboxes.append([])
    for idx in range(len(input_metas)):
        for n in range(len(bboxes_state[idx])):
            if bboxes_state[idx][n][0] < (cur_level - 1):    
                bboxes[idx].append(bboxes_state[idx][n])
    idx = 0
    mask = []
    l0 = self.voxel_size * 2 ** 2  # pool  True :2**3  False:2**2
    for idx, permutation in enumerate(input.decomposition_permutations):
        point = input.coordinates[permutation][:, 1:]* self.voxel_size
        if len(bboxes[idx]) != 0:
            point = input.coordinates[permutation][:, 1:]* self.voxel_size
            boxes = bboxes[idx]
            level = 3
            bboxes_level = [[] for _ in range(level)]
            for n in range(len(boxes)):
                for l in range(level):
                    if boxes[n][0] == l:
                        bboxes_level[l].append(boxes[n])
            inside_box_conditions = torch.zeros((len(permutation)), dtype=torch.bool).to(point.device)
            for l in range(level):
                if len(bboxes_level[l]) != 0:
                    point_l = point.unsqueeze(1).expand(len(point), len(bboxes_level[l]), 3)
                    boxes_l = torch.cat(bboxes_level[l]).reshape([-1, 8]).to(point.device)
                    boxes_l = boxes_l.expand(len(point), len(bboxes_level[l]), 8)
                    shift = torch.stack(
                        (point_l[..., 0] - boxes_l[..., 1], point_l[..., 1] - boxes_l[..., 2],
                        point_l[..., 2] - boxes_l[..., 3]),
                        dim=-1).permute(1, 0, 2)
                    shift = rotation_3d_in_axis(
                        shift, -boxes_l[0, :, 7], axis=2).permute(1, 0, 2)
                    centers = boxes_l[..., 1:4] + shift
                    up_level_l = self.r[cur_level-2] 
                    dx_min = centers[..., 0] - boxes_l[..., 1] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2  
                    dx_max = boxes_l[..., 1] - centers[..., 0] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2 
                    dy_min = centers[..., 1] - boxes_l[..., 2] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2  
                    dy_max = boxes_l[..., 2] - centers[..., 1] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2
                    dz_min = centers[..., 2] - boxes_l[..., 3] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2  
                    dz_max = boxes_l[..., 3] - centers[..., 2] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2


                    distance = torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max), dim=-1)
                    inside_box_condition = distance.min(dim=-1).values > 0
                    inside_box_condition = inside_box_condition.sum(dim=1)
                    inside_box_condition = inside_box_condition >= 1
                    inside_box_conditions += inside_box_condition
            mask.append(inside_box_conditions)
        else:
            inside_box_conditions = torch.zeros((len(permutation)), dtype=torch.bool).to(point.device)
            mask.append(inside_box_conditions)

    prune_mask = torch.cat(mask)
    prune_mask = prune_mask.to(input.device)
    return prune_mask


def _bbox_to_loss(bbox):
    """Transform box to the axis-aligned or rotated iou loss format.
    Args:
        bbox (Tensor): 3D box of shape (N, 6) or (N, 7).
    Returns:
        Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
    """
    # rotated iou loss accepts (x, y, z, w, h, l, heading)
    if bbox.shape[-1] != 6:
        return bbox

    # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
    return torch.stack(
        (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
         bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
         bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
        dim=-1)


def _loss_single(self,
                 bbox_preds,
                 cls_preds,
                 points,
                 gt_bboxes,
                 gt_labels,
                 img_meta,
                 com_pred,com_coords):
    assigned_ids = self.assigner.assign(points, gt_bboxes, gt_labels, img_meta)
    bbox_preds = torch.cat(bbox_preds)
    cls_preds = torch.cat(cls_preds)
    points = torch.cat(points)

    # cls loss
    n_classes = cls_preds.shape[1]
    pos_mask = assigned_ids >= 0

    if len(gt_labels) > 0:
        cls_targets = torch.where(pos_mask, gt_labels[assigned_ids], n_classes)
    else:
        cls_targets = gt_labels.new_full((len(pos_mask),), n_classes)

    cls_loss = self.cls_loss(cls_preds, cls_targets)
    
    assigned_ids_com = self.assigner.assign([com_coords], gt_bboxes, gt_labels, img_meta)
    # cls loss
    pos_mask_com = assigned_ids_com >= 0

    if len(gt_labels) > 0:
        cls_targets = torch.where(pos_mask_com, gt_labels[assigned_ids_com], n_classes)
    else:
        cls_targets = gt_labels.new_full((len(pos_mask_com),), n_classes)

    com_loss = self.com_loss(com_pred, cls_targets)

    # bbox loss
    pos_bbox_preds = bbox_preds[pos_mask]
    if pos_mask.sum() > 0:
        pos_points = points[pos_mask]
        pos_bbox_preds = bbox_preds[pos_mask]
        bbox_targets = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
        pos_bbox_targets = bbox_targets.to(points.device)[assigned_ids][pos_mask]
        if pos_bbox_preds.shape[1] == 6:
            pos_bbox_targets = pos_bbox_targets[:, :6]
        
        bbox_loss = self.bbox_loss(
            self._bbox_to_loss(self._bbox_pred_to_bbox(pos_points, pos_bbox_preds)),
            self._bbox_to_loss(pos_bbox_targets))            
    else:
        bbox_loss = None
    return bbox_loss, cls_loss, pos_mask, com_loss, pos_mask_com


def _loss(self, bbox_preds, cls_preds, points, gt_bboxes, gt_labels, img_metas, 
          keep_preds, keep_gts, bboxes_level, com_pred_training, com_coords_training):
    bbox_losses, cls_losses, pos_masks, com_losses, pos_masks_com = [], [], [], [], []

    #keep loss
    keep_losses = 0
    for i in range(len(img_metas)):
        k_loss = 0
        keep_pred = [x[i] for x in keep_preds]
        keep_gt = [x[i] for x in keep_gts]
        for j in range(len(keep_preds)):
            pred = keep_pred[j]
            gt = (keep_gt[j]).long()

            if gt.sum() != 0:
                keep_loss = self.keep_loss(pred, gt, avg_factor=gt.sum())
                k_loss = torch.mean(keep_loss) / 3 + k_loss
            else:
                keep_loss = self.keep_loss(pred, gt, avg_factor=len(gt))  
                k_loss = torch.mean(keep_loss) / 3 + k_loss

        keep_losses = keep_losses + k_loss

    for i in range(len(img_metas)):
        bbox_loss, cls_loss, pos_mask, com_loss,pos_mask_com = self._loss_single(
            bbox_preds=[x[i] for x in bbox_preds],
            cls_preds=[x[i] for x in cls_preds],
            points=[x[i] for x in points],
            img_meta=img_metas[i],
            gt_bboxes=gt_bboxes[i],
            gt_labels=gt_labels[i],
            com_pred = com_pred_training[i],
            com_coords = com_coords_training[i])
        if bbox_loss is not None:
            bbox_losses.append(bbox_loss)
        cls_losses.append(cls_loss)
        com_losses.append(com_loss)
        pos_masks.append(pos_mask)
        pos_masks_com.append(pos_mask_com)

    return dict(
        bbox_loss=self.bbox_loss_weight * torch.mean(torch.cat(bbox_losses)),
        cls_loss=torch.sum(torch.cat(cls_losses)) / torch.sum(torch.cat(pos_masks)),
        keep_loss=self.keep_loss_weight * keep_losses / len(img_metas),
        com_loss=torch.sum(torch.cat(com_losses)) / torch.sum(torch.cat(pos_masks_com))) 


def forward_train(self, x, text_feats, text_attention_mask, gt_bboxes, gt_labels, gt_all_bbox_new, auxi_bbox, img_metas,pc=None):
    bbox_preds, cls_preds, points, keep_preds, keep_gts, bboxes_level, com_pred_training, com_coords_training = \
        self._forward_train_body(x, text_feats, text_attention_mask, gt_bboxes, gt_labels, gt_all_bbox_new, auxi_bbox, img_metas, pc)

    return self._loss(bbox_preds, cls_preds, points,
                      gt_bboxes, gt_labels, img_metas, keep_preds, keep_gts, bboxes_level,
                      com_pred_training, com_coords_training)


class TR3DAssigner:
    def __init__(self, top_pts_threshold, label2level):
        # top_pts_threshold: per box
        # label2level: list of len n_classes
        #     scannet: [0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0]
        #     sunrgbd: [1, 1, 1, 0, 0, 1, 0, 0, 1, 0]
        #       s3dis: [1, 0, 1, 1, 0]
        self.top_pts_threshold = top_pts_threshold
        self.label2level = label2level

    @torch.no_grad()
    def assign(self, points, gt_bboxes, gt_labels, img_meta):
        # -> object id or -1 for each point
        float_max = points[0].new_tensor(1e8)
        levels = torch.cat([points[i].new_tensor(i, dtype=torch.long).expand(len(points[i]))
                            for i in range(len(points))])
        points = torch.cat(points)
        n_points = len(points)
        n_boxes = len(gt_bboxes)

        if len(gt_labels) == 0:
            return gt_labels.new_full((n_points,), -1)

        boxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
        boxes = boxes.to(points.device).expand(n_points, n_boxes, 7)
        points = points.unsqueeze(1).expand(n_points, n_boxes, 3)

        # condition 1: fix level for label
        label2level = gt_labels.new_tensor(self.label2level)
        label_levels = label2level[gt_labels].unsqueeze(0).expand(n_points, n_boxes)
        point_levels = torch.unsqueeze(levels, 1).expand(n_points, n_boxes)
        level_condition = label_levels == point_levels

        # condition 2: keep topk location per box by center distance
        center = boxes[..., :3]
        center_distances = torch.sum(torch.pow(center - points, 2), dim=-1)
        center_distances = torch.where(level_condition, center_distances, float_max)
        topk_distances = torch.topk(center_distances,
                                    min(self.top_pts_threshold + 1, len(center_distances)),
                                    largest=False, dim=0).values[-1]
        topk_condition = center_distances < topk_distances.unsqueeze(0)

        # condition 3.0: only closest object to point
        center_distances = torch.sum(torch.pow(center - points, 2), dim=-1)
        _, min_inds_ = center_distances.min(dim=1)

        # condition 3: min center distance to box per point
        center_distances = torch.where(topk_condition, center_distances, float_max)
        min_values, min_ids = center_distances.min(dim=1)
        min_inds = torch.where(min_values < float_max, min_ids, -1)
        min_inds = torch.where(min_inds == min_inds_, min_ids, -1)

        return min_inds


# ============================================================================
# Patch the training methods + training state onto TSPHead.
# ============================================================================
_ORIG_INIT = TSPHead.__init__
def _train_init(self, *args, **kwargs):
    _ORIG_INIT(self, *args, **kwargs)
    # Training-only state (parameterless modules -> no state-dict impact, so the
    # frozen inference head and online checkpoint loading are unaffected).
    self.assign_type = "volume"
    self.pts_prune_threshold = (1200, 4000)
    self.random_prune_threshold = (1200, 4000)
    self.num_samples = (3200, 320)
    self.r = (13, 13)
    self.keep_loss_weight = 1.0
    self.bbox_loss_weight = 1.0
    self.assigner = TR3DAssigner(top_pts_threshold=32, label2level=[0])
    self.bbox_loss = AxisAlignedIoULoss2(mode="diou", reduction="none")
    self.cls_loss = FocalLoss(reduction="none")
    self.com_loss = FocalLoss(reduction="none")
    self.keep_loss = FocalLoss(reduction="mean", use_sigmoid=True)
    self.train_cfg = None


TSPHead.__init__ = _train_init
TSPHead._forward_train_body = _forward_train_body
TSPHead.forward_train = forward_train
TSPHead._loss = _loss
TSPHead._loss_single = _loss_single
TSPHead._bbox_to_loss = staticmethod(_bbox_to_loss)
TSPHead._prune_training = _prune_training
TSPHead._get_keep_voxel = _get_keep_voxel
TSPHead.make_down_block = staticmethod(make_down_block)
TSPHead.init_weights = init_weights

if __name__ == "__main__":
    print("multilevel_head_ft patch module loaded.")
