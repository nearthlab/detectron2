import copy
import itertools

import torch
import torch.nn.functional as F

from detectron2.layers import cat, batched_nms
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator, _create_grid_offsets
from detectron2.modeling.proposal_generator.rpn import StandardRPNHead, RPN
from detectron2.onnx.functionalize import register_functionalizer, functionalize
from detectron2.structures import Boxes


@register_functionalizer(StandardRPNHead)
def functionalizeStandardRPNHead(module: StandardRPNHead):
    conv = functionalize(module.conv)
    objectness_logits = functionalize(module.objectness_logits)
    anchor_deltas = functionalize(module.anchor_deltas)

    def forward(features):
        """
        Args:
            features (list[Tensor]): list of feature maps
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = F.relu(conv(x))
            pred_objectness_logits.append(objectness_logits(t))
            pred_anchor_deltas.append(anchor_deltas(t))
        return pred_objectness_logits, pred_anchor_deltas

    return forward


@register_functionalizer(DefaultAnchorGenerator)
def functionalizeDefaultAnchorGenerator(module: DefaultAnchorGenerator):
    strides = module.strides

    def forward(features):
        num_images = len(features[0])
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        cell_anchors = [
            cell_anchor.data.to(device=features[0].device)
            for cell_anchor in module.cell_anchors
        ]
        all_anchors = []
        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            shift_x, shift_y = _create_grid_offsets(size, stride, base_anchors.device)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            all_anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))

        anchors_in_image = []
        for anchors_per_feature_map in all_anchors:
            boxes = Boxes(anchors_per_feature_map)
            anchors_in_image.append(boxes.tensor)

        anchors = [copy.deepcopy(anchors_in_image) for _ in range(num_images)]
        return anchors

    return forward


def clip(boxes: torch.Tensor):
    w, h = 640, 480
    return torch.cat([
        boxes[:, 0].clamp(min=0, max=w),
        boxes[:, 1].clamp(min=0, max=h),
        boxes[:, 2].clamp(min=0, max=w),
        boxes[:, 3].clamp(min=0, max=h)
    ], dim=1)


def nonempty(boxes: torch.Tensor, threshold):
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    keep = (widths > threshold) & (heights > threshold)
    return keep


def find_top_rpn_proposals(
        proposals,
        pred_objectness_logits,
        nms_thresh,
        pre_nms_topk,
        post_nms_topk,
        min_box_side_len
):
    """
    For each feature map, select the `pre_nms_topk` highest scoring proposals,
    apply NMS, clip proposals, and remove small boxes. Return the `post_nms_topk`
    highest scoring proposals among all the feature maps if `training` is True,
    otherwise, returns the highest `post_nms_topk` scoring proposals for each
    feature map.

    Args:
        proposals (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A, 4).
            All proposal predictions on the feature maps.
        pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A).
        images (ImageList): Input images as an :class:`ImageList`.
        nms_thresh (float): IoU threshold to use for NMS
        pre_nms_topk (int): number of top k scoring proposals to keep before applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is per
            feature map.
        post_nms_topk (int): number of top k scoring proposals to keep after applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is total,
            over all feature maps.
        min_box_side_len (float): minimum proposal box side length in pixels (absolute units
            wrt input images).
        training (bool): True if proposals are to be used in training, otherwise False.
            This arg exists only to support a legacy bug; look for the "NB: Legacy bug ..."
            comment.

    Returns:
        proposals (list[Instances]): list of N Instances. The i-th Instances
            stores post_nms_topk object proposals for image i.
    """
    device = proposals[0].device

    # 1. Select top-k anchor for every level and every image
    topk_scores = []  # #lvl Tensor, each of shape N x topk
    topk_proposals = []
    level_ids = []  # #lvl Tensor, each of shape (topk,)
    batch_idx = torch.arange(1, device=device)
    for level_id, proposals_i, logits_i in zip(
            itertools.count(), proposals, pred_objectness_logits
    ):
        Hi_Wi_A = logits_i.shape[1]
        num_proposals_i = min(pre_nms_topk, Hi_Wi_A)

        # sort is faster than topk (https://github.com/pytorch/pytorch/issues/22812)
        # topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)
        logits_i, idx = logits_i.sort(descending=True, dim=1)
        topk_scores_i = logits_i[batch_idx, :num_proposals_i]
        topk_idx = idx[batch_idx, :num_proposals_i]

        # each is N x topk
        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # N x topk x 4

        topk_proposals.append(topk_proposals_i)
        topk_scores.append(topk_scores_i)
        level_ids.append(torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device))

    # 2. Concat all levels together
    topk_scores = cat(topk_scores, dim=1)
    topk_proposals = cat(topk_proposals, dim=1)
    level_ids = cat(level_ids, dim=0)

    # 3. For each image, run a per-level NMS, and choose topk results.
    boxes = clip(topk_proposals[0])
    scores_per_img = topk_scores[0]

    # filter empty boxes
    keep = nonempty(boxes, min_box_side_len)
    lvl = level_ids
    boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], level_ids[keep]

    keep = batched_nms(boxes.tensor, scores_per_img, lvl, nms_thresh)
    # In Detectron1, there was different behavior during training vs. testing.
    # (https://github.com/facebookresearch/Detectron/issues/459)
    # During training, topk is over the proposals from *all* images in the training batch.
    # During testing, it is over the proposals for each image separately.
    # As a result, the training behavior becomes batch-dependent,
    # and the configuration "POST_NMS_TOPK_TRAIN" end up relying on the batch size.
    # This bug is addressed in Detectron2 to make the behavior independent of batch size.
    keep = keep[:post_nms_topk]

    return boxes[keep], scores_per_img[keep]


@register_functionalizer(RPN)
def functionalizeRPN(module: RPN):
    rpn_head = functionalize(module.rpn_head)
    box2box_transform = module.box2box_transform

    def forward(features, anchors):
        pred_objectness_logits, pred_anchor_deltas = rpn_head(features)

        pred_objectness_logits = [
            # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).reshape(1, -1)
            for score in pred_objectness_logits
        ]

        proposals = []
        # Transpose anchors from images-by-feature-maps (N, L) to feature-maps-by-images (L, N)
        anchors = list(zip(anchors))
        # For each feature map
        for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
            B = anchors_i[0].tensor.size(1)
            N, _, Hi, Wi = pred_anchor_deltas_i.shape
            # Reshape: (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N*Hi*Wi*A, B)
            pred_anchor_deltas_i = (
                pred_anchor_deltas_i.view(N, -1, B, Hi, Wi).permute(0, 3, 4, 1, 2).reshape(-1, B)
            )
            # Concatenate all anchors to shape (N*Hi*Wi*A, B)
            # type(anchors_i[0]) is Boxes (B = 4) or RotatedBoxes (B = 5)
            anchors_i = type(anchors_i[0]).cat(anchors_i)
            proposals_i = box2box_transform.apply_deltas(
                pred_anchor_deltas_i, anchors_i.tensor
            )
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals.append(proposals_i.view(N, -1, B))

        proposals = find_top_rpn_proposals(
            proposals,
            pred_objectness_logits,
            module.nms_thresh,
            module.pre_nms_topk[False],
            module.post_nms_topk[False],
            module.min_box_side_len
        )

        return proposals

    return forward
