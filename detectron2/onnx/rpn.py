import itertools

import torch
import torch.nn.functional as F

from detectron2.layers import cat, batched_nms
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator, _create_grid_offsets
from detectron2.modeling.proposal_generator.rpn import StandardRPNHead, RPN
from detectron2.onnx.functionalize import register_functionalizer, functionalize


def boxinit(boxes: torch.Tensor):
    device = boxes.device if isinstance(boxes, torch.Tensor) else torch.device("cpu")
    boxes = torch.as_tensor(boxes, dtype=torch.float32, device=device)
    if boxes.numel() == 0:
        boxes = torch.zeros(0, 4, dtype=torch.float32, device=device)
    return boxes


def boxclip(boxes: torch.Tensor):
    w, h = 640.0, 480.0
    return torch.transpose(torch.cat([
        boxes[:, 0].clamp(min=0, max=w).reshape(1, boxes.shape[0]),
        boxes[:, 1].clamp(min=0, max=h).reshape(1, boxes.shape[0]),
        boxes[:, 2].clamp(min=0, max=w).reshape(1, boxes.shape[0]),
        boxes[:, 3].clamp(min=0, max=h).reshape(1, boxes.shape[0])
    ], dim=1).reshape((1, *boxes.shape[::-1])), 1, 2)[0]


def boxnonempty(boxes: torch.Tensor, threshold):
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    keep = (widths > threshold) & (heights > threshold)
    return keep


def boxcat(boxes):
    cat_boxes = cat([b for b in boxes], dim=0)
    return cat_boxes


def apply_deltas(deltas, boxes, weights, scale_clamp):
    """
    deltas (Tensor): (Batch, N, k*4)
    boxes (Tensor): (Batch, N, k*4)
    """
    widths = boxes[:, :, 2] - boxes[:, :, 0]
    heights = boxes[:, :, 3] - boxes[:, :, 1]
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    d = torch.transpose(deltas.reshape(deltas.size(0), deltas.size(1), deltas.size(2) // 4, 4), 2, 3)
    dx = d[:, :, 0, :] / wx
    dy = d[:, :, 1, :] / wy
    dw = d[:, :, 2, :] / ww
    dh = d[:, :, 3, :] / wh

    # Prevent sending too large values into torch.exp()
    dw = torch.clamp(dw, max=scale_clamp)
    dh = torch.clamp(dh, max=scale_clamp)

    pred_ctr_x = dx * widths[:, :, None] + ctr_x[:, :, None]
    pred_ctr_y = dy * heights[:, :, None] + ctr_y[:, :, None]
    pred_w = torch.exp(dw) * widths[:, :, None]
    pred_h = torch.exp(dh) * heights[:, :, None]

    pred_boxes = torch.transpose(
        torch.cat(
            [
                pred_ctr_x - 0.5 * pred_w,
                pred_ctr_y - 0.5 * pred_h,
                pred_ctr_x + 0.5 * pred_w,
                pred_ctr_y + 0.5 * pred_h
            ],
            dim=1
        ).reshape((deltas.size(0), deltas.size(1), 4, deltas.size(2) // 4)),
        1, 2
    ).reshape(deltas.size(0), deltas.size(1), deltas.size(2))

    return pred_boxes


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
        return all_anchors

    return forward


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
    # level_ids = []  # #lvl Tensor, each of shape (topk,)
    for level_id, proposals_i, logits_i in zip(
            itertools.count(), proposals, pred_objectness_logits
    ):
        Hi_Wi_A = logits_i.shape[1]
        num_proposals_i = min(pre_nms_topk, Hi_Wi_A)

        topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)
        topk_proposals_i = proposals_i[:, topk_idx].reshape((*topk_scores_i.shape, 4))  # N x topk x 4

        if num_proposals_i < pre_nms_topk:
            topk_proposals_i = F.pad(topk_proposals_i, pad=(0, 0, 0, pre_nms_topk - num_proposals_i, 0, 0), mode='constant', value=0)

        print(topk_proposals_i.shape)
        topk_proposals.append(topk_proposals_i)
        topk_scores.append(topk_scores_i)
        # level_ids.append(torch.full((num_proposals_i,), level_id, dtype=torch.int32, device=device))
    return topk_proposals[0], topk_proposals[1], topk_proposals[2], topk_proposals[3], topk_proposals[4]
    """
    # 2. Concat all levels together
    topk_scores = torch.cat(topk_scores, dim=1)
    topk_proposals = torch.cat(topk_proposals, dim=1)
    print(topk_scores.shape, topk_proposals.shape)
    # level_ids = cat(level_ids, dim=0)
    return topk_proposals

    # 3. For each image, run a per-level NMS, and choose topk results.
    boxes = boxclip(topk_proposals[0])
    scores_per_img = topk_scores[0]

    # filter empty boxes
    keep = boxnonempty(boxes, min_box_side_len)
    lvl = level_ids
    boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], level_ids[keep]

    keep = batched_nms(boxes, scores_per_img, lvl, nms_thresh)
    # In Detectron1, there was different behavior during training vs. testing.
    # (https://github.com/facebookresearch/Detectron/issues/459)
    # During training, topk is over the proposals from *all* images in the training batch.
    # During testing, it is over the proposals for each image separately.
    # As a result, the training behavior becomes batch-dependent,
    # and the configuration "POST_NMS_TOPK_TRAIN" end up relying on the batch size.
    # This bug is addressed in Detectron2 to make the behavior independent of batch size.
    keep = keep[:post_nms_topk]

    return boxes[keep], scores_per_img[keep]
    """


@register_functionalizer(RPN)
def functionalizeRPN(module: RPN):
    rpn_head = functionalize(module.rpn_head)
    weights = module.box2box_transform.weights
    scale_clamp = module.box2box_transform.scale_clamp

    def forward(features, anchors):
        pred_objectness_logits, pred_anchor_deltas = rpn_head(features)

        pred_objectness_logits = [
            # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]

        proposals = []
        # For each feature map
        for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
            B = anchors_i.size(2)
            N, AB, Hi, Wi = pred_anchor_deltas_i.shape
            A = AB // B
            pred_anchor_deltas_i = pred_anchor_deltas_i.permute(0, 2, 3, 1).reshape(N, Hi*Wi*A, B)

            proposals_i = apply_deltas(
                pred_anchor_deltas_i, anchors_i, weights, scale_clamp
            )

            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals.append(proposals_i.view(N, Hi*Wi*A, B))

        with torch.no_grad():
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
