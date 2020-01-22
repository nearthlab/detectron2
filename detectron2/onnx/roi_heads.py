import sys

import torch
import torch.nn.functional as F

from detectron2.layers.roi_align import ROIAlign
from detectron2.modeling import StandardROIHeads
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.box_head import FastRCNNConvFCHead
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.onnx.roi_align import roi_align_onnx
from detectron2.onnx.rpn_outputs import apply_deltas
from detectron2.onnx.functionalize import register_functionalizer, functionalize


@register_functionalizer(ROIAlign)
def functionalizeROIAlign(module: ROIAlign):
    output_size = module.output_size
    spatial_scale = module.spatial_scale
    sampling_ratio = module.sampling_ratio
    aligned = module.aligned

    return lambda input, rois: roi_align_onnx(
        input, rois,
        output_size, spatial_scale,
        sampling_ratio, aligned
    )


def boxes_area(boxes: torch.Tensor):
    return (boxes[:, :, 2] - boxes[:, :, 0]) * (boxes[:, :, 3] - boxes[:, :, 1])


def assign_boxes_to_levels(box_lists: torch.Tensor, min_level, max_level, canonical_box_size, canonical_level):
    eps = sys.float_info.epsilon
    box_sizes = torch.sqrt(torch.cat([boxes_area(boxes) for boxes in box_lists]))
    # Eqn.(1) in FPN paper
    level_assignments = torch.floor(
        canonical_level + torch.log2(box_sizes / canonical_box_size + eps)
    )
    # clamp level to (min, max), in case the box size is too large or too small
    # for the available feature maps
    level_assignments = torch.clamp(level_assignments, min=min_level, max=max_level)
    return level_assignments.to(torch.int64) - min_level


def convert_boxes_to_pooler_format(box_lists: list):
    def fmt_box_list(box_tensor, batch_index):
        repeated_index = torch.full(
            (*box_tensor.shape[:2], 1), batch_index, dtype=box_tensor.dtype, device=box_tensor.device
        )
        return torch.cat((repeated_index, box_tensor), dim=2)

    pooler_fmt_boxes = torch.cat(
        [fmt_box_list(box_list, i) for i, box_list in enumerate(box_lists)], dim=0
    )

    return pooler_fmt_boxes


@register_functionalizer(ROIPooler)
def functionalizeROIPooler(module: ROIPooler):
    level_poolers = [functionalize(layer) for layer in module.level_poolers]
    min_level = module.min_level
    max_level = module.max_level
    canonical_box_size = module.canonical_box_size
    canonical_level = module.canonical_level

    def forward(x, box_lists):
        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)
        level_assignments = assign_boxes_to_levels(
            box_lists, min_level, max_level, canonical_box_size, canonical_level
        )

        dtype, device = x[0].dtype, x[0].device
        num_boxes = pooler_fmt_boxes.shape[1]
        num_channels = x[0].shape[1]
        output_size = module.output_size[0]
        output = torch.zeros(
            (1, num_boxes, num_channels, output_size, output_size), dtype=dtype, device=device
        )

        for level, (x_level, pooler) in enumerate(zip(x, level_poolers)):
            inds = torch.nonzero(level_assignments == level)[:, 1]
            output[:, inds] = pooler(x_level, pooler_fmt_boxes[:, inds])

        return output

    return forward


@register_functionalizer(FastRCNNConvFCHead)
def functionalizeFastRCNNConvFCHead(module: FastRCNNConvFCHead):
    conv_norm_relus = [functionalize(block) for block in module.conv_norm_relus]
    fcs = [functionalize(fc) for fc in module.fcs]

    def forward(x):
        for layer in conv_norm_relus:
            x = layer(x)
        if len(fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=2)
            for layer in fcs:
                x = F.relu(layer(x))
        return x

    return forward


@register_functionalizer(FastRCNNOutputLayers)
def functionalizeFastRCNNOutputLayers(module: FastRCNNOutputLayers):
    cls_score = functionalize(module.cls_score)
    bbox_pred = functionalize(module.bbox_pred)

    def forward(x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=2)
        scores = cls_score(x)
        proposal_deltas = bbox_pred(x)
        return scores, proposal_deltas

    return forward


def predict_boxes(pred_proposal_deltas, proposals, weights):
    num_pred = proposals.shape[1]
    B = proposals.shape[2]
    K = pred_proposal_deltas.shape[2] // B
    boxes = apply_deltas(
        pred_proposal_deltas.reshape(1, num_pred * K, B),
        torch.cat([proposals, proposals], dim=2).reshape(1, num_pred * K, B),
        weights
    )
    return boxes.reshape(1, num_pred, K * B)


def predict_probs(pred_class_logits):
    return F.softmax(pred_class_logits, dim=2)


@register_functionalizer(StandardROIHeads)
def functionalizeStandardROIHeads(module: StandardROIHeads):
    box_pooler = functionalize(module.box_pooler)
    box_head = functionalize(module.box_head)
    box_predictor = functionalize(module.box_predictor)

    # def forward(features, proposals):
    #     box_features = box_pooler(features, proposals)
    #     box_features = box_head(box_features)
    #     return box_predictor(box_features)
    def forward(box_features):
        return box_predictor(box_head(box_features))

    return forward


