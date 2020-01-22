import torch
import torch.nn.functional as F

from detectron2.modeling.proposal_generator.rpn import StandardRPNHead, RPN
from detectron2.onnx.functionalize import register_functionalizer, functionalize


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

    # def forward(features):
    #     pred_objectness_logits = []
    #     pred_anchor_deltas = []
    #     for feature in features:
    #         x = F.relu(conv(feature))
    #         pred_objectness_logits.append(objectness_logits(x))
    #         pred_anchor_deltas.append(anchor_deltas(x))
    #
    #     return pred_objectness_logits, pred_anchor_deltas

    def forward(features):
        x = [F.relu(conv(features[0])), F.relu(conv(features[1])), F.relu(conv(features[2])), F.relu(conv(features[3])), F.relu(conv(features[4]))]
        return [objectness_logits(x[0]), objectness_logits(x[1]), objectness_logits(x[2]), objectness_logits(x[3]), objectness_logits(x[4])], \
            [anchor_deltas(x[0]), anchor_deltas(x[1]), anchor_deltas(x[2]), anchor_deltas(x[3]), anchor_deltas(x[4])]

    return forward


@register_functionalizer(RPN)
def functionalizeRPN(module: RPN):
    rpn_head = functionalize(module.rpn_head)
    weights = module.box2box_transform.weights
    scale_clamp = module.box2box_transform.scale_clamp

    def forward(features, anchors):
        pred_objectness_logit, pred_anchor_delta = rpn_head(features)

        proposals = []
        scores = []
        for logit_i, delta_i, anchor_i in zip(pred_objectness_logit, pred_anchor_delta, anchors):
            B = anchor_i.size(2)
            N, AB, Hi, Wi = delta_i.shape
            A = AB // B
            delta_i = delta_i.permute(0, 2, 3, 1).reshape(N, Hi * Wi * A, B)

            proposals.append(apply_deltas(
                delta_i, anchor_i, weights, scale_clamp
            ))
            scores.append(logit_i.permute(0, 2, 3, 1).flatten(1))

        return proposals, scores

    return forward
