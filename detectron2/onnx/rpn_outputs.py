import torch

from detectron2.modeling.box_regression import _DEFAULT_SCALE_CLAMP

def apply_deltas(deltas, boxes, weights, scale_clamp=_DEFAULT_SCALE_CLAMP):
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
            dim=2
        ).reshape((deltas.size(0), deltas.size(1), 4, deltas.size(2) // 4)),
        2, 3
    ).reshape(deltas.size(0), deltas.size(1), deltas.size(2))

    return pred_boxes


def predict_proposals(anchors, pred_anchor_deltas, weights, scale_clamp=_DEFAULT_SCALE_CLAMP):
    proposals = []
    # Transpose anchors from images-by-feature-maps (N, L) to feature-maps-by-images (L, N)
    anchors = list(zip(*anchors))
    # For each feature map
    B = 4  # number of box coordinates
    for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
        N, AB, Hi, Wi = pred_anchor_deltas_i.shape
        A = AB // B

        # Reshape: (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N*Hi*Wi*A, B)
        pred_anchor_deltas_i = (
            # pred_anchor_deltas_i.view(N, -1, B, Hi, Wi).permute(0, 3, 4, 1, 2).reshape(-1, B)
            pred_anchor_deltas_i.view(N, A, B, Hi, Wi).permute(0, 3, 4, 1, 2).reshape(N, Hi * Wi * A, B)
        )
        # Concatenate all anchors to shape (N*Hi*Wi*A, B)
        # type(anchors_i[0]) is Boxes (B = 4) or RotatedBoxes (B = 5)
        # anchors_i = type(anchors_i[0]).cat(anchors_i)
        anchors_i = torch.cat(anchors_i)
        proposals_i = apply_deltas(
            pred_anchor_deltas_i, anchors_i,
            weights, scale_clamp
        )
        # Append feature map proposals with shape (N, Hi*Wi*A, B)
        proposals.append(proposals_i.view(N, Hi * Wi * A, B))
    return proposals