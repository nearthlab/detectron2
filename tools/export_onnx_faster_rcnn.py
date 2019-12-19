import argparse
import multiprocessing as mp
import os

import torch

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.modeling.box_regression import _DEFAULT_SCALE_CLAMP
from detectron2.modeling.proposal_generator.rpn import find_top_rpn_proposals
from detectron2.modeling.proposal_generator.rpn_outputs import RPNOutputs
from detectron2.onnx import (
    TRTFriendlyModule,
    export_onnx,
    test_functionalizer
)
from detectron2.structures import ImageList
from detectron2.utils.logger import setup_logger


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Prints out the layers exported in each module"
    )
    parser.add_argument(
        "--opset-version", default=9, type=int,
        help="ONNX opset version (default: 9)"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test functionalizer"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Check exported ONNX model"
    )
    parser.add_argument(
        "--simplify", action="store_true",
        help="Simplify exported ONNX model"
    )
    parser.add_argument(
        "--skip-optimization", action="store_true",
        help="Skip optimization while simplifying ONNX model"
    )
    return parser


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
            dim=2
        ).reshape((deltas.size(0), deltas.size(1), 4, deltas.size(2) // 4)),
        2, 3
    ).reshape(deltas.size(0), deltas.size(1), deltas.size(2))

    return pred_boxes


def predict_proposals(anchors, pred_anchor_deltas, weights, scale_clamp):
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


def boxes_clip(boxes, box_size):
    h, w = box_size

    return torch.transpose(torch.cat([
        boxes[:, :, 0].clamp(min=0, max=w).unsqueeze(1),
        boxes[:, :, 1].clamp(min=0, max=h).unsqueeze(1),
        boxes[:, :, 2].clamp(min=0, max=w).unsqueeze(1),
        boxes[:, :, 3].clamp(min=0, max=h).unsqueeze(1)
    ], dim=1), 1, 2).reshape(boxes.shape)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    basename = os.path.splitext(os.path.basename(args.config_file))[0]
    get_onnx_name = lambda x: os.path.join(args.output, '{}_{}.onnx'.format(basename, x))

    if args.output:
        assert os.path.isdir(args.output), args.output
        export_options = {
            "check": args.check,
            "simplify": args.simplify,
            "optimize": not args.skip_optimization,
            "output_dir": args.output,
            "opset_version": args.opset_version,
            "verbose": args.verbose
        }

        # If input is provided, run GeneralizedRCNN's "inference" method
        # and, at the same time, export pure CNN parts of the model to ONNX format
        # by tracing the model's calculation
        if args.input:
            assert len(args.input) == 1, len(args.input)

            # load model
            cfg = setup_cfg(args)
            predictor = DefaultPredictor(cfg)
            model = predictor.model

            # read image
            img = read_image(args.input[0])
            print('original image shape: {}'.format(img.shape))

            # the preprocessing implemented in the DefaultPredictor's "__call__" method
            if predictor.input_format == "RGB":
                img = img[:, :, ::-1]
            height, width = img.shape[:2]
            img = predictor.transform_gen.get_transform(img).apply_image(img)
            print('resized image shape: {}'.format(img.shape))
            img = torch.as_tensor(img.astype('float32').transpose(2, 0, 1))

            # the GeneralizedRCNN's "preprocess_image" method
            images = ImageList.from_tensors([model.normalizer(img.to(model.device))], model.backbone.size_divisibility)
            print('input shape: {}'.format(images.tensor.shape))
            features = model.backbone(images.tensor)
            resnet_features = model.backbone.bottom_up(images.tensor)
            fpn_in_features = [resnet_features[f] for f in model.backbone.in_features[::-1]]
            rpn_in_features = [features[f] for f in model.proposal_generator.in_features]
            anchors = model.proposal_generator.anchor_generator(rpn_in_features)

            anchors = [[boxes.tensor.unsqueeze(0) for boxes in anchor] for anchor in anchors]
            pred_objectness_logits, pred_anchor_deltas = model.proposal_generator.rpn_head(rpn_in_features)

            outputs = RPNOutputs(
                model.proposal_generator.box2box_transform,
                model.proposal_generator.anchor_matcher,
                model.proposal_generator.batch_size_per_image,
                model.proposal_generator.positive_fraction,
                images,
                pred_objectness_logits,
                pred_anchor_deltas,
                model.proposal_generator.anchor_generator(rpn_in_features),
                model.proposal_generator.boundary_threshold,
                None,
                model.proposal_generator.smooth_l1_beta,
            )

            proposals = find_top_rpn_proposals(
                outputs.predict_proposals(),
                outputs.predict_objectness_logits(),
                images,
                model.proposal_generator.nms_thresh,
                model.proposal_generator.pre_nms_topk[model.proposal_generator.training],
                model.proposal_generator.post_nms_topk[model.proposal_generator.training],
                model.proposal_generator.min_box_side_len,
                model.proposal_generator.training,
            )

            weights = cfg.MODEL.RPN.BBOX_REG_WEIGHTS
            scale_clamp = _DEFAULT_SCALE_CLAMP
            nms_thresh = cfg.MODEL.RPN.NMS_THRESH
            pre_nms_topk = cfg.MODEL.RPN.PRE_NMS_TOPK_TEST
            post_nms_topk = cfg.MODEL.RPN.POST_NMS_TOPK_TEST
            min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            proposal_predictor = lambda x: predict_proposals(anchors, x, weights, scale_clamp)
            objectness_predictor = lambda x: [score.permute(0, 2, 3, 1).reshape(score.shape[0], score.shape[1] * score.shape[2] * score.shape[3]) for score in x]

            if args.test:
                test_functionalizer(model.backbone.bottom_up, images.tensor)
                test_functionalizer(model.backbone, images.tensor)
                test_functionalizer(model.proposal_generator.rpn_head, rpn_in_features)
                for x, y in zip(proposal_predictor(pred_anchor_deltas), outputs.predict_proposals()):
                    assert (x == y).all(), 'shapes: {} / {}, error: {}'.format(x.shape, y.shape, torch.max(torch.abs(x - y)))
                for x, y in zip(objectness_predictor(pred_objectness_logits), outputs.predict_objectness_logits()):
                    assert (x == y).all(), 'shapes: {} / {}, error: {}'.format(x.shape, y.shape, torch.max(torch.abs(x - y)))

            if args.output:
                # export_onnx(
                #     TRTFriendlyModule(model.backbone),
                #     torch.zeros_like(images.tensor),
                #     **export_options
                # )
                #
                # export_onnx(
                #     TRTFriendlyModule(model.proposal_generator.rpn_head),
                #     [torch.zeros_like(x) for x in rpn_in_features],
                #     **export_options
                # )
                #
                # export_onnx(
                #     TRTFriendlyModule(proposal_predictor, name='proposal_predictor'),
                #     [torch.zeros_like(x) for x in pred_anchor_deltas],
                #     # pred_anchor_deltas,
                #     **export_options
                # )
                #
                # export_onnx(
                #     TRTFriendlyModule(objectness_predictor, name='objectness_predictor'),
                #     [torch.zeros_like(x) for x in pred_objectness_logits],
                #     **export_options
                # )

                backbone = TRTFriendlyModule(model.backbone)
                rpn_head = TRTFriendlyModule(model.proposal_generator.rpn_head)


                def faster_rcnn_prenms_stage(x):
                    fpn_features = backbone(x)
                    rpn_in_features = [fpn_features[f] for f in model.proposal_generator.in_features]
                    objectness_scores, anchor_deltas = rpn_head(rpn_in_features)
                    outputs = [
                        *[fpn_features[key] for key in sorted(fpn_features.keys())],
                        *proposal_predictor(anchor_deltas),
                        *objectness_predictor(objectness_scores)
                    ]
                    return outputs


                export_onnx(
                    TRTFriendlyModule(faster_rcnn_prenms_stage, name='faster_rcnn_prenms_stage'),
                    images.tensor,
                    **export_options
                )
