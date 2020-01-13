import argparse
import multiprocessing as mp
import os

import torch

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.modeling.proposal_generator.rpn import find_top_rpn_proposals
from detectron2.modeling.proposal_generator.rpn_outputs import RPNOutputs
from detectron2.onnx import (
    TRTFriendlyModule,
    export_onnx,
    predict_proposals,
    predict_boxes,
    predict_probs
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

            # unpack major constants
            rpn_weights = cfg.MODEL.RPN.BBOX_REG_WEIGHTS
            roi_box_head_weights = cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS
            nms_thresh = cfg.MODEL.RPN.NMS_THRESH
            pre_nms_topk = cfg.MODEL.RPN.PRE_NMS_TOPK_TEST
            post_nms_topk = cfg.MODEL.RPN.POST_NMS_TOPK_TEST
            min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE

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

            rpn_outputs = RPNOutputs(
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

            with torch.no_grad():
                proposals = find_top_rpn_proposals(
                    rpn_outputs.predict_proposals(),
                    rpn_outputs.predict_objectness_logits(),
                    images,
                    model.proposal_generator.nms_thresh,
                    model.proposal_generator.pre_nms_topk[model.proposal_generator.training],
                    model.proposal_generator.post_nms_topk[model.proposal_generator.training],
                    model.proposal_generator.min_box_side_len,
                    model.proposal_generator.training,
                )
            roi_heads = model.roi_heads
            roi_heads_in_features = [features[f] for f in roi_heads.in_features]

            proposals = [x.proposal_boxes.tensor.unsqueeze(0) for x in proposals]
            box_pooler_features = TRTFriendlyModule(roi_heads.box_pooler)(roi_heads_in_features, proposals)

            proposal_predictor = lambda x: predict_proposals(anchors, x, rpn_weights)
            objectness_predictor = lambda x: [score.permute(0, 2, 3, 1).reshape(score.shape[0], score.shape[1] * score.shape[2] * score.shape[3]) for score in x]

            boxes_predictor = lambda x, y: predict_boxes(x, y, roi_box_head_weights)
            score_predictor = lambda x: predict_probs(x)

            if args.output:
                backbone = TRTFriendlyModule(model.backbone)
                rpn_head = TRTFriendlyModule(model.proposal_generator.rpn_head)
                roi_head_preprocess = TRTFriendlyModule(model.roi_heads)


                def faster_rcnn_stage1(x):
                    fpn_features = backbone(x)
                    rpn_in_features = [fpn_features[f] for f in model.proposal_generator.in_features]
                    objectness_scores, anchor_deltas = rpn_head(rpn_in_features)
                    outputs = [
                        *[fpn_features[key] for key in sorted(fpn_features.keys())],
                        *proposal_predictor(anchor_deltas),
                        *objectness_predictor(objectness_scores)
                    ]
                    return outputs


                def faster_rcnn_stage2(box_pooler_features, proposals):
                    pred_class_logits, pred_proposal_deltas = roi_head_preprocess(box_pooler_features)
                    outputs = [
                        boxes_predictor(pred_proposal_deltas, proposals),
                        score_predictor(pred_class_logits)
                    ]
                    return outputs


                export_onnx(
                    TRTFriendlyModule(faster_rcnn_stage1, name='faster_rcnn_stage1'),
                    images.tensor,
                    **export_options
                )

                export_onnx(
                    TRTFriendlyModule(faster_rcnn_stage2, name='faster_rcnn_stage2'),
                    (box_pooler_features, torch.cat(proposals, dim=1)),
                    **export_options
                )
