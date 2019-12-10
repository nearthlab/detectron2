import argparse
import multiprocessing as mp
import os

import torch

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.onnx import ONNXFriendlyModule, export_onnx
from detectron2.structures import ImageList
from detectron2.utils.logger import setup_logger


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
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

        # the preprocessing implemented in the DefaultPredictor's "__call__" method
        if predictor.input_format == "RGB":
            img = img[:, :, ::-1]
        height, width = img.shape[:2]
        img = predictor.transform_gen.get_transform(img).apply_image(img)
        img = torch.as_tensor(img.astype('float32').transpose(2, 0, 1))

        # the GeneralizedRCNN's "preprocess_image" method
        images = ImageList.from_tensors([model.normalizer(img.to(model.device))], model.backbone.size_divisibility)

        if args.output:
            export_onnx(
                ONNXFriendlyModule(model.backbone),
                images.tensor,
                check=args.check,
                simplify=args.simplify,
                optimize=not args.skip_optimization,
                output_dir=args.output,
                opset_version=args.opset_version,
                verbose=args.verbose
            )
        features = model.backbone(images.tensor)

        rpn_head_in_features = [features[f] for f in model.proposal_generator.in_features]
        if args.output:
            export_onnx(
                ONNXFriendlyModule(model.proposal_generator.rpn_head),
                rpn_head_in_features,
                check=args.check,
                simplify=args.simplify,
                optimize=not args.skip_optimization,
                output_dir=args.output,
                opset_version=args.opset_version,
                verbose=args.verbose
            )
