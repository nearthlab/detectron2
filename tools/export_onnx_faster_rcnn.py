import argparse
import copy
import glob
import multiprocessing as mp
import itertools
import os
import time
import cv2
import torch
import tqdm

from torch import nn
from detectron2.config import get_cfg
from detectron2.structures import ImageList
from detectron2.data.detection_utils import read_image
from detectron2.layers import cat, batched_nms
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2.modeling.poolers import convert_boxes_to_pooler_format, assign_boxes_to_levels
from detectron2.modeling.proposal_generator.rpn_outputs import RPNOutputs, find_top_rpn_proposals
from detectron2.engine.defaults import DefaultPredictor

# constants
WINDOW_NAME = "COCO detections"


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
        "--opset", default=9, type=int,
        help="ONNX opset version (default: 9)"
    )
    parser.add_argument(
        "--ir-version", default=3, type=int,
        help="If --check is specified, performs ONNX Internal Representation version conversion"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Check exported ONNX files"
    )
    return parser


class AnchorGeneratorONNX(nn.Module):
    def __init__(self, anchor_generator):
        super(AnchorGeneratorONNX, self).__init__()
        self.anchor_generator = anchor_generator

    def forward(self, features):
        grid_sizes = [f.shape[-2:] for f in features]
        grid_anchors = self.anchor_generator.grid_anchors(grid_sizes)

        return grid_anchors


class RPNProposalONNX(nn.Module):
    def __init__(self, box2box_transform):
        super(RPNProposalONNX, self).__init__()
        self.box2box_transform = box2box_transform

    def forward(self, anchors, anchor_deltas):
        """
         Transform anchors into proposals by applying the predicted anchor deltas.

         Returns:
            proposals (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A, B), where B is box dimension (4 or 5).
        """
        proposals = []

        # For each feature map
        for anchors_i, anchor_deltas_i in zip(anchors, anchor_deltas):
            B = anchors_i.size(1)
            N, _, Hi, Wi = anchor_deltas_i.shape
            # Reshape: (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N*Hi*Wi*A, B)
            anchor_deltas_i = (
                anchor_deltas_i.view(N, -1, B, Hi, Wi).permute(0, 3, 4, 1, 2).reshape(-1, B)
            )
            # Concatenate all anchors to shape (N*Hi*Wi*A, B)
            # type(anchors_i[0]) is Boxes (B = 4) or RotatedBoxes (B = 5)
            # anchors_i = cat([b for b in anchors_i], dim=0) # type(anchors_i[0]).cat(anchors_i)
            proposals_i = self.box2box_transform.apply_deltas(
                anchor_deltas_i, anchors_i
            )
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals.append(proposals_i.view(N, -1, B))
        return proposals


class RPNObjectnessONNX(nn.Module):
    def __init__(self):
        super(RPNObjectnessONNX, self).__init__()

    def forward(self, objectness_logits):
        return [
            # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).reshape(1, -1)
            for score in objectness_logits
        ]


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    basename = os.path.splitext(os.path.basename(args.config_file))[0]
    get_onnx_name = lambda x: os.path.join(args.output, '{}_{}.onnx'.format(basename, x))

    if args.output:
        assert os.path.isdir(args.output), args.output
        if args.check:
            import onnx
            from onnx import version_converter, helper


    def convert_ir(onnx_model, ir_version, path):
        if onnx_model.ir_version != ir_version:
            logger.info('Converting ONNX model IR version from {} to {} ...'.format(onnx_model.ir_version, ir_version))
            converted_model = version_converter.convert_version(onnx_model, ir_version)
            onnx.save(converted_model, path)


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

        # unpack model
        backbone = model.backbone

        rpn = model.proposal_generator
        rpn_head = rpn.rpn_head
        anchor_generator = AnchorGeneratorONNX(rpn.anchor_generator)
        rpn_proposal = RPNProposalONNX(rpn.box2box_transform)
        rpn_objectness = RPNObjectnessONNX()

        roi_heads = model.roi_heads
        box_pooler = roi_heads.box_pooler
        level_poolers = box_pooler.level_poolers
        box_head = roi_heads.box_head
        box_predictor = roi_heads.box_predictor

        # the GeneralizedRCNN's "preprocess_image" method
        images = ImageList.from_tensors([model.normalizer(img.to(model.device))], model.backbone.size_divisibility)

        feature_map = backbone(images.tensor)
        # extract backbone + FPN features
        if args.output:
            logger.info('Exporting backbone ...')
            backbone_name = get_onnx_name('backbone')
            torch.onnx.export(
                backbone, images.tensor, backbone_name, opset_version=args.opset,
                verbose=args.verbose, input_names=['input_image'], output_names=backbone._out_features
            )

            if args.check:
                print('Checking backbone ...')
                onnx_backbone = onnx.load(backbone_name)
                onnx.checker.check_model(onnx_backbone)
                print('Model checked! (IR version: {})'.format(onnx_backbone.ir_version))
                convert_ir(onnx_backbone, args.ir_version, backbone_name)

        rpn_input_features = [feature_map[f] for f in rpn.in_features]
        # calculate objectness logits and anchor deltas
        if args.output:
            logger.info('Exporting rpn_head ...')
            logit_names = ['objectness_logit_' + f for f in rpn.in_features]
            delta_names = ['anchor_deltas_' + f for f in rpn.in_features]
            rpn_head_output_names = logit_names + delta_names
            torch.onnx.export(
                rpn_head, rpn_input_features, get_onnx_name('rpn_head'), opset_version=args.opset,
                verbose=args.verbose, input_names=rpn.in_features, output_names=rpn_head_output_names
            )

            if args.check:
                logger.info('Checking rpn_head ... ')
                onnx_rpn_head = onnx.load(get_onnx_name('rpn_head'))
                onnx.checker.check_model(onnx_rpn_head)

        objectness_logits, anchor_deltas = rpn_head(rpn_input_features)

        # generate anchors
        if args.output:
            logger.info('Exporting anchor_generator ...')
            anchor_output_names = ['anchors_' + f for f in rpn.in_features]
            torch.onnx.export(
                anchor_generator, rpn_input_features, get_onnx_name('anchor_generator'), opset_version=args.opset,
                verbose=args.verbose, input_names=rpn.in_features, output_names=anchor_output_names
            )
        anchors = anchor_generator(rpn_input_features)

        # calculate proposals
        if args.output:
            logger.info('Exporting rpn_proposal ...')
            proposal_output_names = ['proposals_' + f for f in rpn.in_features]
            torch.onnx.export(
                rpn_proposal, (anchors, anchor_deltas), get_onnx_name('rpn_proposal'), opset_version=args.opset,
                verbose=args.verbose, input_names=anchor_output_names + delta_names, output_names=proposal_output_names
            )
        all_proposals = rpn_proposal(anchors, anchor_deltas)

        # reshape objectness logits
        if args.output:
            logger.info('Exporting rpn_objectness ...')
            objectness_output_names = ['objectness_' + f for f in rpn.in_features]
            torch.onnx.export(
                rpn_objectness, objectness_logits, get_onnx_name('rpn_objectness'), opset_version=args.opset,
                verbose=args.verbose, input_names=logit_names, output_names=objectness_output_names
            )
        all_objectness_logits = rpn_objectness(objectness_logits)

        with torch.no_grad():
            proposals = find_top_rpn_proposals(
                all_proposals,
                all_objectness_logits,
                images,
                rpn.nms_thresh,
                rpn.pre_nms_topk[rpn.training],
                rpn.post_nms_topk[rpn.training],
                rpn.min_box_side_len,
                rpn.training
            )

            inds = [p.objectness_logits.sort(descending=True)[1] for p in proposals]
            proposals = [p[ind] for p, ind in zip(proposals, inds)]

        """
        roi_heads_input_features = [feature_map[f] for f in roi_heads.in_features]
        box_lists = [x.proposal_boxes for x in proposals]
        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)
        level_assignments = assign_boxes_to_levels(
            box_lists, box_pooler.min_level, box_pooler.max_level, box_pooler.canonical_box_size, box_pooler.canonical_level
        )

        num_boxes = len(pooler_fmt_boxes)
        num_channels = roi_heads_input_features[0].shape[1]
        output_size = box_pooler.output_size[0]

        dtype, device = roi_heads_input_features[0].dtype, roi_heads_input_features[0].device
        output = torch.zeros(
            (num_boxes, num_channels, output_size, output_size), dtype=dtype, device=device
        )
        # if args.output:
        #     torch.onnx.export(
        #          box_pooler, ..., get_onnx_name('box_pooler'), opset_version=11,
        #         verbose=args.verbose, input_names=roi_heads.in_features
        #     )
        """
