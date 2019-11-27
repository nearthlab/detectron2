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
        help="If specified, prints out the layers exported in each module"
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


class NonMaxSuppressionONNX(nn.Module):
    def __init__(self,
        image_sizes, nms_thresh, pre_nms_topk,
        post_nms_topk, min_box_side_len
    ):
        super(NonMaxSuppressionONNX, self).__init__()
        self.image_sizes = image_sizes
        self.nms_thresh = nms_thresh
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.min_box_side_len = min_box_side_len

    def forward(self, all_proposals, all_objectness_logits):
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
        device = all_proposals[0].device
        nms_thresh = torch.tensor(self.nms_thresh, dtype=all_proposals[0].dtype)
        pre_nms_topk = torch.tensor(self.pre_nms_topk, dtype=torch.long)
        post_nms_topk = torch.tensor(self.post_nms_topk, dtype=torch.long)
        min_box_side_len = torch.tensor(self.min_box_side_len, dtype=all_proposals[0].dtype)
        num_proposals = [
            min(self.pre_nms_topk, logits_i.shape[1]) for logits_i in all_objectness_logits
        ]

        # 1. Select top-k anchor for every level and every image
        topk_scores = []  # #lvl Tensor, each of shape N x topk
        topk_proposals = []
        level_ids = []  # #lvl Tensor, each of shape (topk,)
        batch_idx = torch.arange(1, device=device) # num_images == 1
        for level_id, proposals_i, logits_i, num_proposals_i in zip(
            itertools.count(), all_proposals, all_objectness_logits, num_proposals
        ):
            # Hi_Wi_A = logits_i.shape[1]
            # num_proposals_i = torch.min(pre_nms_topk, Hi_Wi_A)

            # sort is faster than topk (https://github.com/pytorch/pytorch/issues/22812)
            # But ONNX currently supports topk but not sort.
            topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)
            # logits_i, idx = logits_i.sort(descending=True, dim=1)
            # # topk_scores_i = logits_i[batch_idx, :num_proposals_i]
            # topk_scores_i = torch.index_select(logits_i[batch_idx], 0, proposal_range_i)
            # # topk_idx = idx[batch_idx, :num_proposals_i]
            # topk_idx = torch.index_select(idx[batch_idx], 0, proposal_range_i)

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
        results = []
        for n, image_size in enumerate(self.image_sizes):
            boxes = topk_proposals[n]
            scores_per_img = topk_scores[n]
            # boxes.clip(image_size)
            h, w = image_size
            boxes = torch.cat([
                boxes[:, 0].clamp(min=0, max=w),
                boxes[:, 1].clamp(min=0, max=h),
                boxes[:, 2].clamp(min=0, max=w),
                boxes[:, 3].clamp(min=0, max=h)
            ], dim=0).reshape(-1, 4)

            # filter empty boxes
            # keep = boxes.nonempty(threshold=min_box_side_len)
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            keep = (widths > min_box_side_len) & (heights > min_box_side_len)

            # lvl = level_ids
            # if keep.sum().item() != len(boxes):
            #     boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], level_ids[keep]
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

            # res = Instances(image_size)
            # res.proposal_boxes = boxes[keep]
            # res.objectness_logits = scores_per_img[keep]
            # results.append(res)
        # return results
        return boxes[keep], scores_per_img[keep]


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

        # unpack model
        backbone = model.backbone

        rpn = model.proposal_generator
        rpn_head = rpn.rpn_head
        anchor_generator = AnchorGeneratorONNX(rpn.anchor_generator)
        rpn_proposal = RPNProposalONNX(rpn.box2box_transform)
        rpn_objectness = RPNObjectnessONNX()

        nms = NonMaxSuppressionONNX(
            [(height, width)], # image_sizes
            rpn.nms_thresh,
            rpn.pre_nms_topk[False], # is_training == False
            rpn.post_nms_topk[False], # is_training == False
            rpn.min_box_side_len
        )

        roi_heads = model.roi_heads
        box_pooler = roi_heads.box_pooler
        level_poolers = box_pooler.level_poolers
        box_head = roi_heads.box_head
        box_predictor = roi_heads.box_predictor

        # the GeneralizedRCNN's "preprocess_image" method
        images = ImageList.from_tensors([model.normalizer(img.to(model.device))], model.backbone.size_divisibility)

        # extract backbone + FPN features
        if args.output:
            torch.onnx.export(
                backbone, images.tensor, get_onnx_name('backbone'), opset_version=11,
                verbose=args.verbose, input_names=['input_image'], output_names=backbone._out_features
            )
        feature_map = backbone(images.tensor)
        rpn_input_features = [feature_map[f] for f in rpn.in_features]

        # calculate objectness logits and anchor deltas
        if args.output:
            logit_names = ['objectness_logit_' + f for f in rpn.in_features]
            delta_names = ['anchor_deltas_' + f for f in rpn.in_features]
            rpn_head_output_names = logit_names + delta_names
            torch.onnx.export(
                rpn_head, rpn_input_features, get_onnx_name('rpn_head'), opset_version=11,
                verbose=args.verbose, input_names=rpn.in_features, output_names=rpn_head_output_names
            )
        objectness_logits, anchor_deltas = rpn_head(rpn_input_features)

        # generate anchors
        if args.output:
            anchor_output_names = ['anchors_' + f for f in rpn.in_features]
            torch.onnx.export(
                anchor_generator, rpn_input_features, get_onnx_name('anchor_generator'), opset_version=11,
                verbose=args.verbose, input_names=rpn.in_features, output_names=anchor_output_names
            )
        anchors = anchor_generator(rpn_input_features)

        # calculate proposals
        if args.output:
            proposal_output_names = ['proposals_' + f for f in rpn.in_features]
            torch.onnx.export(
                rpn_proposal, (anchors, anchor_deltas), get_onnx_name('rpn_proposal'), opset_version=11,
                verbose=args.verbose, input_names=anchor_output_names + delta_names, output_names=proposal_output_names
            )
        all_proposals = rpn_proposal(anchors, anchor_deltas)

        # reshape objectness logits
        if args.output:
            objectness_output_names = ['objectness_' + f for f in rpn.in_features]
            torch.onnx.export(
                rpn_objectness, objectness_logits, get_onnx_name('rpn_objectness'), opset_version=11,
                verbose=args.verbose, input_names=logit_names, output_names=objectness_output_names
            )
        all_objectness_logits = rpn_objectness(objectness_logits)

        # run Non-Maximum Suppression
        if args.output:
            torch.onnx.export(
                nms, (all_proposals, all_objectness_logits), get_onnx_name('nms'), opset_version=11,
                verbose=args.verbose, input_names=proposal_output_names + objectness_output_names
            )
        rois = nms(all_proposals, all_objectness_logits)

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
