import os
import cv2
import onnx
import onnxsim
import torch

import detectron2.onnx.fpn
import detectron2.onnx.layers
import detectron2.onnx.nn
import detectron2.onnx.resnet
import detectron2.onnx.rpn
import detectron2.onnx.roi_heads
import detectron2.onnx.vision
from detectron2.onnx.rpn_outputs import apply_deltas, predict_proposals
from detectron2.onnx.roi_heads import predict_boxes, predict_probs
from detectron2.onnx.functionalize import test_functionalizer, test_equal, get_shapes
from detectron2.onnx.trt_friendly_module import TRTFriendlyModule, composeTRTFriendlyModule
from detectron2.utils.file_io import saveTensors, loadTensors


def export_onnx(
        model: TRTFriendlyModule, dummy_input,
        check=False, simplify=False, optimize=False,
        output_dir=None, path=None, **kwargs
):
    assert (path is not None and output_dir is None) or (path is None and output_dir is not None)
    path = path or os.path.join(output_dir, '{}.onnx'.format(model.name))
    output_dir = os.path.split(path)[0]
    assert os.path.isdir(output_dir)

    print('Exporting ONNX model {} ...'.format(path))
    torch.onnx.export(model, dummy_input, path, **kwargs)
    if simplify:
        print('Simplifying ONNX model {} ...'.format(path))
        check_n = 1 if check else 0
        model_sim = onnxsim.simplify(path, check_n=check_n, perform_optimization=optimize)
        onnx.save(model_sim, path)
    if check:
        print('Checking ONNX model {} ...'.format(path))
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)
        print('ONNX model checked! (IR version: {})'.format(onnx_model.ir_version))

        input_json_path = os.path.join(output_dir, '{}_input.json'.format(model.name))
        saveTensors(dummy_input, input_json_path)

        tic = cv2.getTickCount()
        dummy_output = model(*dummy_input) if isinstance(dummy_input, tuple) else model(dummy_input)
        toc = cv2.getTickCount()
        print('inference time: {}'.format((toc - tic) / cv2.getTickFrequency()))

        output_json_path = os.path.join(output_dir, '{}_output.json'.format(model.name))
        saveTensors(dummy_output, output_json_path)
