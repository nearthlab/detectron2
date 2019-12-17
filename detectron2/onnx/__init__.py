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
import detectron2.onnx.vision
from detectron2.onnx.functionalize import test_functionalizer, test_equal, get_shapes
from detectron2.onnx.trt_friendly_module import TRTFriendlyModule, composeTRTFriendlyModule

get_depth = lambda L: max(map(get_depth, L))+1 if type(L) in (list, tuple) else 0

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

        input_path = os.path.join(output_dir, '{}_input.tensor'.format(model.name))
        if isinstance(dummy_input, dict):
            torch.save([dummy_input.get(key) for key in sorted(dummy_input.keys())], input_path)
        else:
            torch.save(dummy_input, input_path)

        with open(os.path.join(output_dir, '{}_input.shape'.format(model.name)), 'w') as fp:
            if isinstance(dummy_input, torch.Tensor):
                fp.write('{}'.format([get_shapes(dummy_input)]))
            else:
                fp.write('{}'.format(get_shapes(dummy_input)))

        tic = cv2.getTickCount()
        dummy_output = model(dummy_input)
        toc = cv2.getTickCount()
        print('inference time: {}'.format((toc - tic) / cv2.getTickFrequency()))

        output_path = os.path.join(output_dir, '{}_output.tensor'.format(model.name))
        if isinstance(dummy_output, dict):
            torch.save([dummy_output.get(key) for key in sorted(dummy_output.keys())], output_path)
        elif get_depth(dummy_output) == 2:
            torch.save([x for y in dummy_output for x in y], output_path)
        else:
            torch.save(dummy_output, output_path)


        with open(os.path.join(output_dir, '{}_output.shape'.format(model.name)), 'w') as fp:
            shapes = get_shapes(dummy_output)
            depth = get_depth(shapes)
            if depth == 3:
                fp.write('{}'.format([x for y in shapes for x in y]))
            elif depth == 1:
                fp.write('{}'.format([shapes]))
            else:
                fp.write('{}'.format(shapes))
