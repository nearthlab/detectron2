import os

import onnx
import onnxsim
import torch

import detectron2.onnx.fpn
import detectron2.onnx.layers
import detectron2.onnx.nn
import detectron2.onnx.resnet
import detectron2.onnx.rpn
from detectron2.onnx.functionalize import test_functionalizer
from detectron2.onnx.onnx_friendly_module import ONNXFriendlyModule


def export_onnx(
        model: ONNXFriendlyModule, dummy_input,
        check=False, simplify=False, optimize=False,
        output_dir=None, path=None, **kwargs
):
    assert (path is not None and output_dir is None) or (path is None and output_dir is not None)
    path = path or os.path.join(output_dir, '{}.onnx'.format(model.name))
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


if __name__ == '__main__':
    import torch.nn as nn
    from detectron2.onnx.functionalize import FUNCTIONALIZERS

    module = nn.Conv2d(3, 64, kernel_size=3)
    test_functionalizer(module, torch.randn(1, 3, 224, 224, device="cuda"))
