import os
import torch
import detectron2.onnx.layers
import detectron2.onnx.nn
import detectron2.onnx.resnet
from detectron2.onnx.functionalize import test_functionalizer
from detectron2.onnx.onnx_friendly_module import ONNXFriendlyModule


def export(model: ONNXFriendlyModule, dummy_input, output_dir=None, path=None, **kwargs):
    assert (path is not None and output_dir is None) or (path is None and output_dir is not None)
    path = path or os.path.join(output_dir, '{}.onnx'.format(model.name))
    torch.onnx.export(model, dummy_input, path, **kwargs)


if __name__ == '__main__':
    import torch.nn as nn
    from detectron2.onnx.functionalize import FUNCTIONALIZERS

    module = nn.Conv2d(3, 64, kernel_size=3)
    test_functionalizer(module, torch.randn(1, 3, 224, 224, device="cuda"))
