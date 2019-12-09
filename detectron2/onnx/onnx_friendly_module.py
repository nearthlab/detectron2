import torch.nn as nn
from detectron2.onnx.functionalize import functionalize

class ONNXFriendlyModule(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.forward = functionalize(module)
        self.name = module.__class__.__name__
