import torch.nn as nn
from detectron2.onnx.functionalize import functionalize

class TRTFriendlyModule(nn.Module):
    def __init__(self, module_or_func, name=None):
        super().__init__()
        self.forward = functionalize(module_or_func) if isinstance(module_or_func, nn.Module) \
            else module_or_func
        self.name = name or module_or_func.__class__.__name__

def composeTRTFriendlyModule(
    bottom: TRTFriendlyModule,
    top: TRTFriendlyModule
):
    return TRTFriendlyModule(lambda x: top.forward(bottom.forward(x)), name='{}->{}'.format(bottom.name, top.name))
