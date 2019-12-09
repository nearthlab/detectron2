import torch.nn.functional as F

from detectron2.layers import Conv2d, ConvTranspose2d, FrozenBatchNorm2d
from detectron2.onnx.functionalize import register_functionalizer, functionalize

@register_functionalizer(Conv2d)
def functionalizeConv2d(module: Conv2d):
    norm = functionalize(module.norm) or (lambda x: x)
    activation = functionalize(module.activation) or (lambda x: x)

    if module.bias is not None:
        return lambda x: activation(norm(F.conv2d(
            x,
            module.weight.data.to(device=x.device),
            module.bias.data.to(device=x.device),
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups
        )))
    else:
        return lambda x: activation(norm(F.conv2d(
            x,
            module.weight.data.to(device=x.device),
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups
        )))

@register_functionalizer(ConvTranspose2d)
def functionalizeConv2dTranspose2d(module: ConvTranspose2d):
    if module.bias is not None:
        return lambda x: F.conv_transpose2d(
            x,
            module.weight.data.to(device=x.device),
            module.bias.data.to(device=x.device),
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups
        )
    else:
        return lambda x: F.conv_transpose2d(
            x,
            module.weight.data.to(device=x.device),
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups
        )

@register_functionalizer(FrozenBatchNorm2d)
def functionalizeFrozenBatchNorm2d(module: FrozenBatchNorm2d):
    def forward(x):
        weight = module.weight.data.to(device=x.device)
        bias = module.bias.data.to(device=x.device)
        running_var = module.running_var.data.to(device=x.device)
        running_mean = module.running_mean.data.to(device=x.device)

        scale = weight * (running_var + module.eps).rsqrt()
        bias = bias - running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)

        return x * scale + bias

    return forward
