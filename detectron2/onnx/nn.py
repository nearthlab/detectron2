import torch.nn as nn
import torch.nn.functional as F

from detectron2.onnx.functionalize import register_functionalizer, functionalize


@register_functionalizer(nn.Sequential)
def functionalizennSequential(module: nn.Sequential):
    layers = [functionalize(module[idx]) for idx in range(len(module))]

    def forward(x):
        for layer in layers:
            x = layer(x)
        return x

    return forward


@register_functionalizer(nn.Conv2d)
def functionalizennConv2d(module: nn.Conv2d):
    if module.bias is not None:
        return lambda x: F.conv2d(
            x,
            module.weight.data.to(device=x.device),
            module.bias.data.to(device=x.device),
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups
        )
    else:
        return lambda x: F.conv2d(
            x,
            module.weight.data.to(device=x.device),
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups
        )


@register_functionalizer(nn.Linear)
def functionalizennLinear(module: nn.Linear):
    if module.bias is not None:
        return lambda x: F.linear(x, module.weight.data.to(device=x.device), module.bias.data.to(device=x.device))
    else:
        return lambda x: F.linear(x, module.weight.data.to(device=x.device))


@register_functionalizer(nn.ReLU)
def functionalizennReLU(module: nn.ReLU):
    return lambda x: F.relu(x, module.inplace)


@register_functionalizer(nn.MaxPool2d)
def functionalizennMaxPool2d(module: nn.MaxPool2d):
    return lambda x: F.max_pool2d(x, module.kernel_size, module.stride,
                                  module.padding, module.dilation, module.ceil_mode,
                                  module.return_indices)


@register_functionalizer(nn.BatchNorm2d)
def functionalizennBatchNorm2d(module: nn.BatchNorm2d):
    exponential_average_factor = module.momentum or 0.0
    return lambda x: F.batch_norm(
        x, module.running_mean.data.to(device=x.device), module.running_var.data.to(device=x.device),
        module.weight.data.to(device=x.device), module.bias.data.to(device=x.device),
        module.training or not module.track_running_stats,
        exponential_average_factor, module.eps)


@register_functionalizer(nn.AdaptiveAvgPool2d)
def functionalizennAdaptiveAvgPool2d(module: nn.AdaptiveAvgPool2d):
    return lambda x: F.adaptive_avg_pool2d(x, module.output_size)
