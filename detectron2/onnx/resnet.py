import torch.nn.functional as F

from detectron2.modeling.backbone.resnet import BasicStem, BottleneckBlock, ResNet
from detectron2.onnx.functionalize import register_functionalizer, functionalize


@register_functionalizer(BasicStem)
def functionalizeBasicStem(module: BasicStem):
    conv = functionalize(module.conv1)

    return lambda x: F.max_pool2d(F.relu(conv(x)), kernel_size=3, stride=2, padding=1)


@register_functionalizer(BottleneckBlock)
def functionalizeBottleneckBlock(module: BottleneckBlock):
    conv1 = functionalize(module.conv1)
    conv2 = functionalize(module.conv2)
    conv3 = functionalize(module.conv3)
    shortcut = functionalize(module.shortcut)

    return lambda x: F.relu(conv3(F.relu(conv2(F.relu(conv1(x))))) + shortcut(x))


@register_functionalizer(ResNet)
def functionalizeResNet(module: ResNet):
    stem = functionalize(module.stem)
    stages_and_names = [(functionalize(stage), name) for stage, name in module.stages_and_names]
    out_features = module._out_features

    def forward(x):
        outputs = dict()
        x = stem(x)
        if "stem" in out_features:
            outputs["stem"] = x
        for stage, name in stages_and_names:
            x = stage(x)
            if name in out_features:
                outputs[name] = x
        return outputs

    return forward
