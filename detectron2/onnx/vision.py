import torch
from torchvision.models.resnet import ResNet, BasicBlock
from detectron2.onnx.functionalize import register_functionalizer, functionalize

@register_functionalizer(BasicBlock)
def functionalizeBasicBlock(module: BasicBlock):
    conv1 = functionalize(module.conv1)
    bn1 = functionalize(module.bn1)
    relu = functionalize(module.relu)
    conv2 = functionalize(module.conv2)
    bn2 = functionalize(module.bn2)
    downsample = functionalize(module.downsample)

    return lambda x: relu(downsample(x) + bn2(conv2(relu(bn1(conv1(x))))))

@register_functionalizer(ResNet)
def functionalizetorchvisionResNet(module: ResNet):
    layers = [
        functionalize(module.conv1),
        functionalize(module.bn1),
        functionalize(module.relu),
        functionalize(module.maxpool),
        functionalize(module.layer1),
        functionalize(module.layer2),
        functionalize(module.layer3),
        functionalize(module.layer4),
        functionalize(module.avgpool),
        lambda x: torch.flatten(x, 1),
        functionalize(module.fc),
    ]

    def forward(x):
        for layer in layers:
            x = layer(x)
        return x

    return forward
