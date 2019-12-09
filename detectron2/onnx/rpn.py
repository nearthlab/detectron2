import torch.nn.functional as F

from detectron2.modeling.proposal_generator.rpn import StandardRPNHead
from detectron2.onnx.functionalize import register_functionalizer, functionalize


@register_functionalizer(StandardRPNHead)
def functionalizeStandardRPNHead(module: StandardRPNHead):
    conv = functionalize(module.conv)
    objectness_logits = functionalize(module.objectness_logits)
    anchor_deltas = functionalize(module.anchor_deltas)

    def forward(features):
        """
        Args:
            features (list[Tensor]): list of feature maps
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = F.relu(conv(x))
            pred_objectness_logits.append(objectness_logits(t))
            pred_anchor_deltas.append(anchor_deltas(t))
        return pred_objectness_logits, pred_anchor_deltas

    return forward
