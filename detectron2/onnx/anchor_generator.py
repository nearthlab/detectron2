import torch
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.onnx.functionalize import register_functionalizer

@register_functionalizer(DefaultAnchorGenerator)
def functionalizeDefaultAnchorGenerator(module: DefaultAnchorGenerator):
    def forward(grid_sizes):
        anchors_over_all_feature_maps = module.grid_anchors(grid_sizes)
        anchors_in_image = []
        for anchors_per_feature_map in anchors_over_all_feature_maps:
            device = anchors_per_feature_map.device
            boxes = torch.as_tensor(anchors_per_feature_map, dtype=torch.float32, device=device)
            if boxes.numel() == 0:
                boxes = torch.zeros(0, 4, dtype=torch.float32, device=device)
            anchors_in_image.append(boxes)

        return [anchors_in_image]
    return forward
