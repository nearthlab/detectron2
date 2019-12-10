import torch
from detectron2.modeling.backbone import FPN
from detectron2.onnx.functionalize import register_functionalizer, functionalize

@register_functionalizer(FPN)
def functionalizeFPN(module: FPN):
    bottom_up = functionalize(module.bottom_up)
    in_features = module.in_features
    out_features = module._out_features
    lateral_convs = [functionalize(lateral_conv) for lateral_conv in module.lateral_convs]
    output_convs = [functionalize(output_conv) for output_conv in module.output_convs]
    top_block = module.top_block
    fuse_type = module._fuse_type

    def forward(x):
        bottom_up_features = bottom_up(x)
        x = [bottom_up_features[f] for f in in_features[::-1]]
        results = []
        prev_features = lateral_convs[0](x[0])
        results.append(output_convs[0](prev_features))
        for features, lateral_conv, output_conv in zip(
            x[1:], lateral_convs[1:], output_convs[1:]
        ):
            # ONNX opset 9 warns about possibly different implementation of interpolate
            # top_down_features = F.interpolate(prev_features, scale_factor=2, mode="nearest")
            fshape = prev_features.shape
            flat = prev_features.flatten(1)
            flat = flat.reshape((1, *flat.shape))
            x_interpolate = torch.transpose(torch.cat([flat, flat], dim=1), 1, 2).reshape((*fshape[:-1], 2*fshape[-1]))
            top_down_features = torch.cat([x_interpolate, x_interpolate], dim=3).reshape((*fshape[:-2], 2*fshape[-2], 2*fshape[-1]))
            lateral_features = lateral_conv(features)
            prev_features = lateral_features + top_down_features
            if fuse_type == "avg":
                prev_features /= 2
            results.insert(0, output_conv(prev_features))

        if top_block is not None:
            top_block_in_feature = bottom_up_features.get(top_block.in_feature, None)
            if top_block_in_feature is None:
                top_block_in_feature = results[out_features.index(top_block.in_feature)]
            results.extend(top_block(top_block_in_feature))

        return dict(zip(out_features, results))

    return forward
