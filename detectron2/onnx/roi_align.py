from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from detectron2 import _C

class _ROIAlignONNX(Function):
    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio, aligned):
        ctx.save_for_backward(roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input.size()
        ctx.aligned = aligned
        output = _C.roi_align_onnx_forward(
            input, roi, spatial_scale, output_size[0], output_size[1], sampling_ratio, aligned
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        raise NotImplementedError()


roi_align_onnx = _ROIAlignONNX.apply