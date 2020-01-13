import json

import torch

__all__ = ['saveTensor', 'loadTensor']

class TensorEncoder(json.JSONEncoder):
    def default(self, tensor: torch.Tensor):
        if isinstance(tensor, torch.Tensor):
            return {
                'data': tensor.cpu().flatten().squeeze().tolist(),
                'shape': tensor.shape,
                'dtype': str(tensor.dtype),
                'device': str(tensor.device),
                'requires_grad': tensor.requires_grad
            }
        # else:
        #     return super(TensorEncoder, self).default(tensor)


class TensorDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super(TensorDecoder, self).__init__(object_hook=self.object_hook, *args, **kwargs)
        self.dtypes = {
            'torch.float': torch.float,
            'torch.float16': torch.float16,
            'torch.float32': torch.float32,
            'torch.float64': torch.float64,
            'torch.int': torch.int,
            'torch.int8': torch.int8,
            'torch.int16': torch.int16,
            'torch.int32': torch.int32,
            'torch.int64': torch.int64,
            'torch.bool': torch.bool,
        }

    def object_hook(self, obj):
        if {'data', 'dtype', 'device', 'requires_grad', 'shape'} <= set(obj.keys()):
            return torch.tensor(
                data=obj.get('data'),
                dtype=self.dtypes.get(obj.get('dtype')),
                device=obj.get('device'),
                requires_grad=obj.get('requires_grad')
            ).reshape(obj.get('shape'))
        return obj


def saveTensor(x, path, **kwargs):
    with open(path, 'w') as fp:
        json.dump(x, fp, cls=TensorEncoder, **kwargs)


def loadTensor(path):
    with open(path, 'r') as fp:
        return json.load(fp, cls=TensorDecoder)


if __name__ == '__main__':
    x = torch.randn(1, 3, 2, 3)
    y = [torch.randn(1, 3, 4, 6), torch.randn(1, 3, 4, 6)]
    data = {'x': x, 'y': y}
    saveTensor(data, 'data.json', indent=2)
    z = loadTensor('data.json')
