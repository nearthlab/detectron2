import torch
import torch.nn as nn


FUNCTIONALIZERS = dict()


def functionalize(module: nn.Module):
    if module is None:
        return lambda x: x
    cls = module.__class__
    if cls in FUNCTIONALIZERS:
        return FUNCTIONALIZERS.get(cls)(module)
    else:
        raise NotImplementedError(
            "functionalizer for the class {} is not implemented.".format(cls.__name__)
        )


def register_functionalizer(cls):
    def wrapper(functionalizer):
        if cls not in FUNCTIONALIZERS:
            FUNCTIONALIZERS[cls] = functionalizer
        else:
            raise Exception(
                "functionalizer for the class {} is already registered".format(cls.__name__)
            )
        return functionalizer

    return wrapper


def test_equal(y1, y2):
    if type(y1) != type(y2):
        return False, 'Different types: {} / {}'.format(type(y1), type(y2))
    if type(y1) == torch.Tensor:
        success = (y1 == y2).all()
        msg = '' if success else 'Different tensors: {} / {}'.format(y1, y2)
        return success, msg
    elif type(y1) in [list, tuple]:
        if len(y1) != len(y2):
            return False, 'Different length: {} / {}'.format(len(y1), len(y2))
        for z1, z2 in zip(y1, y2):
            success, msg = test_equal(z1, z2)
            if not success:
                return success, msg
        return True, ''
    elif type(y1) == dict:
        if y1.keys() != y2.keys():
            return False, 'Different keys: {} / {}'.format(y1.keys(), y2.keys())
        for key in y1:
            success, msg = test_equal(y1.get(key), y2.get(key))
            if not success:
                return success, msg
        return True, ''
    else:
        success = (y1 == y2)
        msg = '' if success else 'Different values: {} / {}'.format(y1, y2)
        return success, msg


def get_shapes(x):
    if type(x) == torch.Tensor:
        return list(x.shape)
    elif type(x) == dict:
        return [get_shapes(x.get(key)) for key in sorted(x.keys())]
    elif type(x) in [list, tuple]:
        return [get_shapes(value) for value in x]
    else:
        return None


def test_functionalizer(module: nn.Module, dummy_input):
    module = module.cuda()
    func = functionalize(module)

    orig_output = module(dummy_input)
    func_output = func(dummy_input)

    success, msg = test_equal(orig_output, func_output)
    if success:
        print("Test passed: module({})".format(module.__class__))
        print("\tinput shapes: {}".format(get_shapes(dummy_input)))
        print("\toutput shapes: {}".format(get_shapes(orig_output)))
    else:
        print("Test failed: module({})\n\t{}".format(module.__class__, msg))


if __name__ == '__main__':
    print(FUNCTIONALIZERS)
