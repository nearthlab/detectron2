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


def test_functionalizer(module: nn.Module, input_tensor: torch.Tensor):
    module = module.cuda()
    func = functionalize(module)

    orig_output = module(input_tensor)
    func_output = func(input_tensor)

    if type(orig_output) == torch.Tensor:
        assert (orig_output == func_output).all(),\
            "module output: {}\nfunctional output: {}".format(orig_output, func_output)

    if type(orig_output) == dict:
        for name in orig_output:
            assert (orig_output.get(name) == func_output.get(name)).all(),\
                "[{}]\nmodule output: {}\nfunctional output: {}".format(name, orig_output, func_output)

    print("Test passed: module({}), input_tensor({})".format(module.__class__, input_tensor.shape))

if __name__ == '__main__':
    print(FUNCTIONALIZERS)
