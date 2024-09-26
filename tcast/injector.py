import torch
import tcast
import copy
from .modules import Linear, Conv2d

def TorchInjector(tcast_specs):
    def torch_to_tcast_module(cls):
        def __init__(self, *args, **kwargs):
            cls.__init__(self, *args, tcast_specs=tcast_specs, **kwargs)
        return type(f'{cls.__name__}_tcast', (cls,), {'__init__': __init__})

    for torchm, tcastm in SUPPORTED_MODULES.items():
        torch.nn.__dict__[torchm] = torch_to_tcast_module(tcastm)

def MixedPrecisionInjector(model, tcast_specs):
    model_mixed = copy.deepcopy(model)
    for name, module in model_mixed.named_modules():
        if isinstance(module, torch.nn.Linear):
            if name in tcast_specs:
                model_mixed.__dict__[name] = Linear(module.in_features, module.out_features, module.bias is not None, tcast_specs=tcast_specs[name], pre_weights=module.weight, pre_bias=module.bias)
        elif isinstance(module, torch.nn.Conv2d):
            if name in tcast_specs:
                model_mixed.__dict__[name] = Conv2d(module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups, module.bias is not None, module.padding_mode, tcast_specs=tcast_specs[name], pre_weights=module.weight, pre_bias=module.bias)
    return model_mixed

SUPPORTED_MODULES = {
    "Linear": Linear,
    "Conv2d": Conv2d,
}
    