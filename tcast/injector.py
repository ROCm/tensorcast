#!/usr/bin/env python
# tcast/injector.py: monkey patch for quant versions of modules
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import copy

import torch

from .modules import Conv2d, Linear


def torch_injector(tcast_specs):
    def torch_to_tcast_module(cls):
        def __init__(self, *args, **kwargs):
            cls.__init__(self, *args, tcast_specs=tcast_specs, **kwargs)

        return type(f"{cls.__name__}_tcast", (cls,), {"__init__": __init__})

    for torchm, tcastm in SUPPORTED_MODULES.items():
        torch.nn.__dict__[torchm] = torch_to_tcast_module(tcastm)


def mixed_precision_injector(model, tcast_specs):
    model_mixed = copy.deepcopy(model)
    for name, module in model_mixed.named_modules():
        if name in tcast_specs:
            if isinstance(module, torch.nn.Linear):
                model_mixed.__dict__[name] = Linear(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    tcast_specs=tcast_specs[name],
                    pre_weights=module.weight,
                    pre_bias=module.bias,
                )
            if isinstance(module, torch.nn.Conv2d):
                model_mixed.__dict__[name] = Conv2d(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size,
                    module.stride,
                    module.padding,
                    module.dilation,
                    module.groups,
                    module.bias is not None,
                    module.padding_mode,
                    tcast_specs=tcast_specs[name],
                    pre_weights=module.weight,
                    pre_bias=module.bias,
                )
    return model_mixed


SUPPORTED_MODULES = {"Linear": Linear, "Conv2d": Conv2d}
