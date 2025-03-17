#!/usr/bin/env python
# tests/test_torch_virtual.py: test castmode == virtual and computemode == torch
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import torch

import tcast
from tests.utils import dependency_assert

dependency_assert("mx")


def mx_quantize(tensor: torch.Tensor, dtype: tcast.DataType, roundmode: str) -> torch.Tensor:
    from mx.elemwise_ops import quantize_elemwise_op
    from mx.mx_ops import quantize_mx_op
    from mx.specs import MxSpecs

    mx_specs = MxSpecs()
    roundmap = {"away": "nearest", "even": "even"}
    if roundmode not in roundmap:
        raise ValueError(f"MX only supports 'away' or 'even' rounding, got {roundmode}")
    mx_specs["round"] = roundmode = roundmap[roundmode]
    if dtype.name.startswith("mxfp"):
        mx_specs["block_size"] = dtype.sspec.tile1
        elem_format = f"fp{dtype.nspec.bits}_e{dtype.nspec.ebits}m{dtype.nspec.mbits}"
        tensor_mx = quantize_mx_op(tensor, mx_specs, elem_format=elem_format, axes=-1)
    else:
        if dtype.nspec.ebits not in (5, 8):
            raise ValueError(f"MX only supports 5 or 8 exponent bits, got {dtype.nspec.ebits}")
        mx_specs["fp" if dtype.nspec.ebits == 5 else "bfloat"] = dtype.nspec.bits
        tensor_mx = quantize_elemwise_op(tensor, mx_specs)
    return tensor_mx


def test_mx_unscaled_datatypes_base(shape: tuple[int], dtype: str | tcast.DataType, roundmode: str, computemode: str):
    tensor = torch.randn(*shape).cuda().float()
    dtype = tcast.datatype(name=str(dtype))
    tensor_mx = mx_quantize(tensor, dtype, roundmode)
    castmode = "virtual" if computemode == "torch" else "actual"
    tensor_tcast = tcast.cast(tensor, dtype=dtype, castmode=castmode, roundmode=roundmode, computemode=computemode)
    if isinstance(tensor, tcast.Tensor):
        tensor_tcast = tensor_tcast.output
    assert type(tensor_mx) is type(
        tensor_tcast
    ), f"MX ({type(tensor_mx)}) and TensorCast ({type(tensor_tcast)}) datatypes do not match"
    assert (
        tensor_mx.dtype == tensor_tcast.dtype
    ), f"MX ({str(tensor_mx.dtype)}) and TensorCast ({str(tensor_tcast.dtype)}) datatypes do not match"
    torch.testing.assert_close(tensor_mx, tensor_tcast)


def test_mx_mxfp_datatypes_base(shape: tuple[int], dtype: str | tcast.DataType, roundmode: str, computemode: str):
    tensor = torch.randn(*shape).cuda().float()
    dtype = tcast.datatype(name=str(dtype))
    tensor_mx = mx_quantize(tensor, dtype, roundmode)
    castmode = "virtual" if computemode == "torch" else "actual"
    tensor_tcast = tcast.cast(
        tensor, dtype=dtype, castmode=castmode, roundmode=roundmode, scalemode="floor", computemode=computemode
    )
    if isinstance(tensor, tcast.Tensor):
        tensor_tcast = tensor_tcast.output
    torch.testing.assert_close(tensor_mx, tensor_tcast)
