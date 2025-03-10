#!/usr/bin/env python
# tests/test_mx.py: test quantization against MX library
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import importlib

import pytest
import torch

import tcast

logger = tcast.get_logger("tcast_unittest")


MX_AVAILABLE = importlib.util.find_spec("mx") is not None


def mx_quantize(tensor: torch.Tensor, dtype: tcast.DataType, roundmode: str) -> torch.Tensor:
    assert MX_AVAILABLE, "MX library is not available. github.com/microsoft/microxcaling"
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


@pytest.mark.parametrize("shape", [(1024, 1024), (8, 128, 64)])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("roundmode", ["even", "away"])
@pytest.mark.parametrize("computemode", ["torch", "triton"])
@pytest.mark.skipif(not MX_AVAILABLE, reason="MX library is not available. github.com/microsoft/microxcaling")
def test_mx_unscaled_datatypes(shape: tuple[int], dtype: str | tcast.DataType, roundmode: str, computemode: str):
    tensor = torch.randn(*shape).cuda().float()
    dtype = tcast.datatype(name=str(dtype))
    tensor_mx = mx_quantize(tensor, dtype, roundmode)
    castmode = "virtual" if computemode == "torch" else "actual"
    tensor_tcast = tcast.cast(tensor, dtype=dtype, castmode=castmode, roundmode=roundmode, computemode=computemode)
    if isinstance(tensor, tcast.Tensor):
        tensor_tcast = tensor_tcast.output
    torch.testing.assert_close(tensor_mx, tensor_tcast)


@pytest.mark.parametrize("shape", [(1024, 1024), (8, 128, 64)])
@pytest.mark.parametrize("dtype", ["mxfp8e5", "mxfp8e4", "mxfp6e3", "mxfp6e2", "mxfp4e2"])
@pytest.mark.parametrize("roundmode", ["even", "away"])
@pytest.mark.parametrize("computemode", ["torch", "triton"])
@pytest.mark.skipif(not MX_AVAILABLE, reason="MX library is not available. github.com/microsoft/microxcaling")
def test_mx_mxfp_datatypes(shape: tuple[int], dtype: str | tcast.DataType, roundmode: str, computemode: str):
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
