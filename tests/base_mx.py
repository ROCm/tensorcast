#!/usr/bin/env python
# tests/base_tests_mx.py: test tcast against MX (microxscaling)
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import torch

import tcast
import tests.utils as U

U.dependency_assert("mx")

from mx.elemwise_ops import quantize_elemwise_op
from mx.mx_ops import quantize_mx_op
from mx.specs import MxSpecs

# TODO(ericd): Add 2d tile support


def supported(dtype: tcast.DataType =None, roundmode: str = None, scalemode: str = None) -> bool:
    """Check if the given TensorCast DataType is supported by MX."""
    if not dtype.is_mxfp: #  or (dtype.is_unscaled and dtype.nspec.ebits in (5, 8))):
        return False
    if roundmode and roundmode not in ("away", "even"):
        return False
    if scalemode and scalemode != "floor":
        return False
    return True


def convert_tcast_dtype_to_mx(dtype: tcast.DataType, roundmode: str = None, scalemode: str = None) -> MxSpecs:
    """Convert a TensorCast DataType to an MX DataType."""
    assert supported(dtype, roundmode, scalemode), f"Unsupported dtype {dtype} for MX"
    mx_specs = MxSpecs()
    roundmap = {"away": "nearest", "even": "even"}
    mx_specs["round"] = roundmode = roundmap[roundmode]
    if dtype.is_mxfp:
        mx_specs["block_size"] = dtype.sspec.tile1
    else:
        mx_specs["fp" if dtype.nspec.ebits == 5 else "bfloat"] = dtype.nspec.bits
    return mx_specs


def base_torchcast_mx(dtype: str | tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str, transpose: bool):
    tensor = torch.randn(*shape).cuda().float()
    dtype = tcast.datatype(name=str(dtype))
    mx_specs = convert_tcast_dtype_to_mx(dtype, roundmode)
    if dtype.is_mxfp:
        axis = -1 if transpose else 0
        elem_format = f"fp{dtype.nspec.bits}_e{dtype.nspec.ebits}m{dtype.nspec.mbits}"
        tensor_mx = quantize_mx_op(tensor, mx_specs, elem_format=elem_format, axes=axis)
    else:
        tensor_mx = quantize_elemwise_op(tensor, mx_specs)
    tcast.initialize(roundmode=roundmode, computemode="torch", castmode="virtual")
    tensor_tc = tcast.cast(tensor, dtype=dtype)
    torch.testing.assert_close(tensor_mx, tensor_tc)
