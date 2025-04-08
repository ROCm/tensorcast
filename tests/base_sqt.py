#!/usr/bin/env python
# tests/base_sqt.py: body of test code comparing with SQT
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

# TODO(ericd): implement upcast dependencies to enable tests

import torch

import tcast
from tests.utils import dependency_assert

dependency_assert("sqt")

import sqt.core as sqt

RMODES = ("zero", "away", "ceil", "floor", "nearest", "even")


def supported(dtype: tcast.DataType = None, roundmode: str = None, scalemode: str = None) -> bool:
    """Check if the given TensorCast DataType is supported by SQT."""
    if dtype:
        if dtype.is_2d or dtype.is_subtile or dtype.is_channel or not (dtype.is_unscaled or dtype.sspec.scale.is_exponent):
            return False
    if roundmode and roundmode not in RMODES:
        return False
    if scalemode and scalemode != "floor":
        return False
    return True


def convert_tcast_dtype_to_sqt(dtype: tcast.DataType, roundmode: str = None, scalemode: str = None) -> tuple[sqt.DataType, str]:
    """Convert a TensorCast DataType to an SQT DataType."""
    assert supported(dtype, roundmode, scalemode), f"Unsupported dtype {dtype} for SQT"
    # "(fp|ts|bs|be|ap)_e(\d+)m(\d+)b(\d+)(_n[irs]+)?(_b\d+)?(_p\d+)?"
    category = (
        "fp"
        if dtype.is_unscaled
        else "ts"
        if dtype.is_tensor
        else "be"
        if dtype.is_tile and dtype.nspec.is_int
        else "bs"
        if dtype.nspec.is_float
        else None
    )
    assert category, f"Unsupported dtype {dtype} for SQT"
    nspec = dtype.nspec
    ebits, mbits, bias = 8 if nspec.ebits == 1 else nspec.ebits, nspec.mbits, nspec.bias
    ncode = "_ni" if nspec.is_int else "_nr" if nspec.infnan == tcast.get_enum(tcast.InfNaN, "ieee") else ""
    sqt_code = f"{category}_e{ebits}m{mbits}b{bias}{ncode}"
    assert sqt.DataType.is_valid_code(sqt_code), f"Invalid SQT code {sqt_code}"
    return sqt.DataType(code=sqt_code), roundmode


def base_torchcast_sqt(dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str, transpose: bool):
    """Test torchcast vs SQT."""
    if isinstance(dtype, str):
        dtype = tcast.DataType(dtype)
    assert isinstance(dtype, tcast.DataType), f"Invalid dtype {dtype}"
    sqt_dtype, sqt_roundmode = convert_tcast_dtype_to_sqt(dtype, roundmode)
    sqt.initialize_core(roundmode=sqt_roundmode)
    tcast.initialize(roundmode=roundmode, computemode="torch", castmode="virtual")
    tensor = torch.randn(shape, dtype=torch_dtype)
    assert tensor.is_cuda, "default tensor type must be cuda"
    axis = 0 if transpose else -1
    tensor_sqt = sqt.quantize(tensor, axis, sqt_dtype)
    tensor_tc = tcast.cast(tensor, dtype, transpose_scale=transpose)
    torch.testing.assert_close(tensor_tc, tensor_sqt)
