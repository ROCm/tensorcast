#!/usr/bin/env python
# tests/base_tests_v2.py: implementation of test code comparing with tensorcast v2 branch
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import torch

import tcast
from tests.utils import dependency_assert

dependency_assert("v2")

import tcastv2


def supported(dtype: tcast.DataType) -> bool:
    """Check for datatype support (otherwise skip test)."""
    return not dtype.is_2d


def convert_tcast_dtype_to_v2(dtype: tcast.DataType):
    """Convert a triton DataType to a v2 DataType."""
    if dtype.is_codebook:
        raise NotImplementedError("Codebook not supported in v2/triton conversion")
    if dtype.nspec.torch_dtype:
        ncode = dtype.nspec.torch_dtype
    else:
        n = dtype.nspec
        bias = "" if n.std_bias else f"b{n.bias}"
        ncode = f"e{n.ebits}m{n.mbits}{bias}{n.infnan.suffix()}"
    if dtype.is_unscaled:
        scode = None
    else:
        if dtype.sspec.is_2d:
            raise NotImplementedError("2D sspec not supported in v2/triton conversion")
        s = dtype.sspec
        if s.scale.is_exponent:
            scode = f"e{s.scale.ebits}m{s.scale.mbits}"
        else:
            infnan = s.scale.infnan.suffix()
            scode = f"e{s.scale.ebits}m{s.scale.mbits}b{s.scale.bias}{infnan}"
        if s.zero:
            if s.zero.is_int:
                scode += f"_int{s.zero.bits}"
            elif s.zero.is_float:
                infnan = s.zero.infnan.suffix()
                scode += f"_e{s.zero.ebits}m{s.zero.mbits}b{s.zero.bias}{infnan}"
            else:
                raise ValueError(f"Unknown zero type {s.zero}")
        if not s.is_tensor:
            scode += f"_t{0 if s.is_channel else s.tile1}"
    return tcastv2.DataType(nspec=ncode, sspec=scode)


def base_torchcast_v2(
        dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str, transpose: bool = False
    ):
    """Test torchcast with virtual."""
    if isinstance(dtype, str):
        dtype = tcast.DataType(dtype)
    tcastv2.initialize(roundmode=roundmode, compmode="torch")
    tcast.initialize(roundmode=roundmode, computemode="torch", castmode="virtual")
    tensor = torch.randn(shape, device="cuda", dtype=torch_dtype)
    dtype_v2 = convert_tcast_dtype_to_v2(dtype)
    assert isinstance(dtype_v2, tcastv2.DataType)
    tensor_v2 = tcastv2.cast(tensor, dtype_v2).tensor
    tensor_tc = tcast.cast(tensor, dtype)
    torch.testing.assert_close(tensor_v2, tensor_tc)
