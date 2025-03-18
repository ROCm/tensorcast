#!/usr/bin/env python
# tests/base_tests_v2.py: body of test code comparing with tensorcast v2 branch
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import torch

import tcast
from tests.utils import convert_triton_dtype_to_v2, dependency_assert

dependency_assert("v2")

import tcastv2

logger = tcast.get_logger("tcast_unittest")


def test_torchcast_virtual_unscaled_base(
    dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str, computemode: str
):
    """Test torchcast with virtual mode and unscaled data."""
    print(dtype, isinstance(dtype, str), isinstance(dtype, tcast.DataType))
    if isinstance(dtype, str):
        dtype = tcast.DataType(dtype)
    print(dtype, isinstance(dtype, str), isinstance(dtype, tcast.DataType))
    tcastv2.initialize(roundmode=roundmode, compmode=computemode)
    tcast.initialize(roundmode=roundmode, computemode=computemode, castmode="virtual")
    tensor = torch.randn(shape, device="cuda", dtype=torch_dtype)
    dtype_v2 = convert_triton_dtype_to_v2(dtype)
    assert isinstance(dtype_v2, tcastv2.DataType)
    tensor_v2 = tcastv2.cast(tensor, dtype_v2).tensor
    tensor_tc = tcast.cast(tensor, dtype)
    torch.testing.assert_close(tensor_v2, tensor_tc)


def test_torchcast_virtual_unscaled_torch_base(
    dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str, computemode: str
):
    """Test torchcast with virtual mode and tcast dtypes that have a torch dtype, compare to torch ".to"."""
    tcast.initialize(roundmode=roundmode, computemode=computemode, castmode="virtual")
    tensor = torch.randn(shape, device="cuda", dtype=torch_dtype)
    tensor_pt = tensor.to(dtype.nspec.torch_dtype).to(tensor.dtype)
    tensor_tc = tcast.cast(tensor, dtype)
    torch.testing.assert_close(tensor_pt, tensor_tc)
