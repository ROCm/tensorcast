#!/usr/bin/env python
# tests/test_torch_virtual.py: test castmode == virtual and computemode == torch
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import pytest
import torch

import tcast
import tests.base_tests_v2 as B
from tests.utils import SHAPES_2D, TORCH_DTYPES_16, TORCH_DTYPES_32_16


@pytest.mark.parametrize("dtype", tcast.DataType.gather_registered(lambda x: x.is_unscaled and x.nspec.bits <= 16))
@pytest.mark.parametrize("torch_dtype", TORCH_DTYPES_32_16)
@pytest.mark.parametrize("shape", SHAPES_2D)
@pytest.mark.parametrize("roundmode", ["even"])
@pytest.mark.parametrize("computemode", ["torch"])
@pytest.mark.parametrize("castmode", ["virtual"])
def test_torchcast_virtual_unscaled(
    dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str, computemode: str, castmode: str
):
    """Test torchcast with virtual mode and unscaled data."""
    B.test_torchcast_virtual_unscaled_base(dtype, torch_dtype, shape, roundmode, computemode, castmode)


@pytest.mark.parametrize("dtype", tcast.DataType.gather_registered(lambda x: x.nspec.torch_dtype is not None))
@pytest.mark.parametrize("torch_dtype", TORCH_DTYPES_16)
@pytest.mark.parametrize("shape", SHAPES_2D)
@pytest.mark.parametrize("roundmode", ["even"])
@pytest.mark.parametrize("computemode", ["torch"])
@pytest.mark.parametrize("castmode", ["virtual"])
def test_torchcast_virtual_unscaled_torch(
    dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str, computemode: str, castmode: str
):
    """Test torchcast with virtual mode and tcast dtypes that have a torch dtype, compare to torch ".to"."""
    if dtype.nspec.torch_dtype == torch_dtype:
        pytest.skip("tcast dtype is the same as torch_dtype")
    B.test_torchcast_virtual_unscaled_torch_base(dtype, torch_dtype, shape, roundmode, computemode, castmode)
