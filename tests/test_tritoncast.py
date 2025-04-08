#!/usr/bin/env python
# tests/test_tritoncast.py: PyTest tritoncast actual and virtual with tritoncast virtual
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import pytest
import torch

import tcast
import tests.base_tritoncast as TRI
import tests.utils as U

INPUT = U.TORCH_DTYPES_32_16
SHAPES = U.SHAPES_2D
ROUND = U.ROUNDMODE_EA


def _tritoncast_generic(
    dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str, castmode: str, transpose: bool = False
):
    if castmode == "virtual":
        TRI.base_tritoncast_virtual(dtype, torch_dtype, shape, roundmode, transpose)
    if castmode == "actual":
        TRI.base_tritoncast_actual(dtype, torch_dtype, shape, roundmode, transpose)


@pytest.mark.parametrize("dtype", tcast.DataType.gather_registered(lambda x: x.is_unscaled and x.nspec.bits <= 16))
@pytest.mark.parametrize("torch_dtype", INPUT)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("roundmode", ROUND)
@pytest.mark.parametrize("castmode", ["virtual", "actual"])
def test_tritoncast_unscaled(dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str, castmode: str):
    """Test tritoncast actual/virtual with torchcast virtual on unscaled datatypes."""
    _tritoncast_generic(dtype, torch_dtype, shape, roundmode, castmode)


@pytest.mark.parametrize("dtype", tcast.DataType.gather_registered(lambda x: x.is_tensor and x.nspec.bits <= 8))
@pytest.mark.parametrize("torch_dtype", INPUT)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("roundmode", ROUND)
@pytest.mark.parametrize("castmode", ["virtual", "actual"])
def test_tritoncast_tensor(dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str, castmode: str):
    """Test tritoncast actual/virtual  and tensor scaled tcast dtypes."""
    _tritoncast_generic(dtype, torch_dtype, shape, roundmode, castmode)


@pytest.mark.parametrize("dtype", tcast.DataType.gather_registered(lambda x: x.is_channel and x.nspec.bits <= 8))
@pytest.mark.parametrize("torch_dtype", INPUT)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("roundmode", ROUND)
@pytest.mark.parametrize("castmode", ["virtual", "actual"])
@pytest.mark.parametrize("transpose", [False, True])
def test_tritoncast_channel(
    dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str, castmode: str, transpose: bool
):
    """Test tritoncast actual/virtual  and channel scaled tcast dtypes."""
    _tritoncast_generic(dtype, torch_dtype, shape, roundmode, castmode, transpose)


@pytest.mark.parametrize("dtype", tcast.DataType.gather_registered(lambda x: not x.is_mxfp and x.is_tile and x.nspec.bits == 8))
@pytest.mark.parametrize("torch_dtype", INPUT)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("roundmode", ROUND)
@pytest.mark.parametrize("castmode", ["virtual", "actual"])
@pytest.mark.parametrize("transpose", [False, True])
def test_tritoncast_tile_fp8(
    dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str, castmode: str, transpose: bool
):
    """Test tritoncast actual/virtual with FP8 tiled dtypes."""
    _tritoncast_generic(dtype, torch_dtype, shape, roundmode, castmode, transpose)


@pytest.mark.parametrize("dtype", tcast.DataType.gather_registered(lambda x: not x.is_mxfp and x.is_tile and x.nspec.bits < 8))
@pytest.mark.parametrize("torch_dtype", INPUT)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("roundmode", ROUND)
@pytest.mark.parametrize("castmode", ["virtual", "actual"])
@pytest.mark.parametrize("transpose", [False, True])
def test_tritoncast_tile(
    dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str, castmode: str, transpose: bool
):
    """Test tritoncast actual/virtual with smaller tile scaled datatypes."""
    _tritoncast_generic(dtype, torch_dtype, shape, roundmode, castmode, transpose)


@pytest.mark.parametrize("dtype", tcast.DataType.gather_registered(lambda x: x.is_mxfp))
@pytest.mark.parametrize("torch_dtype", INPUT)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("roundmode", ROUND)
@pytest.mark.parametrize("castmode", ["virtual", "actual"])
@pytest.mark.parametrize("transpose", [False, True])
def test_tritoncast_tile_mxfp(
    dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str, castmode: str, transpose: bool
):
    """Test tritoncast actual/virtual with MXFP datatypes."""
    _tritoncast_generic(dtype, torch_dtype, shape, roundmode, castmode, transpose)
