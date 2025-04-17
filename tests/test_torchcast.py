#!/usr/bin/env python
# tests/test_torchcast.py: PyTest torchcast with mx/sqt/tcastv2/torch
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import pytest
import torch

import tcast
import tests.base_misc as M
import tests.base_mx as MX
import tests.base_sqt as SQT
import tests.base_torch as T
import tests.base_v2 as V2
import tests.utils as U

INPUT = U.TORCH_DTYPES_32_16
SHAPES = U.SHAPES_2D
ROUND = U.ROUNDMODE_EA

# These tests are for virtual castmode and torch compute mode
# TODO(ericd): check to see which of these MX supports


def _compare_v2(dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str, transpose: bool):
    if not U.V2_AVAILABLE:
        pytest.skip("V2 not available, skipping test.")
    if not V2.supported(dtype=dtype):
        pytest.skip(f"V2 does not support {str(dtype)}, skipping test.")
    V2.base_torchcast_v2(dtype, torch_dtype, shape, roundmode)


def _compare_sqt(dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str, transpose: bool):
    if not U.SQT_AVAILABLE:
        pytest.skip("SQT not available, skipping test.")
    if not SQT.supported(dtype=dtype):
        pytest.skip(f"SQT does not support {str(dtype)}, skipping test.")
    if not SQT.supported(roundmode=roundmode):
        pytest.skip(f"SQT does not support roundmode {roundmode}, skipping test.")
    SQT.base_torchcast_sqt(dtype, torch_dtype, shape, roundmode, transpose)


def _compare_mx(dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str, transpose: bool):
    if not U.MX_AVAILABLE:
        pytest.skip("MX not available, skipping test.")
    if not MX.supported(dtype=dtype):
        pytest.skip(f"MX does not support {str(dtype)}, skipping test.")
    if not MX.supported(roundmode=roundmode):
        pytest.skip(f"MX does not support roundmode {roundmode}, skipping test.")
    MX.base_torchcast_mx(dtype, torch_dtype, shape, roundmode, transpose)


def _compare_pytorch(dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str, transpose: bool = False):
    if dtype.nspec.torch_dtype == torch_dtype:
        pytest.skip("tcast dtype is the same as torch_dtype")
    if roundmode != "even":
        pytest.skip("PyTorch only supports roundmode even")
    T.base_torchcast_torch(dtype, torch_dtype, shape, roundmode)


def _torchcast_generic(
    compare: str, dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str, transpose: bool = False
):
    """Test torchcast with virtual mode and tcast dtypes selected by the specific test function."""
    if compare == "v2":
        _compare_v2(dtype, torch_dtype, shape, roundmode, transpose)
    elif compare == "sqt":
        _compare_sqt(dtype, torch_dtype, shape, roundmode, transpose)
    elif compare == "mx":
        _compare_mx(dtype, torch_dtype, shape, roundmode, transpose)
    elif compare == "pytorch":
        _compare_pytorch(dtype, torch_dtype, shape, roundmode, transpose)
    else:
        raise ValueError(f"Unknown compare mode: {compare}")


@pytest.mark.parametrize("compare", ["v2", "sqt", "mx"])
@pytest.mark.parametrize("dtype", tcast.DataType.gather_registered(lambda x: x.is_unscaled and x.nspec.bits <= 16))
@pytest.mark.parametrize("torch_dtype", INPUT)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("roundmode", ROUND)
def test_torchcast_unscaled(compare: str, dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str):
    """Test torchcast with virtual mode and unscaled datatypes."""
    _torchcast_generic(compare, dtype, torch_dtype, shape, roundmode)


@pytest.mark.parametrize("compare", ["v2", "sqt"])
@pytest.mark.parametrize("dtype", tcast.DataType.gather_registered(lambda x: x.is_tensor and x.nspec.bits <= 8))
@pytest.mark.parametrize("torch_dtype", INPUT)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("roundmode", ROUND)
def test_torchcast_tensor(compare: str, dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str):
    """Test torchcast virtual mode and tensor scaled tcast dtypes."""
    _torchcast_generic(compare, dtype, torch_dtype, shape, roundmode)


@pytest.mark.parametrize("compare", ["v2"])
@pytest.mark.parametrize("dtype", tcast.DataType.gather_registered(lambda x: x.is_channel and x.nspec.bits <= 8))
@pytest.mark.parametrize("torch_dtype", INPUT)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("roundmode", ROUND)
@pytest.mark.parametrize("transpose", [False, True])
def test_torchcast_channel(
    compare: str, dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str, transpose: bool
):
    """Test torchcast virtual mode and channel scaled tcast dtypes."""
    _torchcast_generic(compare, dtype, torch_dtype, shape, roundmode, transpose)


@pytest.mark.parametrize("compare", ["v2", "sqt"])
@pytest.mark.parametrize("dtype", tcast.DataType.gather_registered(lambda x: x.is_tile and x.nspec.bits == 8 and not x.is_mxfp))
@pytest.mark.parametrize("torch_dtype", INPUT)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("roundmode", ROUND)
@pytest.mark.parametrize("transpose", [False, True])
def test_torchcast_tile_fp8(
    compare: str, dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str, transpose: bool
):
    """Test torchcast with FP8 tiled datatypes."""
    _torchcast_generic(compare, dtype, torch_dtype, shape, roundmode, transpose)


@pytest.mark.parametrize("compare", ["v2", "sqt"])
@pytest.mark.parametrize("dtype", tcast.DataType.gather_registered(lambda x: x.is_tile and x.nspec.bits < 8))
@pytest.mark.parametrize("torch_dtype", INPUT)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("roundmode", ROUND)
@pytest.mark.parametrize("transpose", [False, True])
def test_torchcast_tile(
    compare: str, dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str, transpose: bool
):
    """Test torchcast with smaller tile scaled datatypes."""
    _torchcast_generic(compare, dtype, torch_dtype, shape, roundmode, transpose)


@pytest.mark.parametrize("compare", ["v2", "sqt", "mx"])
@pytest.mark.parametrize("dtype", tcast.DataType.gather_registered(lambda x: x.is_mxfp))
@pytest.mark.parametrize("torch_dtype", INPUT)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("roundmode", ROUND)
@pytest.mark.parametrize("transpose", [False, True])
def test_torchcast_tile_mxfp(
    compare: str, dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str, transpose: bool
):
    """Test torchcast with MXFP datatypes."""
    _torchcast_generic(compare, dtype, torch_dtype, shape, roundmode, transpose)


@pytest.mark.parametrize("compare", ["pytorch"])
@pytest.mark.parametrize("dtype", tcast.DataType.gather_registered(lambda x: x.nspec.torch_dtype is not None))
@pytest.mark.parametrize("torch_dtype", INPUT)
@pytest.mark.parametrize("shape", SHAPES)
def test_torchcast_torch(compare: str, dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int]):
    """Unscaled torchcast with virtual mode and tcast dtypes that have a torch dtype, compare to torch ".to"."""
    _torchcast_generic(compare, dtype, torch_dtype, shape, "even")


@pytest.mark.parametrize("dtype", tcast.DataType.gather_registered(lambda x: x.is_unscaled and x.nspec.bits <= 16))
@pytest.mark.parametrize("torch_dtype", INPUT)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("roundmode", ROUND)
@pytest.mark.parametrize("transpose", [False, True])
def test_torchcast_quantization_error(
    dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str, transpose: bool
):
    """Test torchcast with virtual mode and unscaled datatypes."""
    if not tcast.TorchCast.supports(dtype=dtype):
        pytest.skip(f"tcast does not support {str(dtype)}, skipping test.")
    M.base_quantization_error(dtype, torch_dtype, shape, roundmode, transpose)


@pytest.mark.parametrize("dtype", tcast.DataType.gather_registered(lambda x: x.is_unscaled and x.nspec.bits <= 16))
@pytest.mark.parametrize("torch_dtype", INPUT)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("roundmode", ROUND)
@pytest.mark.parametrize("transpose", [False, True])
def test_torchcast_quantization_unscaled(
    dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str, transpose: bool
):
    """Test torchcast with virtual mode and unscaled datatypes."""
    if not tcast.TorchCast.supports(dtype=dtype):
        pytest.skip(f"tcast does not support {str(dtype)}, skipping test.")
    M.base_quantization_representable(dtype, torch_dtype, shape, roundmode, transpose)


@pytest.mark.parametrize("dtype", tcast.DataType.gather_registered(lambda x: x.is_tensor and x.nspec.bits <= 8))
@pytest.mark.parametrize("torch_dtype", INPUT)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("roundmode", ROUND)
@pytest.mark.parametrize("transpose", [False, True])
def test_torchcast_quantization_tensor(
    dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str, transpose: bool
):
    """Test torchcast with virtual mode and tensor scaled datatypes."""
    if not tcast.TorchCast.supports(dtype=dtype):
        pytest.skip(f"tcast does not support {str(dtype)}, skipping test.")
    M.base_quantization_representable(dtype, torch_dtype, shape, roundmode, transpose)


@pytest.mark.parametrize("dtype", tcast.DataType.gather_registered(lambda x: x.is_channel and x.nspec.bits <= 8))
@pytest.mark.parametrize("torch_dtype", INPUT)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("roundmode", ROUND)
@pytest.mark.parametrize("transpose", [False, True])
def test_torchcast_quantization_channel(
    dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str, transpose: bool
):
    """Test torchcast with virtual mode and channel scaled datatypes."""
    if not tcast.TorchCast.supports(dtype=dtype):
        pytest.skip(f"tcast does not support {str(dtype)}, skipping test.")
    M.base_quantization_representable(dtype, torch_dtype, shape, roundmode, transpose)


@pytest.mark.parametrize("dtype", tcast.DataType.gather_registered(lambda x: x.is_tile and x.nspec.bits == 8 and not x.is_mxfp))
@pytest.mark.parametrize("torch_dtype", INPUT)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("roundmode", ROUND)
@pytest.mark.parametrize("transpose", [False, True])
def test_torchcast_quantization_tile_fp8(
    dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str, transpose: bool
):
    """Test torchcast with virtual mode and unscaled datatypes."""
    if not tcast.TorchCast.supports(dtype=dtype):
        pytest.skip(f"tcast does not support {str(dtype)}, skipping test.")
    M.base_quantization_representable(dtype, torch_dtype, shape, roundmode, transpose)


@pytest.mark.parametrize("dtype", tcast.DataType.gather_registered(lambda x: x.is_tile and x.nspec.bits < 8 and not x.is_mxfp))
@pytest.mark.parametrize("torch_dtype", INPUT)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("roundmode", ROUND)
@pytest.mark.parametrize("transpose", [False, True])
def test_torchcast_quantization_tile(
    dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str, transpose: bool
):
    """Test torchcast with virtual mode and unscaled datatypes."""
    if not tcast.TorchCast.supports(dtype=dtype):
        pytest.skip(f"tcast does not support {str(dtype)}, skipping test.")
    M.base_quantization_representable(dtype, torch_dtype, shape, roundmode, transpose)


@pytest.mark.parametrize("dtype", tcast.DataType.gather_registered(lambda x: x.is_mxfp))
@pytest.mark.parametrize("torch_dtype", INPUT)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("roundmode", ROUND)
@pytest.mark.parametrize("transpose", [False, True])
def test_torchcast_quantization_mxfp(
    dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str, transpose: bool
):
    """Test torchcast with virtual mode and unscaled datatypes."""
    if not tcast.TorchCast.supports(dtype=dtype):
        pytest.skip(f"tcast does not support {str(dtype)}, skipping test.")
    M.base_quantization_representable(dtype, torch_dtype, shape, roundmode, transpose)
