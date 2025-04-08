#!/usr/bin/env python
# tests/base_tests_torch.py: implementation of test code comparing with PyTorch
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import torch

import tcast


def supports(dtype: tcast.DataType):
    """Check if the dtype is supported by PyTorch and TensorCast."""
    return (
        dtype.is_unscaled
        and dtype.nspec.is_float
        and dtype.nspec.torch_dtype is not None
        and dtype.nspec.torch_dtype.itemsize <= 2
    )


def _generic_test(dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str, computemode: str):
    """Generic test for both tritoncast and torchcast."""
    tcast.initialize(roundmode=roundmode, computemode=computemode, castmode="virtual")
    tensor = torch.randn(shape, device="cuda", dtype=torch.float32)
    if torch_dtype.itemsize == 1:
        scale = dtype.nspec.finfo.maxfloat / torch.max(torch.abs(tensor))
        tensor = tensor * scale
    tensor_pt = tensor.to(dtype.nspec.torch_dtype).to(tensor.dtype)
    tensor_tc = tcast.cast(tensor, dtype)
    torch.testing.assert_close(tensor_pt, tensor_tc)


def base_torchcast_torch(dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str = None):
    """Scales the tensor to the FP8 range, then casts to the target dtype."""
    _generic_test(dtype, torch_dtype, shape, roundmode=roundmode, computemode="torch")


def base_tritoncast_torch(dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int]):
    """Scales the tensor to the FP8 range, then casts to the target dtype."""
    _generic_test(dtype, torch_dtype, shape, computemode="triton")
