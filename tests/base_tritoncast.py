#!/usr/bin/env python
# tests/base_tritoncast.py: implementation of test code comparing TritonCase against TorchCast virtual
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import torch

import tcast


def base_tritoncast_virtual(dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str):
    """Test tritoncast "virtual" vs torchcast "virtual"."""
    tcast.initialize(roundmode=roundmode, castmode="virtual")
    tensor = torch.randn(shape, dtype=torch_dtype)
    tensor_tor = tcast.cast(tensor, dtype, computemode="torch")
    tensor_tri = tcast.cast(tensor, dtype, computemode="triton")
    torch.testing.assert_close(tensor_tor, tensor_tri)


def base_tritoncast_actual(dtype: tcast.DataType, torch_dtype: torch.dtype, shape: tuple[int], roundmode: str):
    """Test tritoncast "actual" + upcast vs tritoncast "virtual"."""
    tcast.initialize(roundmode=roundmode, computemode="triton", castmode="virtual")
    tensor = torch.randn(shape, dtype=torch_dtype)
    tensor_actual = tcast.cast(tensor, dtype, castmode="actual")
    tensor_actual = tcast.upcast(tensor_actual)
    tensor_virtual = tcast.cast(tensor, dtype, castmode="virtual")
    torch.testing.assert_close(tensor_actual, tensor_virtual)
