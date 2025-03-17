#!/usr/bin/env python
# tests/test_torch.py: test quantization against PyTorch
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import pytest
import torch

import tcast

logger = tcast.get_logger("tcast_unittest")

FP8_AVAILABLE = tcast.utils.is_float8_available()


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("shape", [(1024, 1024), (8, 128, 64)])
@pytest.mark.parametrize("computemode", ["torch", "triton"])
@pytest.mark.parametrize("castmode", ["virtual"])
def test_torch_unscaled_datatypes(shape: tuple[int], dtype: str | tcast.DataType, computemode: str, castmode: str):
    tensor = torch.randn(*shape).cuda().float()
    dtype = tcast.datatype(name=str(dtype))
    tensor_pt = tensor.to(dtype=dtype.nspec.torch_dtype)
    tensor_tcast = tcast.cast(tensor, dtype=dtype, castmode=castmode, roundmode="even", computemode=computemode)
    if castmode == "actual":
        tensor_tcast = tensor_tcast * dtype.nspec.scale
        tcast.cast(tensor_tcast.output, dtype=dtype, castmode=castmode, oundmode="even", computemode=computemode)
    torch.testing.assert_close(tensor_pt, tensor_tcast)


@pytest.mark.parametrize("dtype", ["float8_e5m2", "float8_e4m3fn"])
@pytest.mark.parametrize("shape", [(1024, 1024), (8, 128, 64)])
@pytest.mark.parametrize("computemode", ["torch", "triton"])
@pytest.mark.skipif(not FP8_AVAILABLE, reason="PyTorch FP8 is not available.")
def test_torch_fp8_datatypes(shape: tuple[int], dtype: str | tcast.DataType, computemode: str):
    tensor = torch.randn(*shape).cuda().float()
    dtype = tcast.datatype(name=str(dtype))
    scale = dtype.nspec.finfo.maxfloat / tensor.abs().max()
    tensor = tensor * scale
    tensor_pt = tensor.to(dtype=dtype.nspec.torch_dtype)
    tensor_tcast = tcast.cast(tensor, dtype=dtype, roundmode="even", computemode=computemode)
    torch.testing.assert_close(tensor_pt, tensor_tcast)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
