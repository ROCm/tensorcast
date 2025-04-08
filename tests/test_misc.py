#!/usr/bin/env python
# tests/test_misc.py: test LP configuration, incoherence, etc.
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import pytest
import torch

import tests.base_misc as M
import tests.utils as U

# TODO(ericd): expand shapes to include 3D and 4D tensors
# TODO(ericd): add stochastic error with amended atol/rtol (never exact)

INPUT = U.TORCH_DTYPES_32_16
SHAPES = U.SHAPES_2D
ROUND = U.ROUNDMODE_EA


@pytest.mark.parametrize("torch_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("size", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("walsh, randomize", [(False, False), (True, False), (True, True)])
def test_incoherence_matrix(torch_dtype, size, walsh, randomize):
    """Test if a matrix is a Hadamard or Walsh-Hadamard matrix."""
    M.base_incoherence_matrix(torch_dtype, size, walsh, randomize)


@pytest.mark.parametrize("torch_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("size", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("odim", [1, 2, 4])
@pytest.mark.parametrize("walsh, randomize", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("float32", [False, True])
@pytest.mark.parametrize("outlier_scale, outlier_range, outlier_prob", [(0.0, 0, 0.0), (4.0, 4, 0.01)])
def test_incoherence_matmul(torch_dtype, size, odim, walsh, randomize, float32, outlier_scale, outlier_range, outlier_prob):
    """Check effects of transforms."""
    M.base_incoherence_matmul(torch_dtype, size, odim, walsh, randomize, float32, outlier_scale, outlier_range, outlier_prob)
