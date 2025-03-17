#!/usr/bin/env python
# tests/test_misc.py: test LP configuration, incoherence, etc.
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import pytest
import torch

import tcast
import tests.base_tests_misc as B

logger = tcast.get_logger("tcast_unittest")


@pytest.mark.parametrize("torch_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("size", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("walsh, randomize", [(False, False), (True, False), (True, True)])
def test_incoherence_matrix(torch_dtype, size, walsh, randomize):
    """Test if a matrix is a Hadamard or Walsh-Hadamard matrix."""
    B.test_incoherence_matrix_base(torch_dtype, size, walsh, randomize)


@pytest.mark.parametrize("torch_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("size", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("odim", [1, 2, 4])
@pytest.mark.parametrize("walsh, randomize", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("float32", [False, True])
@pytest.mark.parametrize("outlier_scale, outlier_range, outlier_prob", [(None, 0, 0.0), (4.0, 4, 0.01)])
def test_incoherence_matmul(torch_dtype, size, odim, walsh, randomize, float32, outlier_scale, outlier_range, outlier_prob):
    """Check effects of transforms."""
    B.test_incoherence_matmul_base(torch_dtype, size, odim, walsh, randomize, float32, outlier_scale, outlier_range, outlier_prob)


# fmt: off
@pytest.mark.parametrize("index1, index2", [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
        (2, 3), (2, 4), (2, 5), (2, 6), (2, 7),
        (3, 4), (3, 5), (3, 6), (3, 7),
        (4, 5), (4, 6), (4, 7),
        (5, 6), (5, 7),
        (6, 7)
    ]
)
def test_lpconfig(index1, index2):
    match = B.test_lpconfig_base(index1, index2)
    if not match:
        raise AssertionError(f"Mismatch in LP configuration for indices {index1} and {index2}")
#fmt on
