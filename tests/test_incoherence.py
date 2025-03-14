#!/usr/bin/env python
# tests/test_incoherence.py: incoherence processing
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import pytest
import torch

import tcast

logger = tcast.get_logger("tcast_unittest")


@pytest.mark.parametrize("torch_dtype,", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("size", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("walsh, randomize", [(False, False), (True, False), (True, True)])
def test_incoherence_matrix(torch_dtype, size, walsh, randomize):
    """Test if a matrix is a Hadamard or Walsh-Hadamard matrix."""
    imatrix = tcast.LPConfig.get_imatrix(size, torch_dtype, walsh=walsh, randomize=randomize)
    if walsh:
        tcast.LPConfig.check_walsh(imatrix)
    else:
        tcast.LPConfig.check_hadamard(imatrix)


@pytest.mark.parametrize("torch_dtype,", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("size", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("odim", [1, 2, 4])
@pytest.mark.parametrize("walsh, randomize", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("float32", [False, True])
@pytest.mark.parametrize("outlier_scale, outlier_range, outlier_prob", [(None, 0, 0.0), (4.0, 4, 0.01)])
def test_incoherence_matmul(torch_dtype, size, odim, walsh, randomize, float32, outlier_scale, outlier_range, outlier_prob):
    """Check effects of transforms."""
    AB_ref, AB_icp = tcast.config.test_icp(
        logger, torch_dtype, size, odim, walsh, randomize, float32, outlier_scale, outlier_range, outlier_prob
    )
    torch.testing.assert_close(AB_ref, AB_icp)
