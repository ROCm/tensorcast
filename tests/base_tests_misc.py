#!/usr/bin/env python
# tests/base_tests_misc.py.py: miscellaneous tests
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import torch

import tcast

logger = tcast.get_logger("tcast_unittest")


def test_incoherence_matrix_base(torch_dtype, size, walsh, randomize):
    """Test if a matrix is a Hadamard or Walsh-Hadamard matrix."""
    imatrix = tcast.LPConfig.get_imatrix(size, torch_dtype, walsh=walsh, randomize=randomize)
    if walsh:
        tcast.LPConfig.check_walsh(imatrix)
    else:
        tcast.LPConfig.check_hadamard(imatrix)


def test_incoherence_matmul_base(torch_dtype, size, odim, walsh, randomize, float32, outlier_scale, outlier_range, outlier_prob):
    """Check effects of transforms."""
    AB_ref, AB_icp = tcast.LPConfig.test_icp(
        logger, torch_dtype, size, odim, walsh, randomize, float32, outlier_scale, outlier_range, outlier_prob
    )
    # TODO(ericd): dial in the tolerances
    torch.testing.assert_close(AB_ref, AB_icp)


def test_lpconfig_base(logger, index1: int, index2: int) -> bool:
    """Check configuration torch interfaces."""
    return tcast.LPConfig.check_config(logger, index1, index2)
