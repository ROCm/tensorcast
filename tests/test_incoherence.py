#!/usr/bin/env python
# tests/test_incoherence.py: incoherence processing
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import pytest
import torch

import tcast

logger = tcast.get_logger("tcast_unittest")


def _check_hadamard(matrix):
    """Check if a matrix is a Hadamard matrix."""
    # Ensure the matrix is scaled correctly, i.e. +- a single scaled value
    assert (matrix.abs() == matrix[0, 0].abs()).all(), "Matrix absolute values are not all the same"
    matrix = matrix.sign()
    # Check that all entries are either +1 or -1
    assert torch.all((matrix == 1) | (matrix == -1)), "Matrix contains elements other than +1 and -1"
    # Check orthogonality: H * H^T = n * I
    size = matrix.size(0)
    identity_matrix = size * torch.eye(size, dtype=matrix.dtype, device=matrix.device)
    torch.testing.assert_close(matrix @ matrix.t(), identity_matrix)


def _check_walsh(matrix):
    """Check if a matrix is a Walsh matrix."""
    _check_hadamard(matrix)
    size = matrix.size(0)
    changes = [sum(int(matrix[j, i] != matrix[j, i + 1]) for i in range(size - 1)) for j in range(size)]
    # check if the number of sign changes is in increasing order by row
    order = torch.tensor(changes, dtype=matrix.dtype, device=matrix.device).argsort()
    reorder = torch.argsort(order)
    torch.testing.assert_close(order, reorder)


@pytest.mark.parametrize("torch_dtype,", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("size", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("walsh, randomize", [(False, False), (True, False), (True, True)])
def test_incoherence_matrix(torch_dtype, size, walsh, randomize):
    """Test if a matrix is a Hadamard or Walsh-Hadamard matrix."""
    imatrix = tcast.get_imatrix(size, torch_dtype, walsh=walsh, randomize=randomize)
    if walsh:
        _check_walsh(imatrix)
    else:
        _check_hadamard(imatrix)


@pytest.mark.parametrize("torch_dtype,", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("size", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("odim", [1, 2, 4])
@pytest.mark.parametrize("walsh, randomize", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("float32", [False, True])
@pytest.mark.parametrize("outlier_scale, outlier_range, outlier_prob", [(None, 0, 0.0), (4.0, 4, 0.01)])
def test_incoherence_matmul(torch_dtype, size, odim, walsh, randomize, float32, outlier_scale, outlier_range, outlier_prob):
    """Check effects of transforms."""
    M: torch.Tensor = tcast.get_imatrix(size, torch_dtype, walsh=walsh, randomize=randomize)
    A = torch.randn(size * odim, size, dtype=torch_dtype, device=M.device)
    B = torch.randn(size * odim, size, dtype=torch_dtype, device=M.device)
    if outlier_prob > 0.0:
        A = tcast.make_outliers(A, scale=outlier_scale, range=outlier_range, prob=outlier_prob)
        B = tcast.make_outliers(B, scale=outlier_scale, range=outlier_range, prob=outlier_prob)
    if float32:
        M, A, B = M.float(), A.float(), B.float()
    A_kurtosis = tcast.kurtosis(A)
    B_kurtosis = tcast.kurtosis(B)
    AB_ref = (A @ B.t()).to(torch_dtype)
    AB_ref_norm = torch.norm(AB_ref).item()
    AM = (A @ M).to(torch_dtype)
    BM = (B @ M).to(torch_dtype)
    AM_kurtosis = tcast.kurtosis(AM)
    BM_kurtosis = tcast.kurtosis(BM)
    AB_icp = (AM @ BM.t()).to(torch_dtype)
    AB_icp_norm = torch.norm(AB_icp).item()
    diff_norm = torch.norm(AB_ref - AB_icp).item() / AB_ref_norm
    args = f"{size*odim}x{size} {str(torch_dtype)[6:]} "
    args += "RWH" if walsh and randomize else "WH" if walsh else "H"
    args += " F32MM" if float32 else ""
    if outlier_prob > 0.0:
        args += f" O{outlier_scale}R{outlier_range}P{outlier_prob}"
    logger.info(f"test_incoherence_matmul({args}):")
    logger.info(f"\tnorms:    diff_norm={diff_norm} ref_norm={AB_ref_norm} icp_norm={AB_icp_norm}")
    logger.info(f"\tkurtosis: A={A_kurtosis} AM={AM_kurtosis} B={B_kurtosis} BM={BM_kurtosis}")
    torch.testing.assert_close(AB_ref, AB_icp)
