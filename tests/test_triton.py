#!/usr/bin/env python
# tests/test_triton.py: PyTest for triton generic kernels
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import pytest
import torch

import tcast
import tests.base_kernel as K
import tests.utils as U


@pytest.mark.parametrize("size", U.IMSIZES)
@pytest.mark.parametrize("qdtype", U.TORCH_DTYPES_8 + [None])
@pytest.mark.parametrize("kdtype", U.TORCH_DTYPES_8 + [None])
@pytest.mark.parametrize("sdtype", U.TORCH_DTYPES_32_16 + [None])
@pytest.mark.parametrize("torch_dtype", U.TORCH_DTYPES_32_16)
@pytest.mark.parametrize("Z", [1, 8])
@pytest.mark.parametrize("M", [16, 32, 64])
@pytest.mark.parametrize("N", [16, 32, 64])
@pytest.mark.parametrize("outlier_scale, outlier_range, outlier_prob", [(0.0, 0, 0.0), (4.0, 4, 0.01)])
@pytest.mark.parametrize("icp_qk, icp_fp32", [(False, False), (True, False), (True, True)])
def test_triton_generic_gemm(
    size, qdtype, kdtype, sdtype, torch_dtype, Z, M, N, outlier_scale, outlier_range, outlier_prob, icp_qk, icp_fp32
):
    if torch_dtype == torch.float32 and icp_fp32:
        pytest.skip("redundant test with float32 tensor and icp_fp32, skipping test.")
    if qdtype is None and kdtype is None:
        pytest.skip("skipping test with no quantization")
    has_outliers = outlier_prob > 0.0
    q = torch.randn((Z, M, size), dtype=torch_dtype)
    k = torch.randn((Z, N, size), dtype=torch_dtype)
    if has_outliers:
        q = tcast.make_outliers(q, outlier_scale, outlier_range, outlier_prob)
    K.base_kernel_generic_gemm(q, qdtype, k, kdtype, sdtype, icp_qk, icp_fp32, 2, 0, False, has_outliers)


@pytest.mark.parametrize("size", U.IMSIZES)
@pytest.mark.parametrize("torch_dtype", U.TORCH_DTYPES_32_16)
@pytest.mark.parametrize("outlier_scale, outlier_range, outlier_prob", [(0.0, 0, 0.0), (4.0, 4, 0.01)])
@pytest.mark.parametrize("trans", [False, True])
@pytest.mark.parametrize("icp_fp32", [False, True])
def test_triton_apply_incoherence(size, torch_dtype, outlier_scale, outlier_range, outlier_prob, trans, icp_fp32):
    """Test triton apply_incoherence."""
    if torch_dtype == torch.float32 and icp_fp32:
        pytest.skip("redundant test with float32 tensor and icp_fp32, skipping test.")
    x = torch.randn(1024, size, dtype=torch_dtype)
    has_outliers = outlier_prob > 0.0
    if has_outliers:
        x = tcast.make_outliers(x, outlier_scale, outlier_range, outlier_prob)
    if trans:
        x = x.t().contiguous()
    K.base_kernel_apply_incoherence(x, trans, icp_fp32, has_outliers)


@pytest.mark.parametrize("size", U.IMSIZES)
@pytest.mark.parametrize("torch_dtype", U.TORCH_DTYPES_32_16)
@pytest.mark.parametrize("Z", [1, 8])
@pytest.mark.parametrize("M", [16, 32, 64])
@pytest.mark.parametrize("N", [16, 32, 64])
@pytest.mark.parametrize("outlier_scale, outlier_range, outlier_prob", [(0.0, 0, 0.0), (4.0, 4, 0.01)])
@pytest.mark.parametrize("icp_fp32", [False, True])
def test_triton_incoherence_gemm(size, torch_dtype, Z, N, M, IM, outlier_scale, outlier_range, outlier_prob, trans, icp_fp32):
    """Test triton incoherence gemm outputs."""
    if torch_dtype == torch.float32 and icp_fp32:
        pytest.skip("redundant test with float32 tensor and icp_fp32, skipping test.")
    q = torch.randn((Z, M, size), dtype=torch_dtype)
    k = torch.randn((Z, N, size), dtype=torch_dtype)
    has_outliers = outlier_prob > 0.0
    if has_outliers:
        q = tcast.make_outliers(q, outlier_scale, outlier_range, outlier_prob)
    K.base_kernel_incoherence_gemm(q, k, icp_fp32, has_outliers)
