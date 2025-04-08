#!/usr/bin/env python
# tests/base_kernel.py: implementation of test code for triton kernels other than attention interface
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import torch
import triton
import triton.language as tl

import tcast
from tcast import kernels as T

EL = tcast.get_extern_libs()


def _error_stats(x, q, kx=None, kq=None, teststr=""):
    """Compute and log error statistics."""
    logger = tcast.get_logger()
    absolute_error = (q - x).abs()
    relative_error = absolute_error / (x.abs() + 1e-8)
    maerr = absolute_error.mean()
    mrerr = relative_error.mean()
    xaerr = absolute_error.max()
    xrerr = relative_error.max()
    kstr = f"kurtosis: {kx:10.4f}, {kq:10.4f} " if kx is not None else ""
    absstr = f"{maerr:11.9f}, {xaerr:11.8f}"
    relstr = f"{mrerr:11.7f}, {xrerr:11.7f}"
    teststr = f" # {teststr}" if teststr else ""
    logger.info(f"{kstr}absolute: {absstr} relative: {relstr}{teststr}")


# fmt: off
@triton.jit
def _generic_gemm_kernel(
    Q, K, Out, IM, stride_qz, stride_qm, stride_qi, stride_kz, stride_kn, stride_ki, stride_oz, stride_om, stride_on,
    qtype, qemax, qemin, qmbits, qmaxfloat, ktype, kemax, kemin, kmbits, kmaxfloat, stype,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_I: tl.constexpr, QUANTQ: tl.constexpr, QUANTK: tl.constexpr,
    ICP_QK: tl.constexpr, ICP_FP32: tl.constexpr, IS_EXP: tl.constexpr, RMODE: tl.constexpr, CMODE: tl.constexpr,
):
    """Kernel to perform an incoherence test with matrix multiply."""
    # k is loaded transposed
    off_z = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_i = tl.arange(0, BLOCK_I)
    qptr = tl.load(Q + off_z * stride_qz + offs_m[:, None] * stride_qm + offs_i[None, :] * stride_qi)
    kptr = tl.load(K + off_z * stride_kz + offs_i[:, None] * stride_ki + offs_n[None, :] * stride_kn)
    if QUANTQ or QUANTK:
        if stype:
            stype = T.get_triton_type(stype)
        else:
            stype = tl.float32
    if ICP_QK:
        imptr = tl.load(IM + offs_i[:, None] * BLOCK_I + offs_i[None, :])
        qptr = T.apply_incoherence(qptr, imptr, False, ICP_FP32)
        kptr = T.apply_incoherence(kptr, imptr, True, ICP_FP32)
    descale = 1.0
    if QUANTQ:
        qs = T.get_scale(qptr, IS_EXP, qemax, maxfloat=qmaxfloat)
        descale *= qs
        qq = T.quantize_float(qptr, qs, qemax, qemin, qmbits, qmaxfloat, RMODE, CMODE, IS_EXP, qtype)
        if IS_EXP and CMODE != T.CMODE_VIRTUAL:
            qs = T.bias_scale(qs, qemax, qemin)
        elif not IS_EXP:
            qs = qs.to(stype)
    else:
        qq = qptr
        qs = 1.0
    if QUANTK:
        ks = T.get_scale(kptr, IS_EXP, kemax, maxfloat=kmaxfloat)
        descale *= ks
        kq = T.quantize_float(kptr, ks, kemax, kemin, kmbits, kmaxfloat, RMODE, CMODE, IS_EXP, ktype)
        if IS_EXP and CMODE != T.CMODE_VIRTUAL:
            ks = T.bias_scale(ks, kemax, kemin)
        elif not IS_EXP:
            ks = ks.to(stype)
    else:
        kq = kptr
        ks = 1.0
    out = (tl.dot(qq, kq) / descale).to(Out.type.element_ty)
    tl.store(Out + off_z * stride_oz + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, out)
# fmt: on


def _experiment_name(qdtype, kdtype, sdtype, icp_qk, icp_fp32, M, N, IM, randomize: bool = False, outliers: bool = False):
    """Generate a string for the experiment name."""
    if not qdtype and not kdtype and not icp_qk:
        expstr = "REFERENCE"
    else:
        expstr = ""
        for name, dtype in {"Q": qdtype, "K": kdtype, "S": sdtype}.items():
            tname = None if dtype is None or dtype.nspec.torch_dtype is None else dtype.nspec.torch_dtype
            if name != "S" or dtype is not None:
                tname = tcast.torch_dtype_to_triton(tname)
                if expstr:
                    expstr += ","
                expstr += f"{name}:{tname:8s}"
        if icp_qk:
            expstr += " ICP"
            if icp_fp32:
                expstr += "32"
            if randomize:
                expstr += "(R)"
        if outliers:
            expstr += " (OUTLIERS)"
    expstr = f"[{M:3d},{N:3d},{IM:3d}] {expstr}"
    return expstr


def _get_datatype_info(dtype: tcast.DataType):
    if dtype:
        n = dtype.nspec
        name = tcast.torch_dtype_to_triton(dtype.nspec.torch_dtype)
        if name is None:
            raise AssertionError(f"Unsupported dtype in base_kernel_generic_gemm {dtype}")
        return name, n.emax, n.emin, n.mbits, n.finfo.maxfloat
    return "", 0, 0, 0, 0.0


# fmt: off
def base_kernel_generic_gemm(
    q: torch.Tensor,
    qdtype: tcast.DataType,
    k: torch.Tensor,
    kdtype: tcast.DataType,
    sdtype: tcast.DataType,
    icp_qk: bool,
    icp_fp32: bool,
    roundmode,
    castmode,
    randomize: bool = False,
    outliers: bool = False,
):
    """Base gemm test using kernels instead of attention interfaces."""
    imatrix = tcast.ICP.get_imatrix(q.size(-1), torch.float32, q.device, randomize=randomize)
    # input is q, k, which are 3D tensors with shape (Z, M, IM) and (Z, N, IM) respectively,
    # so q @ k.T is (Z, M, N)
    # the grid is dimension 0
    BLOCK_I = imatrix.size(0)
    BLOCK_M = q.size(1)
    BLOCK_N = k.size(1)

    assert q.dim() == k.dim() == 3 and q.size(2) == k.size(2) == BLOCK_I and q.size(0) == k.size(0)
    assert BLOCK_M >= BLOCK_I and BLOCK_M % BLOCK_I == 0 or BLOCK_M < BLOCK_I and BLOCK_I % BLOCK_M == 0
    assert BLOCK_N >= BLOCK_I and BLOCK_N % BLOCK_I == 0 or BLOCK_N < BLOCK_I and BLOCK_I % BLOCK_N == 0

    out = torch.empty((q.size(0), BLOCK_M, BLOCK_N), dtype=q.dtype, device=q.device)
    out_ref = torch.bmm(q, k.transpose(1, 2))
    # incoherence processing and/or quantization (WAS lp_code())
    qname, qemax, qemin, qmbits, qmaxfloat = _get_datatype_info(qdtype)
    kname, kemax, kemin, kmbits, kmaxfloat = _get_datatype_info(kdtype)
    expstr = _experiment_name(
        qdtype, kdtype, sdtype, icp_qk, icp_fp32, BLOCK_M, BLOCK_N, BLOCK_I, randomize=randomize, outliers=outliers
    )
    storch = None if sdtype is None or sdtype.nspec.torch_dtype is None else sdtype.nspec.torch_dtype
    sname = tcast.torch_dtype_to_triton(storch)
    _generic_gemm_kernel[(q.size(0),)](
        q, k, out, imatrix, *q.stride(), *k.stride(), *out.stride(), qname, qemax, qemin, qmbits, qmaxfloat,
        kname, kemax, kemin, kmbits, kmaxfloat, sname, BLOCK_M, BLOCK_N, BLOCK_I,
        qmaxfloat > 0.0, kmaxfloat > 0.0, icp_qk, icp_fp32, sname == "u8", roundmode, castmode,
    )
    _error_stats(out_ref, out, None, None, expstr)
    # TODO(ericd): figure out ATOL and RTOL based on the quantization datatypes
    # torch.testing.assert_close(out, out_ref)
# fmt: on


# fmt: off
def base_kernel_incoherence_gemm(q: torch.Tensor, k: torch.Tensor, icp_fp32: bool, has_outliers: bool):
    """Base incoherence test without quantization using triton kernels."""
    imatrix = tcast.ICP.get_imatrix(q.size(-1), torch.float32, q.device, randomize=False)
    # input is q, k, which are 3D tensors with shape (Z, M, IM) and (Z, N, IM) respectively,
    # so q @ k.T is (Z, M, N)
    # the grid is dimension 0
    assert q.dim() == k.dim() == 3
    BLOCK_I, BLOCK_M, BLOCK_N = imatrix.size(0), q.size(1), k.size(1)
    assert q.size(2) == BLOCK_I and k.size(2) == BLOCK_I and q.size(0) == k.size(0)
    assert BLOCK_M >= BLOCK_I and BLOCK_M % BLOCK_I == 0 or BLOCK_M < BLOCK_I and BLOCK_I % BLOCK_M == 0
    assert BLOCK_N >= BLOCK_I and BLOCK_N % BLOCK_I == 0 or BLOCK_N < BLOCK_I and BLOCK_I % BLOCK_N == 0
    out_im = torch.zeros((q.size(0), BLOCK_M, BLOCK_N), dtype=q.dtype, device=q.device)
    out_non_im = torch.zeros((q.size(0), BLOCK_M, BLOCK_N), dtype=q.dtype, device=q.device)
    out_ref = torch.bmm(q, k.transpose(1, 2))
    expname = _experiment_name(None, None, None, True, icp_fp32, BLOCK_M, BLOCK_N, BLOCK_I, False, has_outliers)
    # incoherence processing only (no quantization)
    _generic_gemm_kernel[(q.size(0),)](
        q, k, out_im, imatrix, *q.stride(), *k.stride(), *out_im.stride(),
        "", 0, 0, 0, 0.0, "", 0, 0, 0, 0.0, "fp32", BLOCK_M, BLOCK_N, BLOCK_I,
        False, False, True, icp_fp32, False, 2, 0,
    )
    # neither incoherence nor quantization, should match torch reference
    _generic_gemm_kernel[(q.size(0),)](
        q, k, out_non_im, imatrix, *q.stride(), *k.stride(), *out_non_im.stride(),
        "", 0, 0, 0, 0.0, "", 0, 0, 0, 0.0, "fp32", BLOCK_M, BLOCK_N, BLOCK_I,
        False, False, True, icp_fp32, False, 2, 0,
    )
    _error_stats(out_ref, out_im, None, None, f"INCOHERENCE: {expname}")
    torch.testing.assert_close(out_non_im, out_ref)
# fmt: on


@triton.jit
def _apply_incoherence_kernel(
    X, Out, IM, BLOCK_I: tl.constexpr, BLOCK_OTHER: tl.constexpr, x_st_0, x_st_1, o_st_0, o_st_1, trans: bool, use_fp32: bool
):
    """Kernel to apply an incoherence test."""
    offs_other = tl.arange(0, BLOCK_OTHER)
    offs_im = tl.arange(0, BLOCK_I)
    if trans:
        xptr = tl.load(X + offs_im[:, None] * x_st_0 + offs_other[None, :] * x_st_1)
        o_offsets = Out + offs_im[:, None] * o_st_0 + offs_other[None, :] * o_st_1
    else:
        xptr = tl.load(X + offs_other[:, None] * x_st_0 + offs_im[None, :] * x_st_1)
        o_offsets = Out + offs_other[:, None] * o_st_0 + offs_im[None, :] * o_st_1
    imptr = tl.load(IM + offs_im[:, None] * BLOCK_I + offs_im[None, :])
    out = T.apply_incoherence(xptr, imptr, trans, use_fp32)
    tl.store(o_offsets, out)


def base_kernel_apply_incoherence(x: torch.Tensor, trans: bool, use_fp32: bool, has_outliers: bool):
    """Base incoherence test using triton generic kernel.  Information only."""
    logger = tcast.get_logger()
    # imatrix will never be big, so we can use float32
    imatrix = tcast.ICP.get_imatrix(x.size(-1 - int(trans)), torch.float32, x.device, randomize=False)
    BLOCK_I = imatrix.size(0)
    assert x.is_contiguous() and x.dim() == 2 and x.size(1 - int(trans)) == BLOCK_I
    BLOCK_OTHER = x.size(int(trans))
    xicp = torch.empty_like(x)
    _apply_incoherence_kernel[(1,)](x, xicp, imatrix, BLOCK_I, BLOCK_OTHER, *x.stride(), *xicp.stride(), trans, use_fp32)
    optstr = f"{(' ', 'T')[trans]} {('    ', 'FP32')[use_fp32]} {('   ', 'OUT')[has_outliers]}"
    teststr = f"[{x.shape[0]:3d},{x.shape[1]:3d}] {str(x.dtype)[6:]:8s} {optstr}"
    # see how the outliers affect the kurtosis
    k, kicp = tcast.kurtosis(x.float()), tcast.kurtosis(xicp.float())
    if kicp in (float("nan"), float("inf"), float("-inf")):
        logger.warning(f"NaN/inf in processed xicp {teststr}")
    else:
        x_gemm = x.T @ x if trans else x @ x.T
        xicp_gemm = xicp.T @ xicp if trans else xicp @ xicp.T
        if not xicp_gemm.isfinite().all():
            logger.warning(f"NaN/inf in output xicp_gemm {teststr}")
        else:
            _error_stats(x_gemm, xicp_gemm, k, kicp, teststr)
