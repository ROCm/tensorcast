#!/usr/bin/env python
# tests/base_attention.py: implementation of test code for attention interface triton kernels
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import os
from pathlib import Path

import torch
import triton
import triton.language as tl

import tcast
from tcast import attention as A

# tests for attention interface kernels (that use Attention and ACODE)

JSON_PATH = Path(__file__).parent.parent / "tcast/tests/config_attrs.json"
ENV_PATH = Path(__file__).parent.parent / "tcast/tests/config_attrs.sh"

@triton.jit
def generic_gemm_kernel(
    Q,
    K,
    Out,
    IM,
    stride_qz,
    stride_qm,
    stride_qi,
    stride_kz,
    stride_kn,
    stride_ki,
    stride_oz,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_I: tl.constexpr,
    LPCODE: tl.constexpr,
):
    """Kernel to perform an incoherence test with matrix multiply."""
    # k is loaded transposed
    off_z = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_i = tl.arange(0, BLOCK_I)
    qptr = tl.load(Q + off_z * stride_qz + offs_m[:, None] * stride_qm + offs_i[None, :] * stride_qi)
    kptr = tl.load(K + off_z * stride_kz + offs_i[:, None] * stride_ki + offs_n[None, :] * stride_kn)
    imptr = tl.load(IM + offs_i[:, None] * BLOCK_I + offs_i[None, :])
    qq, qs = A.scale_and_quantize(qptr, imptr, LPCODE, A.Q_INDEX)
    kq, ks = A.scale_and_quantize(kptr, imptr, LPCODE, A.K_INDEX, trans=True)
    # if only one is quantized, that one needs to be upcast to the other one's type
    quantq = A.needs_quant(LPCODE, A.Q_INDEX)
    quantk = A.needs_quant(LPCODE, A.K_INDEX)
    if quantq and not quantk:
        qq = qq.to(kq.type.element_ty)
    elif quantk and not quantq:
        kq = kq.to(qq.type.element_ty)
    out = (tl.dot(qq, kq) / (qs * ks)).to(Out.type.element_ty)
    tl.store(Out + off_z * stride_oz + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, out)


def _experiment_name(attention: tcast.Attention, M, N, IM, randomize: bool = False, outliers: bool = False):
    """Generate a string for the experiment name."""
    expstr = attention.short_repr("qk")
    if not expstr:
        expstr = "REFERENCE"
    added_cfg = "OUTLIERS" if outliers else ""
    if attention.need_imatrix and randomize:
        if added_cfg:
            added_cfg += ", "
    if added_cfg:
        added_cfg = f" ({added_cfg})"
    expstr += added_cfg
    expstr = f"[{M:3d},{N:3d},{IM:3d}] {expstr}"
    return expstr


def base_attention_generic_gemm(
    q: torch.Tensor, k: torch.Tensor, attention: tcast.Attention, randomize: bool = False, outliers: bool = False
):
    """Base incoherence test without quantization using triton snippet."""
    # this test uses kernels that use LPCODE, so we need a configuration
    imatrix = tcast.ICP.get_imatrix(q.size(-1), torch.float32, q.device, randomize=randomize)
    # input is q, k, which are 3D tensors with shape (Z, M, IM) and (Z, N, IM) respectively,
    # so q @ k.T is (Z, M, N)
    # the grid is dimension 0
    BLOCK_I = imatrix.size(0)
    BLOCK_M = q.size(1)
    BLOCK_N = k.size(1)
    expstr = _experiment_name(attention, BLOCK_M, BLOCK_N, BLOCK_I, randomize=randomize, outliers=outliers)

    assert q.dim() == k.dim() == 3 and q.size(2) == k.size(2) == BLOCK_I and q.size(0) == k.size(0)
    assert BLOCK_M >= BLOCK_I and BLOCK_M % BLOCK_I == 0 or BLOCK_M < BLOCK_I and BLOCK_I % BLOCK_M == 0
    assert BLOCK_N >= BLOCK_I and BLOCK_N % BLOCK_I == 0 or BLOCK_N < BLOCK_I and BLOCK_I % BLOCK_N == 0

    out = torch.empty((q.size(0), BLOCK_M, BLOCK_N), dtype=q.dtype, device=q.device)
    out_ref = torch.bmm(q, k.transpose(1, 2))
    # incoherence processing and/or quantization (WAS lp_code())
    generic_gemm_kernel[(q.size(0),)](
        q, k, out, imatrix, *q.stride(), *k.stride(), *out.stride(), BLOCK_M, BLOCK_N, BLOCK_I, attention.code
    )
    absolute_error = (out - out_ref).abs()
    relative_error = absolute_error / (out_ref.abs() + 1e-8)
    maerr = absolute_error.mean()
    mrerr = relative_error.mean()
    xaerr = absolute_error.max()
    xrerr = relative_error.max()
    absstr = f"{maerr:12.9f}, {xaerr:15.6f}"
    relstr = f"{mrerr:12.9f}, {xrerr:15.6f}"
    tcast.get_logger().info(f"absolute: {absstr} relative: {relstr} # {expstr}")
    # TODO(ericd): figure out ATOL and RTOL based on the quantization datatypes
    # torch.testing.assert_close(out, out_ref)


def _method_param_code():
    return tcast.Attention(code=0x10a929201c) # 0x10a9292014


def _method_param_json():
    return tcast.Attention(json_path=JSON_PATH)


def _method_param_shortcut():
    return tcast.Attention(shortcut="split_match_e4m3fnuz_e5m2fnuz_icpqk32_32x32_FEV")


def _method_param_attrs():
    return tcast.Attention(
        block_size=(32, 32),
        scale_dtype=None,
        q_dtype="e4m3fnuz",
        k_dtype="e4m3fnuz",
        v_dtype="e4m3fnuz",
        p_dtype="e5m2fnuz",
        ds_dtype="e5m2fnuz",
        do_dtype="e5m2fnuz",
        icp_qk=True,
        icp_pv=False,
        icp_fp32=True,
    )


def _method_env_code():
    os.environ["TC_CODE"] = "0x10a929201c"
    return tcast.Attention()


def _method_env_json():
    os.environ["TC_JSON_PATH"] = str(JSON_PATH)
    return tcast.Attention()


def _method_env_shortcut():
    os.environ["TC_SHORTCUT"] = "split_match_e4m3fnuz_e5m2fnuz_icpqk32_32x32_FEV"
    return tcast.Attention()


def _method_env_attrs():
    import subprocess

    command = f"source {str(ENV_PATH)} && env"
    process = subprocess.Popen(command, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        tcast.get_logger().error(f"Error executing script: {stderr.decode()}")
        return None
    env_vars = {}
    for line in stdout.decode().splitlines():
        if line.startswith("TC_") and "=" in line:
            key, value = line.split("=", 1)
            env_vars[key] = value
    for key, value in env_vars.items():
        os.environ[key] = value
    return tcast.Attention()


ALL_METHODS: list = [
    _method_param_code,
    _method_param_json,
    _method_param_shortcut,
    _method_param_attrs,
    _method_env_code,
    _method_env_json,
    _method_env_shortcut,
    _method_env_attrs,
]

ALL_CONFIGS: list = []

for method in ALL_METHODS:
    for key in os.environ.copy():
        if key.startswith("TC_"):
            os.environ.pop(key)
    config = method()
    ALL_CONFIGS.append(config)
    tcast.get_logger().debug(f"Attention {method.__name__}: {hex(config.code)}")


def base_attention_configuration(index1: int, index2: int):
    """Test the Attention class for a pair of methods for configuring the class."""
    logger = tcast.get_logger()
    if index1 == index2 or index1 < 0 or index2 < 0 or index1 >= len(ALL_CONFIGS) or index2 >= len(ALL_CONFIGS):
        logger.error(f"Invalid indices: {index1}, {index2}")
        return False
    # sort the indices
    index1, index2 = min(index1, index2), max(index1, index2)
    cfg1, cfg2 = ALL_CONFIGS[index1], ALL_CONFIGS[index2]
    matches = cfg1 == cfg2
    logger.info(f"code {index1} {hex(cfg1.code)} code {index2} {hex(cfg2.code)} {'PASS' if matches else 'FAIL'}")
    if not matches:
        for key, value in cfg1.__dict__.items():
            if not hasattr(cfg2, key):
                logger.info(f"KEY missing in cfg2: {key}: {value}")
            elif getattr(cfg2, key) != value:
                logger.info(f"KEY mismatch: {key}: {value} != {getattr(cfg2, key)}")
        for key, value in cfg2.__dict__.items():
            if not hasattr(cfg1, key):
                logger.info(f"KEY missing in cfg1: {key}: {value}")
        raise AssertionError(f"Attention class configuration failed: {index1} {index2}")
