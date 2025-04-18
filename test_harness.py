#!/usr/bin/env python
# tensorcast/test_harness.py: test entry point external to tcast package
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import torch

import tcast
import tests.base_attention as A
import tests.base_kernel as K
import tests.base_misc as MISC
import tests.base_v2 as V2
import tests.utils as U
import tests.base_mx as MX
import tests.base_sqt as SQT

TEST_DICT = {"mx": MX, "sqt": SQT, "v2": V2}

def run_attention_configuration():
    num_methods = len(A.ALL_CONFIGS)
    for i in range(0, num_methods - 1):
        for j in range(i + 1, num_methods):
            A.base_attention_configuration(i, j)


def run_base_misc_imatrix_matmul() -> int:
    tcount = 0
    for torch_dtype in U.TORCH_DTYPES_32_16:
        for size in (8, 16, 32, 64, 128):
            for walsh in (False, True):
                for randomize in (False, True):
                    for outliers in ((None, 0, 0.0), (4.0, 4, 0.01)):
                        ref, icp = MISC.base_incoherence_matmul(torch_dtype, size, walsh, randomize, *outliers)
                        tcount += 1
    return tcount


def run_base_misc_imatrix() -> int:
    tcount = 0
    for torch_dtype in U.TORCH_DTYPES_32_16:
        for size in (8, 16, 32, 64, 128):
            for walsh in (False, True):
                for randomize in (False, True):
                    MISC.base_incoherence_matrix(torch_dtype, size, walsh, randomize)
                    tcount += 1
    return tcount


def run_base_v2_unscaled() -> int:
    tcount = 0
    for dtype in tcast.DataType.gather_registered(lambda x: x.is_unscaled and x.nspec.bits <= 16):
        for torch_dtype in U.TORCH_DTYPES_32_16:
            for shape in U.SHAPES_234D:
                for roundmode in U.ROUNDMODE_EA:
                    for computemode in ["torch"]:
                        V2.base_torchcast_virtual_v2(dtype, torch_dtype, shape, roundmode, computemode)
                        tcount += 1
    return tcount


def run_base_v2_tensor() -> int:
    tcount = 0
    for dtype in tcast.DataType.gather_registered(lambda x: x.is_tensor and x.nspec.bits <= 8):
        for torch_dtype in U.TORCH_DTYPES_32_16:
            for shape in U.SHAPES_234D:
                for roundmode in U.ROUNDMODE_E:
                    V2.base_torchcast_virtual_v2(dtype, torch_dtype, shape, roundmode)
                    tcount += 1
    return tcount


def run_base_v2_channel() -> int:
    tcount = 0
    for dtype in tcast.DataType.gather_registered(lambda x: x.is_channel and x.nspec.bits <= 8):
        for torch_dtype in U.TORCH_DTYPES_32_16:
            for shape in U.SHAPES_234D:
                for roundmode in U.ROUNDMODE_E:
                    for transpose in (False, True):
                        V2.base_torchcast_virtual_v2(dtype, torch_dtype, shape, roundmode, transpose=transpose)
                        tcount += 1
    return tcount


def run_base_v2_mxfp() -> int:
    tcount = 0
    for dtype in tcast.DataType.gather_registered(lambda x: x.is_mxfp):
        for torch_dtype in U.TORCH_DTYPES_32_16:
            for shape in U.SHAPES_234D:
                for roundmode in U.ROUNDMODE_E:
                    V2.base_torchcast_virtual_v2(dtype, torch_dtype, shape, roundmode)
                    tcount += 1
    return tcount


def run_base_v2_unscaled_torch():
    for dtype in tcast.DataType.gather_registered(lambda x: x.nspec.torch_dtype is not None and x.is_unscaled):
        for torch_dtype in U.TORCH_DTYPES_32_16:
            for shape in U.SHAPES_234D:
                V2.base_torchcast_virtual_v2(dtype, torch_dtype, shape)

def run_torchcast_unscaled():
    import tests.test_torchcast as TORCAST
    for compare in ["v2", "sqt", "mx"]:
        for dtype in tcast.DataType.gather_registered(lambda x: x.nspec.torch_dtype is not None and x.is_unscaled):
            if TEST_DICT[compare].supported(dtype):
                for torch_dtype in (torch.float32, torch.float16):
                    for shape in U.SHAPES_2D:
                        for roundmode in U.ROUNDMODE_E:
                            print(f"Running {compare} {dtype} {torch_dtype} {shape} {roundmode}")
                            TORCAST._torchcast_generic(compare, dtype, torch_dtype, shape, roundmode)
            else:
                print(f"Skipping: {compare} does not support {dtype}")


def run_kernel_apply_incoherence():
    for torch_dtype in (torch.float32, torch.float16):  # U.TORCH_DTYPES_32_16:
        for outer in (32, 64, 128):
            for inner in (8, 16, 32, 64):
                x = torch.randn((outer, inner), dtype=torch_dtype, device="cuda")
                xT = x.t().contiguous()
                ox = tcast.make_outliers(x, 0.0, 4, 0.02)
                oxT = ox.T.contiguous()
                for trans in (False, True):
                    for outliers in (False, True):
                        use_x = oxT if trans and outliers else xT if trans else ox if outliers else x
                        for use_fp32 in (False, True):
                            if torch_dtype != torch.float32 or not use_fp32:
                                K.base_kernel_apply_incoherence(use_x, trans, use_fp32, outliers)

def run_kernel_incoherence_gemm():
    for torch_dtype in (torch.float32, torch.float16):  # U.TORCH_DTYPES_32_16:
        for Z in (1, 8):
            for M in (32, 64, 128):
                for N in (32, 64):
                    for IM in (8, 16, 32, 64):
                        q = torch.randn((Z, M, IM), dtype=torch_dtype, device="cuda")
                        k = torch.randn((Z, N, IM), dtype=torch_dtype, device="cuda")
                        oq = tcast.make_outliers(q, 0.0, 4, 0.02)
                        for outliers in (False, True):
                            use_q = oq if outliers else q
                            for use_fp32 in (False, True):
                                if torch_dtype != torch.float32 or not use_fp32:
                                    K.base_kernel_incoherence_gemm(use_q, k, use_fp32, outliers)


def run_kernel_generic_gemm():
    shapes = []
    for Z in (1, 8):
        for M in (32, 64, 128):
            for N in (32, 64):
                for IM in (16, 32, 64):
                    shapes.append((Z, M, N, IM))
    quants = (
        (None, None),
        (None, "e4m3fn"),
        ("e4m3fnuz", None),
        ("e4m3fn", "e4m3fn"),
        ("e4m3fnuz", "e5m2fnuz"),
        ("e5m2", "e5m2"),
    )
    for torch_dtype in (torch.float32, torch.float16):
        for scale_dtype in (None, "float32", "float16"):
            for q_dtype, k_dtype in quants:
                for icp_qk, icp_fp32 in ((False, False), (True, False), (True, True)):
                    for randomize in (False, True):
                        for shape in shapes:
                            Z, M, N, IM = shape
                            q = torch.randn((Z, M, IM), dtype=torch_dtype, device="cuda")
                            k = torch.randn((Z, N, IM), dtype=torch_dtype, device="cuda")
                            oq = tcast.make_outliers(q, 0.0, 4, 0.02)
                            for outliers in (False, True):
                                use_q = oq if outliers else q
                                if torch_dtype != torch.float32 or not icp_fp32:
                                    K.base_kernel_generic_gemm(
                                        use_q, q_dtype, k, k_dtype, scale_dtype, icp_qk, icp_fp32,
                                        "even", "actual", randomize, outliers
                                    )


def run_quant_and_gemm_error():
    import tcastv2

    x = torch.randn(1, 32).cuda()
    y = torch.randn(1, 32).cuda()
    xy = x @ y.T

    def pp(n, d, s, am, ax, rm, rx):
        print(f"{n:4s} {str(d):18s} ({s}): amean {am:9.6f} amax {ax:9.6f} rmean {rm*100:10.4f}% rmax {rx*100:10.3f}%")

    tcount = 0
    for dtype in (tcastv2.mxfp8e4, tcastv2.mxint8, tcastv2.mxfp8e5, tcastv2.mxfp4e2, tcastv2.mxint4):
        qx0, qx1 = tcastv2.cast(x, dtype=dtype, scalemode="midmax").tensor, tcastv2.cast(x, dtype=dtype, scalemode="max").tensor
        qx0_a, qx1_a = (qx0 - x).abs(), (qx1 - x).abs()
        qx0_r, qx1_r = qx0_a / x.abs(), qx1_a / x.abs()
        pp("x", dtype, "mid", qx0_a.mean(), qx0_a.max(), qx0_r.mean(), qx0_r.max())
        pp("x", dtype, "max", qx1_a.mean(), qx1_a.max(), qx1_r.mean(), qx1_r.max())
        qy0, qy1 = tcastv2.cast(y, dtype=dtype, scalemode="midmax").tensor, tcastv2.cast(y, dtype=dtype, scalemode="max").tensor
        qy0_a, qy1_a = (qy0 - y).abs(), (qy1 - y).abs()
        qy0_r, qy1_r = qy0_a / y.abs(), qy1_a / y.abs()
        pp("y", dtype, "mid", qy0_a.mean(), qy0_a.max(), qy0_r.mean(), qy0_r.max())
        pp("y", dtype, "max", qy1_a.mean(), qy1_a.max(), qy1_r.mean(), qy1_r.max())
        qxy0, qxy1 = qx0 @ qy0.T, qx1 @ qy1.T
        qxy0_a, qxy1_a = (qxy0 - xy).abs(), (qxy1 - xy).abs()
        qxy0_r, qxy1_r = qxy0_a / xy.abs(), qxy1_a / xy.abs()
        pp("x@y", dtype, "mid", qxy0_a.mean(), qxy0_a.max(), qxy0_r.mean(), qxy0_r.max())
        pp("x@y", dtype, "max", qxy1_a.mean(), qxy1_a.max(), qxy1_r.mean(), qxy1_r.max())
        tcount += 1
    return tcount


if __name__ == "__main__":
    run_torchcast_unscaled()
    # run_attention_configuration()
    # run_kernel_apply_incoherence()
    # run_quant_and_gemm_error()
    # run_kernel_generic_gemm()
    # run_kernel_incoherence_gemm()
    # tcount_unscaled = run_base_v2_unscaled()
    # tcount_torch = run_base_v2_unscaled_torch()
    # tcount_tensor = run_base_v2_tensor()
    # tcount_channel = run_base_v2_channel()
    # tcount_config = run_base_misc_config()
    # tcount_imatrix = run_base_misc_imatrix()
    # tcount_icp_apply = run_snippets_icp_apply()
    # tcount_icp_gemm = run_snippets_icp_gemm()
    # tcount_generic_gemm = run_snippets_generic_gemm()
