#!/usr/bin/env python
# examples/low_precision_attention_example.py: launch and triton kernel for LP GEMM
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import argparse
from pathlib import Path

import torch
import triton
import triton.language as tl

import tcast
import tcast.snippets as lp


@triton.jit
def get_block_ptr(base, sz0, sz1, st0, st1, BLOCK_Q: tl.constexpr, off0, off1):
    return tl.make_block_ptr(
        base=base, shape=(sz0, sz1), strides=(st0, st1), offsets=(off0, off1), block_shape=(BLOCK_Q, BLOCK_Q), order=(1, 0),
    )

@triton.jit
def simple_qdot(acc, qptr, kptr, imatrix, LPCODE: tl.constexpr):
    """Simple dot product kernel, both quantized, no stochastic rounding, ICP optional."""
    qq, qs = lp.scale_and_quantize(qptr, imatrix, LPCODE, lp.Q_INDEX)
    kq, ks = lp.scale_and_quantize(kptr, imatrix, LPCODE, lp.K_INDEX, trans=True)
    acc += tl.dot(qq, tl.trans(kq)) / (qs * ks)
    return acc

@triton.jit
def example_kernel_qdot(acc, qptr, qqptr, qsptr, kptr, kqptr, ksptr, LPCODE: tl.constexpr, imatrix, seed, offset):
    """Inner kernel for QK dot product, called from example_kernel."""
    # q and k can be loaded at any time before the scale_and_quantize call.
    q = tl.load(qptr)
    k = tl.load(kptr)
    if not lp.enabled(LPCODE):
        # we must be baselining
        return acc + tl.dot(q, tl.trans(k))
    # if both q and k are quantized to fp8, we can use the simple method:
    #      return acc + simple_qdot(acc, qptr, kptr, imatrix, LPCODE)
    # otherwise, complexity begets complexity
    call_sq_for_q, quantq, is_icp = lp.needs_quant_or_icp(LPCODE, lp.Q_INDEX)
    quantk = lp.needs_quant(LPCODE, lp.K_INDEX)
    call_sq_for_k = quantk or is_icp
    if not (quantq or quantk):
        # if we are not quantizing either, just the regular dot product
        return acc + tl.dot(q, tl.trans(k))
    # ICP is for both q and k or neither; if one is unquantized but both need ICP, we need to quantize
    qq, qs, kq, ks = q, 1.0, k, 1.0
    if call_sq_for_q:
        qq, qs = lp.scale_and_quantize(qptr, imatrix, LPCODE, lp.Q_INDEX, seed, offset, trans=False)
    if call_sq_for_k:
        kq, ks = lp.scale_and_quantize(kptr, imatrix, LPCODE, lp.K_INDEX, seed, offset, trans=True)
    # at this point, do the quantized dot for one or both quantized
    if quantq and quantk:
        # typical case
        acc += tl.dot(qq, tl.trans(kq)) / (qs * ks)
    elif quantq:
        # q is quantized, k is not, and kq is possibly processed with icp
        # we need to upcast q to k's type
        acc += tl.dot(qq.to(k.type.ty_element), tl.trans(kq)) / qs
    else:
        # k is quantized, q is not
        acc += tl.dot(qq, tl.trans(kq.to(q.type.ty_element))) / ks
    # what to save for backward? Both qq and kq, but only the scale if it was quantized
    if call_sq_for_q:
        tl.store(qqptr, qq)
        if quantq:
            tl.store(qsptr, qs)
    if call_sq_for_k:
        tl.store(kqptr, kq)
        if quantk:
            tl.store(ksptr, ks)
    return acc

# fmt: off
@triton.jit
def example_kernel(
    Q, Qq, Qs, K, Kq, Ks, Out, Imatrix,
    q_sz_m, q_sz_d, q_st_m, q_st_d, k_sz_n, k_sz_d, k_st_n, k_st_d, o_sz_m, o_sz_n, o_st_m, o_st_n,
    qq_st_m, qq_st_d, qs_st_m, qs_st_d, kq_st_n, kq_st_d, ks_st_n, ks_st_d, philox_seed, philox_offset,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_Q: tl.constexpr, LPCODE: tl.constexpr,
):
    """Main kernel for QK dot product, called from example_launch."""
    if Imatrix is not None:
        offs_q = tl.arange(0, BLOCK_Q)
        imatrix = tl.load(Imatrix + offs_q[:, None] * BLOCK_Q + offs_q[None, :])
    else:
        imatrix = None
    # Get the block ptrs
    start_m = tl.program_id(0)
    m_offset = start_m * BLOCK_M
    start_n = tl.program_id(1)
    n_offset = start_n * BLOCK_N
    NUMQ: tl.constexpr = tl.cdiv(BLOCK_D, BLOCK_Q)
    q_blk_ptr = get_block_ptr(Q, q_sz_m, q_sz_d, q_st_m, q_st_d, BLOCK_Q, m_offset, 0)
    if lp.needs_quant(LPCODE, lp.Q_INDEX):
        qq_blk_ptr = get_block_ptr(Qq, q_sz_m, q_sz_d, qq_st_m, qq_st_d, BLOCK_Q, m_offset, 0)
        qs_blk_ptr = get_block_ptr(Qs, q_sz_m, NUMQ, qs_st_m, qs_st_d, BLOCK_Q, m_offset, 0)
    else:
        qq_blk_ptr = None
        qs_blk_ptr = None
    k_blk_ptr = get_block_ptr(K, k_sz_n, k_sz_d, k_st_n, k_st_d, BLOCK_Q, n_offset, 0)
    if lp.needs_quant(LPCODE, lp.K_INDEX):
        kq_blk_ptr = get_block_ptr(Kq, k_sz_n, k_sz_d, kq_st_n, kq_st_d, BLOCK_Q, n_offset, 0)
        ks_blk_ptr = get_block_ptr(Ks, k_sz_n, NUMQ, ks_st_n, ks_st_d, BLOCK_Q, n_offset, 0)
    else:
        kq_blk_ptr = None
        ks_blk_ptr = None
    o_blk_ptr = get_block_ptr(Out, o_sz_m, o_sz_n, o_st_m, o_st_n, BLOCK_Q, m_offset, n_offset)

    # we are going to accumulate a QXQ tile in the output, the sum of NUMQ dot products, each of which is QxQ
    acc = tl.zeros([BLOCK_Q, BLOCK_Q], dtype=tl.float32)
    # tl.range could spawn more parallelism, as opposed to python range ot tl.arange, which is torch.arange
    for d in tl.range(0, BLOCK_D, BLOCK_Q):
        q_blk_ptr = tl.advance(q_blk_ptr, (0, d))
        k_blk_ptr = tl.advance(k_blk_ptr, (0, d))
        # the seed and offset are used to generate the random numbers for stochastic rounding
        acc += example_kernel_qdot(
            acc, q_blk_ptr, qq_blk_ptr, qs_blk_ptr, k_blk_ptr, kq_blk_ptr, ks_blk_ptr,
            LPCODE, imatrix, philox_seed, philox_offset
        )
    tl.store(o_blk_ptr, acc)
# fmt on

def example_launch(lpconfig, shape, torch_dtype, im_dtype, outlier_prob, outlier_scale, outlier_range):
    """Main function to run the example."""
    torch_dtype = tcast.datatype(torch_dtype).nspec.torch_dtype
    im_dtype = tcast.datatype(im_dtype).nspec.torch_dtype
    q = torch.randn(shape, dtype=torch_dtype, device="cuda")
    k = torch.randn(shape, dtype=torch_dtype, device="cuda")
    if outlier_prob > 0.0:
        q = tcast.utils.make_outliers(q, outlier_scale, outlier_range, outlier_prob)
        k = tcast.utils.make_outliers(k, outlier_scale, outlier_range, outlier_prob)
    out = torch.zeros((q.size(0), k.size(0)), dtype=torch_dtype, device="cuda")
    qsize, ksize, osize = q.size(), k.size(), out.size()
    qstride, kstride, ostride = q.stride(), k.stride(), out.stride()
    # setup for quantizaton
    if lpconfig.enabled:
        qq, qs = lpconfig.make_quant_and_scale(q, lp.Q_INDEX)
        kq, ks = lpconfig.make_quant_and_scale(k, lp.K_INDEX)
        qqstride, qsstride, kqstride, ksstride = qq.stride(), qs.stride(), kq.stride(), ks.stride()
    else:
        qq = qs = kq = ks = None
        qqstride = qsstride = kqstride = ksstride = (0, 0)

    # block sizes
    BLOCK_Q = lpconfig.block_size[-1]
    BLOCK_M, BLOCK_N, BLOCK_D = 64, 64, 128
    LPCODE = lpconfig.code

    assert BLOCK_M % BLOCK_Q == 0, "BLOCK_M must be multiple of BLOCK_Q"
    assert BLOCK_N % BLOCK_Q == 0, "BLOCK_N must be multiple of BLOCK_Q"
    assert BLOCK_D % BLOCK_Q == 0, "BLOCK_D must be multiple of BLOCK_Q"

    # incoherency matrix
    imatrix = lpconfig.get_incoherence_matrix(BLOCK_Q, im_dtype)
    philox_seed, philox_offset = 0x1BF58, 0x1D4B49
    # launch the kernel
    grid = (triton.cdiv(q.size(0), BLOCK_M), triton.cdiv(k.size(0), BLOCK_N))
    example_kernel[grid](
        q, qq, qs, k, kq, ks, out, imatrix,
        *qsize, *qstride, *ksize, *kstride, *osize, *ostride, *qqstride, *qsstride, *kqstride, *ksstride,
        philox_seed, philox_offset, BLOCK_M, BLOCK_N, BLOCK_D, BLOCK_Q, LPCODE
    )

def get_args(args):
    desc = f"TensorCast attention example tcast version {tcast.__version__}"
    parser = argparse.ArgumentParser(description=desc)
    ### params for configuration

    # less verbose configuration methods; use one of these or all of the ones below these three
    parser.add_argument("--code", type=int, default=None, help="get params from code, 0 to disable")
    parser.add_argument("--json_path", type=Path, default=None, help="get params from json file")
    parser.add_argument("--shortcut", type=str, default=None, help="get params from predefined shortcut")

    # datatypes: "None" or no argument means do not quantize, all four standard fp8 types are supported
    parser.add_argument("--q_dtype", type=str, default="fp8", help="dtype for q, in [fp8, fp8n, bf8, bf8n, None]")
    parser.add_argument("--k_dtype", type=str, default="fp8", help="dtype for k")
    # below not needed for qk GEMM example, but are for full attention
    # parser.add_argument("--v_dtype", type=str, default="fp8", help="dtype for v")
    # parser.add_argument("--p_dtype", type=str, default="fp8", help="dtype for p")
    # parser.add_argument("--ds_dtype", type=str, default="bf8", help="dtype for ds")
    # parser.add_argument("--do_dtype", type=str, default="bf8", help="dtype for do")

    # scale datatype: "None" or no argument means uae the type of the unquantized tensor, fp32, fp16, and bf16 are supported
    parser.add_argument("--scale_dtype", type=str, default="float32", help="dtype for scale [float32, float16, bfloat16, None]")

    # incoherence processing
    parser.add_argument("--icp_qk", action="store_true", help="do incoherence for qk, dsk, dsq")
    # not needed for qk GEMM example
    # parser.add_argument("--icp_pv", action="store_true", help="do incoherence for pv, dov, pdo")
    parser.add_argument("--icp_fp32", action="store_true", help="cast icp matrices to fp32 before tl.dot")

    # block size: square blocks for now, powers of two up to 128
    parser.add_argument("--block_size", type=int, default=32, help="square block size for quantization")

    ### other params for example

    # size of input tensors
    parser.add_argument("--input_size", type=int, nargs="+", default=[1024, 1024], help="QKV size")
    parser.add_argument("--input_dtype", type=str, default="bfloat16", help="QKV torch.dtype")
    parser.add_argument("--im_dtype", type=str, default="float32", help="incoherency matrix torch.dtype")

    # outlier insertion, to exercise ICP
    parser.add_argument("--outlier_prob", type=float, default=0.005, help="outliers probability, use 0.0 to disable")
    parser.add_argument("--outlier_scale", type=int, default=3, help="outliers scale for last dimension")
    parser.add_argument("--outlier_range", type=int, default=5, help="2**n multiplier for outliers")

    return parser.parse_args(args if args else None)


def get_configuration(args):
    if args.code is not None:
        return args.code
    if args.json_path is not None:
        return args.json_path
    if args.shortcut is not None:
        return args.shortcut
    return {
        "block_size": (args.block_size, args.block_size),
        "block_axes": (0, 1),
        "q_dtype": tcast.datatype(name=args.q_dtype) if args.q_dtype.lower() != "none" else None,
        "k_dtype": tcast.datatype(name=args.k_dtype) if args.k_dtype.lower() != "none" else None,
        # "v_dtype": tcast.datatype(name=args.v_dtype) if args.v_dtype.lower() != "none" else None,
        # "p_dtype": tcast.datatype(name=args.p_dtype) if args.p_dtype.lower() != "none" else None,
        # "ds_dtype": tcast.datatype(name=args.ds_dtype) if args.ds_dtype.lower() != "none" else None,
        # "do_dtype": tcast.datatype(name=args.do_dtype) if args.do_dtype.lower() != "none" else None,
        "scale_dtype": tcast.datatype(args.scale_dtype) if args.scale_dtype.lower() != "none" else None,
        "icp_qk": args.icp_qk,
        # "icp_pv": args.icp_pv,
        "icp_fp32": args.icp_fp32,
    }


def get_example_args(args):
    return (
        tuple(args.input_size),
        args.input_dtype,
        args.im_dtype,
        args.outlier_prob,
        args.outlier_scale,
        args.outlier_range,
    )


if __name__ == "__main__":
    import sys
    args = get_args(sys.argv[1:])
    cfg_input = get_configuration(args)
    lpconfig = tcast.configuration(cfg_input) if isinstance(cfg_input, Path | int | str) else tcast.configuration(**cfg_input)
    shape, torch_dtype, im_dtype, outlier_prob, outlier_scale, outlier_range = get_example_args(args)
    example_launch(lpconfig, shape, torch_dtype, im_dtype, outlier_prob, outlier_scale, outlier_range)
