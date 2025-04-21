#!/usr/bin/env python
# tests/utils.py: unit test utilities
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import functools
import importlib

import torch

import tcast

SHAPES_2D = [(1024, 1024)]
SHAPES_3D = [(8, 128, 64)]
SHAPES_4D = [(8, 128, 64, 64)]
SHAPES_23D = SHAPES_2D + SHAPES_3D
SHAPES_234D = SHAPES_2D + SHAPES_3D + SHAPES_4D

IMSIZES = [8, 16, 32, 64, 128]

TORCH_DTYPES_16 = [torch.float16, torch.bfloat16]
TORCH_DTYPES_32 = [torch.float32]
TORCH_DTYPES_8 = [torch.float8_e5m2, torch.float8_e5m2fnuz, torch.float8_e4m3fn, torch.float8_e4m3fnuz]
TORCH_DTYPES_32_16 = TORCH_DTYPES_32 + TORCH_DTYPES_16
TORCH_DTYPES_16_8 = TORCH_DTYPES_16 + TORCH_DTYPES_8
TORCH_DTYPES_32_16_8 = TORCH_DTYPES_32_16 + TORCH_DTYPES_8

ROUNDMODE_E, ROUNDMODE_A, ROUNDMODE_Z, ROUNDMODE_S = [["even"], ["away"], ["zero"], ["stochastic"]]
ROUNDMODE_EA = ROUNDMODE_E + ROUNDMODE_A
ROUNDMODE_ALL = ROUNDMODE_E + ROUNDMODE_A + ROUNDMODE_S + ROUNDMODE_Z

COMPMODE_T, COMPMODE_P = [["triton"], ["torch"]]
COMPMODDE_ALL = COMPMODE_T + COMPMODE_P

SCALEMODE_F, SCALEMODE_C, SCALEMODE_M, SCALEMODE_O, SCALEMODE_T = [["floor"], ["ceil"], ["midmax"], ["option3"], ["topbinade"]]
SCALEMODE_FM = SCALEMODE_F + SCALEMODE_M
SCALEMODE_ALL = SCALEMODE_F + SCALEMODE_C + SCALEMODE_M + SCALEMODE_O + SCALEMODE_T

V2_AVAILABLE = importlib.util.find_spec("tcastv2") is not None
MX_AVAILABLE = importlib.util.find_spec("mx") is not None
SQT_AVAILABLE = importlib.util.find_spec("sqt") is not None
TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None

COMPARE_MX, COMPARE_V2, COMPARE_SQT, COMPARE_TORCH, COMPARE_TCAST = ["mx"], ["v2"], ["sqt"], ["torch"], ["tcast"]
COMPARE_EXTERNAL = COMPARE_MX + COMPARE_SQT + COMPARE_V2

tcast.initialize(
    roundmode="even",
    scalemode="floor",
    computemode="torch",
    castmode="virtual",
    precision=8,
    sci_mode=False,
    logname="tcast_test",
    device="cuda",
    seed=19,
)
logger = tcast.get_logger()


def _versions():
    """Get versions of TensorCast, PyTorch, Triton, etc."""
    triton_version = tcastv2_version = sqt_version = None
    if TRITON_AVAILABLE:
        import triton

        triton_version = triton.__version__
    if SQT_AVAILABLE:
        import sqt

        sqt_version = sqt.core.__version__.version
    if V2_AVAILABLE:
        import tcastv2

        tcastv2_version = tcastv2.__version__
    return triton_version, tcastv2_version, sqt_version


triton_version, tcastv2_version, sqt_version = _versions()

logger.info(
    "TensorCast: Specification, conversion and compression of arbitrary datatypes.\n"
    f"TensorCast version: {tcast.__version__}\n"
    f"PyTorch version: {torch.__version__}\n"
    f"Triton version: {triton_version if TRITON_AVAILABLE else 'unavailable'}\n"
    f"tcast V2 version: {tcastv2_version if V2_AVAILABLE else 'unavailable'}\n"
    f"Microxcaling MX version: {'unknown, but available' if MX_AVAILABLE else 'unavailable'}\n"
    f"SQT version: {sqt_version if SQT_AVAILABLE else 'unavailable'}\n"
)

assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)


def assert_quantized_tolerance(q, ref, dtype, virtual):  # TODO(ericd): implement
    """Assert that quantized tensor is within tolerance (not that it is correct)."""
    # assumes that q is not scaled to dtype range, so scale the reference tensor as well
    ...


def assert_quantized_representable(q, dtype):
    """Assert that quantized tensor contains values representable in the dtype."""
    # assumes that q is scaled to dtype range
    nline = torch.tensor(tcast.NumberLine(dtype.nspec).line)
    unique = torch.unique(q)
    membership = torch.isin(unique, nline, assume_unique=True)
    if not membership.all():
        wrong = unique.numel() - membership.sum()
        logger.info(f"FAIL: Quantized tensor contains {wrong} unique values not representable in dtype {dtype}")
        return True
    return False # did we fail?
        # raise AssertionError(f"Quantized tensor contains {wrong} unique values not representable in dtype {dtype}")


def dependency_assert(lib: str = "v2"):
    """Assert with message for v2/mx/sqt tests."""
    if lib == "v2" and not V2_AVAILABLE:
        logger.error(
            "tcastv2 package not found:\n"
            "> mkdir tmp; cd tmp\n"
            "> git clone https://github.com/ROCm/tensorcast.git\n"
            "> git switch v2\n"
            "> cp -r tensorcast/tcast $TCAST/tcastv2\n"
        )
        raise AssertionError("tcastv2 package not found")
    if lib == "mx" and not MX_AVAILABLE:
        logger.error(
            "mx package not found:\n"
            "> mkdir tmp; cd tmp\n"
            "> git clone github.com/microsoft/microxcaling\n"
            "> cp -r microxcaling/mx $TCAST/mx\n"
        )
        raise AssertionError("mx package not found")
    if lib == "sqt" and not SQT_AVAILABLE:
        logger.error(
            "sqt package not found; need to get access from @alirezak:\n"
            "> mkdir tmp; cd tmp\n"
            "> git clone https://github.com/Xilinx/sqt.git\n"
            "> cp -r sqt/sqt $TCAST/sqt\n"
        )
        raise AssertionError("sqt package not found")


def get_dset(dset_name) -> tuple[tcast.DataType]:
    """Get a dset from tcast with an abbreviated name."""
    if not dset_name.startswith("dset_"):
        dset_name = "dset_" + dset_name
    if not hasattr(tcast, dset_name):
        raise AttributeError(f"tcast has no attribute {dset_name}")
    return getattr(tcast, dset_name)


def filter_dset(dset: tuple[tcast.DataType], attrs: str, maxbits: int = None, minbits: int = None) -> tuple[tcast.DataType]:
    """Filter a datatype_set by max and min bits."""
    if maxbits is not None:
        dset = tuple(d for d in dset if d.nspec.bits <= maxbits)
    if minbits is not None:
        dset = tuple(d for d in dset if d.nspec.bits >= minbits)
    return dset
