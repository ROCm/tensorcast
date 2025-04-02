#!/usr/bin/env python
# tcast/utils.py: utility functions for tensorcast package
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import importlib
import logging
import math
from pathlib import Path
import random
import struct

import numpy
from scipy.stats import kurtosis as ref_kurtosis
import torch
import triton

logger: logging.Logger = None
extern_libs: dict = None


def get_logger(name: str = "tcast", replace: bool = False) -> logging.Logger:
    """Set up logging."""
    global logger
    if logger is not None and not replace:
        return logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    fh = logging.FileHandler((name + ".log").replace(".log.log", ".log"), mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def get_extern_libs() -> dict:
    global extern_libs
    if extern_libs is not None:
        return extern_libs
    if is_triton_available():
        triton_dir = Path(triton.__file__).parent.parent.parent / "third_party"
        if is_cuda():
            extern_libs = {"libdevice": str(triton_dir / "nvidia/backend/lib/libdevice.10.bc")}
        elif is_hip():
            libdir = triton_dir / "amd/backend/lib"
            extern_libs = {str(libdir / f"{lib}.bc"): lib for lib in ["ocml", "ockl"]}
        else:
            raise RuntimeError("unknown backend")
    return extern_libs


# fmt: off
def is_triton_available() -> bool: return importlib.util.find_spec("triton") is not None
def is_float8_available() -> bool: return hasattr(torch, "float8_e4m3fn")
def is_float8_fnuz_available() -> bool: return hasattr(torch, "float8_e4m3fnuz")
def is_power_of_2(n: int) -> bool: return (n & (n - 1)) == 0 if n >= 0 else False
def next_power_of_2(n: int) -> bool: return 1 << (n - 1).bit_length() if n else 1
def cdiv(a: int, b: int) -> int: return math.ceil(a / b)
def next_multiple(n: int, m: int) -> int: return cdiv(n, m) * m
def is_multiple(n: int, m: int) -> int: return n % m == 0
def triton_target() -> bool: return triton.runtime.driver.active.get_current_target()
def is_hip() -> bool: return triton_target().backend == "hip"
def is_cuda() -> bool: return triton_target().backend == "cuda"
def get_arch(): return triton_target().arch
def is_mi300() -> bool: return is_hip() and get_arch().startswith("gfx94")
def is_fp8() -> bool: return is_mi300() or get_arch() >= 89
def is_cdna() -> bool: return is_hip() and get_arch() in ('gfx940', 'gfx941', 'gfx942', 'gfx90a', 'gfx908')
def is_rdna() -> bool: return is_hip() and get_arch() in ("gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201")
def maybe_contiguous(x: torch.Tensor): return x.contiguous() if x is not None and x.stride(-1) != 1 else x
# fmt: on


def triton_fp8_support() -> int:
    """Check if what Triton FP8 support is on this device."""
    # HACK ALERT! We need to know what FP8 support we have in Triton for the snippets,
    # but it isn't clear yet how to do that fro within a @triton.jit kernel.  So it's
    # hardcoded here.  Returns a 4 bit code, where support is 1 for e5m2, 2 for e5m2fnuz,
    # 4 for e4m3fn, and 8 for e4m3fnuz.
    code = 0
    if is_triton_available():
        arch = get_arch()
        if is_hip():
            if arch in ("gfx940", "gfx941", "gfx942"):
                code = 15
            elif arch in ("gfx950",):
                code = 5
            else:
                code = 1
        else:
            code = 1 | int(arch >= 89) << 2
    return code


def set_seed(seed=0, backend: bool = False):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if backend:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    if importlib.util.find_spec("transformers") is not None:
        import transformers

        transformers.set_seed(seed)


def cleanup_memory():
    """Clear GPU memory by running garbage collection and emptying cache."""
    import gc

    if torch.cuda.is_available():
        before = sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))
        gc.collect()
        torch.cuda.empty_cache()
        after = sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))
        logging.debug(f"Memory before: {before}, after: {after}")
    else:
        gc.collect()


def set_default_device(device: str = "cuda") -> None:
    """Set the default device for all tensor creation."""
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("tcast.set_default_device: CUDA is not available")
        dev = device.split(":")
        if len(dev) > 1:
            if dev[1].isdigit() and int(dev[1]) >= torch.cuda.device_count():
                raise ValueError(f"tcast.set_default_device: device {device} is not available")
    elif device != "cpu":
        raise ValueError("tcast.set_default_device: device must be 'cuda' or 'cpu'")
    torch.set_default_device(device)


def printoptions(precision: int = 8):
    """Set PyTorch printoptions to something useful."""
    torch.set_printoptions(precision=precision, sci_mode=False)


def kurtosis(x):
    """Compute the kurtosis of a tensor."""
    return ref_kurtosis(x.float().flatten().cpu().numpy(), fisher=True)


def make_outliers(
    x: torch.Tensor,
    scale: float = 0.0,  # multiply x by rand * scale
    range: int = 5,  # exponent (2**range) to multiply by x values to produce outliers
    prob: float = 0.0,  # probability of outliers (default no outliers)
) -> torch.Tensor:
    """Applies outliers to the given tensor to better test efficacy of incoherence processing."""
    if scale or prob:
        assert range > 1, "Outlier range must be > 1"
        x = x.clone()
        if scale:
            x *= torch.rand(x.size(-1), device=x.device) * scale
        if prob > 0.0:
            mask = torch.nn.functional.dropout(x, p=1.0 - prob) != 0.0
            oscale = torch.randint_like(x, 1, range + 1).float().exp2()
            x[mask] *= oscale[mask]
    return x


def to_string(val, spaced=True, code='f'):
    """ Debug util for visualizing float values."""
    s = ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!' + code, val))
    spaced = spaced and len(s) == 32
    return f"{s[0]} {s[1:9]} {s[9:]}" if spaced else s


def to_float(s, sign=1, exp=8, _=23, bias=127):
    """ Debug util for converting string to float32."""
    frac = 0.0
    s = ''.join(s.strip().split(' '))
    if s[1:9] == "11111111":
        if int(s[9:], base=2) != 0:
            return float("NaN")
        elif s[0] == "0":
            return float("Inf")
        return float("-Inf")
    m = s[exp+sign:]
    for i, ch in enumerate(m):
        if ch == '1':
            frac += pow(2.0, -i -1)
    e = int(s[sign:sign+exp], base=2)
    if e == 0:
        if frac == 0.0:
            return 0.0
        f = 2**(1 - bias) * frac
    else:
        f = 2**(e - bias) * (1.0 + frac)
    if sign == 1 and int(s[0]) == 1:
        f = -f
    return f
