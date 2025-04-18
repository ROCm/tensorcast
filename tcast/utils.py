#!/usr/bin/env python
# tcast/utils.py: utility functions for tensorcast package
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

from enum import Enum
import importlib
import inspect
import logging
import math
import os
from pathlib import Path
import random

import numpy as np
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
    fh = logging.FileHandler(("../" + name + ".log").replace(".log.log", ".log"), mode="w")
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


# filter a datatype_set
def filter_dset(dset, maxbits: int = None, minbits: int = None) -> tuple:
    """Filter a datatype_set by max and min bits."""
    if maxbits is not None:
        dset = tuple(d for d in dset if d.nspec.bits <= maxbits)
    if minbits is not None:
        dset = tuple(d for d in dset if d.nspec.bits >= minbits)
    return dset


def triton_fp8_support() -> int:
    """Check if what Triton FP8 support is on this device."""
    # HACK ALERT! We need to know what FP8 support we have in Triton,
    # but it isn't clear yet how to do that from within a @triton.jit kernel.  So it's
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
    np.random.seed(seed)
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


def convert_enum(enum_type, enum_value, convert_type=None) -> str | int | None:
    """Convert a string, int, or enum instance to a string, int, or enum instance."""
    # enum_type is a class derived from Enum (if not, return None)
    # enum_value is a string, int, or enum instance
    # convert_type is str, int, or None
    if inspect.isclass(enum_type) and issubclass(enum_type, Enum):
        # first, convert enum_value to enum_instance
        if isinstance(enum_value, str):
            enum_value = enum_value.upper()
            if enum_value not in dir(enum_type):
                raise ValueError(f"'{enum_value}' is not a valid {enum_type.__name__} name.")
            enum_instance = enum_type[enum_value]
        elif isinstance(enum_value, int):
            if enum_value >= len(enum_type):
                raise ValueError(f"'{enum_value}' is not a valid {enum_type.__name__} value.")
            enum_instance = enum_type(enum_value)
        elif isinstance(enum_value, enum_type):
            enum_instance = enum_value
        else:
            raise ValueError(f"'{enum_value}' type {type(enum_value)} is not a valid type for enum_value.")
        # convert enum_instance to a str, int, or enum instance
        if convert_type is str:
            return enum_instance.name
        if convert_type is int:
            return enum_instance.value
        return enum_instance
    return None


def getenv(x: str, dflt: str = "", dtype=str, prefix: str = ""):
    """Get an environment variable with optional default and type conversion."""
    # for TensorCast attention interface, prefix is "TC_"
    # in flash_attn, prefix is usually FLASH_ATTENTION_TRITON_AMD_
    x = x.upper()
    if prefix and not x.endswith(prefix):
        prefix += "_"
    prefix = prefix.upper()
    if prefix and not x.startswith(prefix):
        x = f"{prefix}{x}"
    setting = os.getenv(x, dflt)
    if setting is None or setting.lower() == "none":
        return None
    if dtype is bool:
        return setting in ("1", "true", "yes")
    if dtype in (np.int64,  int):
        try:
            if setting.startswith("0x"):
                return dtype(int(setting, 16))
            elif setting.replace("-", "").isdigit():
                return dtype(setting)
        except ValueError as err:
            raise ValueError(f"Invalid integer value for {x}: {setting}") from err
    if dtype is float and setting.replace(".", "").replace("-", "").isdigit():
        return dtype(setting)
    if dtype is tuple:
        return tuple(int(i) for i in setting.split(","))
    if issubclass(dtype, Enum):
        if setting.isdigit():
            return convert_enum(Enum, int(setting))
        else:
            return convert_enum(Enum, setting)
    if dtype is Path:
        return Path(setting)
    return setting


def to_string(val, spaced=True, dtype=torch.float32):
    """Debug util for visualizing float values."""
    if dtype == torch.float32:
        idtype, space = torch.int32, 8
    elif dtype == torch.float16:
        idtype, space = torch.int16, 5
    elif dtype == torch.bfloat16:
        idtype, space = torch.int16, 8
    else:
        raise ValueError(f"Unsupported dtype: {dtype}. Supported types are torch.float32, torch.float16, and torch.bfloat16.")
    bits = torch.tensor([val], dtype=dtype).view(idtype).item()
    s = f"{bits:0(dtype.itemsize*8}b"
    spaced = spaced and len(s)in [32, 16]
    return f"{s[0]} {s[1:space+1]} {s[space+1:]}" if spaced else s


def to_float(s: str, dtype=torch.float32) -> float:
    """Convert a binary string to a float32."""
    s = s.replace(" ", "").strip()
    size = dtype.itemsize * 8
    if len(s) != size:
        raise ValueError(f"Input string must be {size} bits long for dtype {dtype}")
    if dtype == torch.float32:
        ebits, mbits, bias = 8, 23, 127
    elif dtype == torch.float16:
        ebits, mbits, bias = 5, 10, 15
    elif dtype == torch.bfloat16:
        ebits, mbits, bias = 8, 7, 127
    else:
        raise ValueError(f"Unsupported dtype: {dtype}. Supported types are torch.float32, torch.float16, and torch.bfloat16.")
    sign_bit = int(s[0], 2)
    exponent = int(s[1:1 + ebits], 2)
    mantissa = int(s[1 + ebits:], 2)
    if exponent == (1 << ebits) - 1:  # All exponent bits are 1
        if mantissa != 0:
            return float("NaN")
        return float("Inf") if sign_bit == 0 else float("-Inf")
    elif exponent == 0:  # All exponent bits are 0 (denormalized number)
        if mantissa == 0:
            return 0.0
        return (-1) ** sign_bit * 2 ** (1 - bias) * (mantissa / (1 << mbits))
    return (-1) ** sign_bit * 2 ** (exponent - bias) * (1 + mantissa / (1 << mbits))
