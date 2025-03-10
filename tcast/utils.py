#!/usr/bin/env python
# tcast/utils.py: utility functions for tensorcast package
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import importlib
import logging
import math
import random

import numpy
from scipy.stats import kurtosis as ref_kurtosis
import torch
import triton


# fmt: off
def is_triton_available() -> bool: return importlib.util.find_spec("triton") is not None
def is_float8_available() -> bool: return hasattr(torch, "float8_e4m3fn")
def is_float8_fnuz_available() -> bool: return hasattr(torch, "float8_e4m3fnuz")
def is_power_of_2(n: int) -> bool: return (n & (n - 1)) == 0 if n >= 0 else False
def next_power_of_2(n: int) -> bool: return 1 << (n - 1).bit_length() if n else 1
def cdiv(a: int, b: int) -> int: return math.ceil(a / b)
def next_multiple(n: int, m: int) -> int: return cdiv(n, m) * m
def is_hip() -> bool: return triton.runtime.driver.active.get_current_target().backend == "hip"
def get_arch(): return triton.runtime.driver.active.get_current_target().arch
def is_mi300() -> bool: return is_hip() and get_arch().startswith("gfx94")
def is_fp8() -> bool: return is_mi300() or get_arch() >= 89
def is_cdna() -> bool: return is_hip() and get_arch() in ('gfx940', 'gfx941', 'gfx942', 'gfx90a', 'gfx908')
def is_rdna() -> bool: return is_hip() and get_arch() in ("gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201")
def maybe_contiguous(x: torch.Tensor): return x.contiguous() if x is not None and x.stride(-1) != 1 else x
# fmt: on


def set_seed(seed=0):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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


def printoptions(precision: int = 8):
    """Set PyTorch printoptions to something useful."""
    torch.set_printoptions(precision=precision, sci_mode=False)


logger = None


def get_logger(name: str = "tcast") -> logging.Logger:
    """Set up logging."""
    global logger
    if logger is not None:
        return logger
    logger = logging.getLogger(name)
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


def kurtosis(x):
    """Compute the kurtosis of a tensor."""
    return ref_kurtosis(x.flatten().cpu().numpy(), fisher=True)


def make_outliers(
    x: torch.Tensor,
    scale: float = None,  # multiply x by rand * scale
    range: int = 5,  # exponent (2**range) to multiply by x values to produce outliers
    prob: float = 0.0,  # probability of outliers (default no outliers)
) -> torch.Tensor:
    """Applies outliers to the given tensor to better test efficacy of incoherence processing."""
    if scale is not None:
        x *= torch.rand(x.size(-1), device=x.device) * scale
    if prob > 0.0:
        mask = torch.nn.functional.dropout(x, p=1.0 - prob) != 0.0
        oscale = torch.randint_like(x, 1, range + 1).float().exp2()
        x[mask] *= oscale[mask]
    return x
