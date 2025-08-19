"""TensorCast: Conversion and compression of arbitrary datatypes."""
# tcast/utils.py: utility functions for tensorcast package

import importlib

import torch


def is_installed(name: str) -> bool:
    """Without importing, see if a package is present."""
    return importlib.util.find_spec(name) is not None


def is_gpu_available():
    """Check to see if a GPU is present."""
    return torch.cuda.is_available()


def is_float8_available():
    """Check to see if float8 is present in this version of PyTorch."""
    return hasattr(torch, "float8_e4m3fn")


def is_float8_fnuz_available():
    """Check to see if MI300 float8 is present in this version of PyTorch."""
    return hasattr(torch, "float8_e4m3fnuz")


def printoptions(precision: int = 8):
    """Set PyTorch printoptions to something useful."""
    torch.set_printoptions(precision=precision, sci_mode=False)


def check_literal(s: str, lit, none_ok=False) -> None:
    """Check that a string is a literal from a given Literal."""
    vals = lit.__args__
    if s not in vals and (s is not None or not none_ok):
        raise ValueError(f"{s} is not a valid {lit.__name__} value: {','.join(vals)}")
    return s


def is_power_of_2(n: int) -> bool:
    """Check for power of 2."""
    return (n & (n - 1)) == 0


def next_power_of_2(n: int) -> bool:
    """If this is not, find the next highest one."""
    return 1 << (n - 1).bit_length()
