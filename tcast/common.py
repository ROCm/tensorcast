#!/usr/bin/env python
# tcast/common.py: constants, enums, etc.
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

from enum import Enum
import logging
from typing import ClassVar, NamedTuple

import torch

from .utils import is_triton_available

logger = logging.getLogger(f"tcast.{__name__}")

EPS = torch.finfo(torch.float32).smallest_normal
STD_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz)


class RoundMode(Enum):
    """Round modes, must match extension."""

    ZERO = 0  # nearest, ties toward zero
    AWAY = 1  # nearest, ties away from zero
    EVEN = 2  # nearest, ties to even
    STOCHASTIC = 3


class ScaleMode(Enum):
    """Scale modes, must match extension."""

    FLOOR = 0  # OCP MX standard
    CEIL = 1  # proposed by Intel
    MIDMAX = 2  # AMD's finest
    OPTION3 = 3  # proposed by Meta
    TOPBINADE = 4  # proposed by Microsoft


class ComputeMode(Enum):
    """Compute mode (if available)."""

    TORCH = 0  # torch operations
    TRITON = 1  # triton operations


class CastMode(Enum):
    """Virtual (fake), actual (with scales), compressed (packed sparsity and multiple values packed into single elements."""

    VIRTUAL = 0  # fake quantization, just returns with the same torch.dtype and shape
    ACTUAL = 1  # data, scale, zero, meta, and mask in the appropriate datatypes
    COMPRESS = 2  # packs multiple values into single elements, e.g. 2 4-bit codebook indices, sparsity mask 8x compression
    UPCAST = 3  # upcast to a larger datatype, e.g. 4-bit to 8-bit, 8-bit to 16-bit, etc.


class InfNaN(Enum):
    """Inf and NaN handling."""

    IEEE = 0  # standard IEEE 754 methods
    FN = 1  # all non-sign bits on = NaN, all non-sign bits off = Inf
    FNUZ = 2  # neg zero = Nan, no inf
    INUZ = 3  # proposed by IEEE working group P3109

    def suffix(self) -> str:
        """Return the suffix for the datatype."""
        return self.name.lower() if self != InfNaN.IEEE else ""


def get_enum(etype: Enum, s: str, silent: bool = False) -> int | None:
    """Create the enum from a string matching the name."""
    if isinstance(s, etype):
        return s
    if isinstance(s, str):
        if s.upper() in dir(etype):
            return etype[s.upper()]
        if not silent:
            raise ValueError(f"'{s}' is not a valid {etype.__name__} name.")
    return None


class ScaleData(NamedTuple):
    """ScaleData: scales, zero points, and meta data for a datatype."""

    scale: torch.Tensor
    zero: torch.Tensor
    tenscale: torch.Tensor
    meta: torch.Tensor
    mask: torch.Tensor


IMPLICIT_CODEBOOKS = dict(
    cb42fe0123_e3m2fnuz=dict(
        index_bits=4,
        meta_bits=1,
        midmax=14.0,
        mappings=[
            [0.0000, 1.0000, 2.0000, 3.0000, 4.0000, 6.0000, 8.0000, 12.0000],
            [0.0000, 0.5000, 1.0000, 1.5000, 2.0000, 3.0000, 4.0000, 6.0000],
            [0.0000, 0.2500, 0.5000, 0.7500, 1.0000, 1.5000, 2.0000, 3.0000],
            [0.0000, 0.1250, 0.2500, 0.3750, 0.5000, 0.7500, 1.0000, 1.5000],
        ],
        midpoints=[
            [0.5000, 1.5000, 2.5000, 3.5000, 5.0000, 7.0000, 10.0000],
            [0.2500, 0.7500, 1.2500, 1.7500, 2.5000, 3.5000, 5.0000],
            [0.1250, 0.3750, 0.6250, 0.8750, 1.2500, 1.7500, 2.5000],
            [0.0625, 0.1875, 0.3125, 0.4375, 0.6250, 0.8750, 1.2500],
        ],
    ),
    cb42f1346_e2m3fnuz=dict(
        index_bits=4,
        meta_bits=1,
        midmax=7.5,
        mappings=[
            [0.0000, 0.7500, 1.2500, 1.7500, 2.5000, 3.5000, 5.0000, 7.0000],
            [0.0000, 0.5000, 1.0000, 1.5000, 2.0000, 3.0000, 4.0000, 6.0000],
            [0.0000, 0.3750, 0.8750, 1.3750, 1.8750, 2.7500, 3.7500, 5.5000],
            [0.0000, 0.1250, 0.6250, 1.1250, 1.6250, 2.2500, 3.2500, 4.5000],
        ],
        midpoints=[
            [0.3750, 1.0000, 1.5000, 2.0000, 3.0000, 4.0000, 6.0000],
            [0.2500, 0.7500, 1.2500, 1.7500, 2.5000, 3.5000, 5.0000],
            [0.1875, 0.6250, 1.1250, 1.6250, 2.2500, 3.2500, 4.5000],
            [0.0625, 0.3750, 0.8750, 1.3750, 1.8750, 2.7500, 3.7500],
        ],
    ),
)


MX2NUMSPEC = dict(
    mxfp8e5='DataType("e5m2", "e8m0_t32", "mxfp8e5")',
    mxfp8e4='DataType("e4m3fn", "e8m0_t32", "mxfp8e4")',
    mxfp6e3='DataType("e3m2fnuz", "e8m0_t32", "mxfp6e3")',
    mxfp6e2='DataType("e2m3fnuz", "e8m0_t32", "mxfp6e2")',
    mxfp4e2='DataType("e2m1fnuz", "e8m0_t32", "mxfp4e2")',
    mxint8='DataType("int8", "e8m0_t32", "mxint8")',
    mxint4='DataType("int4", "e8m0_t32", "mxint4")',
    bfp16='DataType("int8", "e8m0_t8", "bfp16")',
    mx9='DataType("cb81e01", "e8m0_t16s2", "mx9")',
    mx6='DataType("cb51e01", "e8m0_t16s2", "mx6")',
    mx4='DataType("cb31e01", "e8m0_t16s2", "mx4")',
)


class Modes:
    """Static class to manage modes used in casting."""

    round: ClassVar[RoundMode] = RoundMode.EVEN
    scale: ClassVar[ScaleMode] = ScaleMode.FLOOR
    compute: ClassVar[ComputeMode] = ComputeMode.TORCH
    cast: ClassVar[CastMode] = CastMode.ACTUAL
    saved_round: ClassVar[RoundMode] = RoundMode.EVEN
    saved_scale: ClassVar[ScaleMode] = ScaleMode.FLOOR
    saved_compute: ClassVar[ComputeMode] = ComputeMode.TORCH
    saved_cast: ClassVar[CastMode] = CastMode.ACTUAL
    warn_cpu: ClassVar[bool] = True
    warn_gpu: ClassVar[bool] = True
    warn_triton: ClassVar[bool] = True

    @classmethod
    def restore_modes(cls):
        """Restore the saved modes."""
        cls.round, cls.scale, cls.compute, cls.cast = cls.saved_round, cls.saved_scale, cls.saved_compute, cls.saved_cast

    @classmethod
    def set_modes(
        cls, roundmode: str | RoundMode, scalemode: str | ScaleMode, computemode: str | ComputeMode, castmode: str | CastMode
    ) -> dict:
        """Check and set the rounding and other modes."""
        cls.saved_round, cls.saved_scale, cls.saved_compute, cls.saved_cast = cls.round, cls.scale, cls.compute, cls.cast
        roundmode, scalemode, computemode, castmode = (
            get_enum(RoundMode, roundmode),
            get_enum(ScaleMode, scalemode),
            get_enum(ComputeMode, computemode),
            get_enum(CastMode, castmode),
        )
        if roundmode is not None:
            cls.round = roundmode
        if scalemode is not None:
            cls.scale = scalemode
        if castmode is not None:
            cls.cast = castmode
        if computemode is not None:
            if computemode == ComputeMode.TRITON and cls.warn_triton and not is_triton_available():
                logger.warning("triton not installed, using torch")
                cls.warn_triton = False
                cls.compute = ComputeMode.TORCH
            else:
                cls.compute = computemode
        return dict(roundmode=cls.round, scalemode=cls.scale, computemode=cls.compute, castmode=cls.cast)
