#!/usr/bin/env python
# tcast/scale.py: scaling format specification
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

from dataclasses import dataclass
import re

from .number import NumberSpec
from .utils import cdiv, is_power_of_2


@dataclass
class ScaleSpec:
    """Specifies scaling method for a given NumberSpec."""

    name: str = None
    tile0: int = None
    tile1: int = None
    subtile0: int = None
    subtile1: int = None
    sparse_n: int = 1  # per the literature, M is the total values and N is the dense values
    sparse_m: int = 1  # even if that makes no sense to me whatsoever
    scale: NumberSpec = None
    zero: NumberSpec = None
    _tenscale: NumberSpec = None

    def __init__(self, code: str):
        self._decode(code)
        self._check()

    # fmt: off
    @property
    def is_tile(self) -> bool: return self.tile1 is not None and self.tile1 > 1
    @property
    def is_2d(self) -> bool: return self.is_tile and self.tile0 != 1
    @property
    def is_channel(self) -> bool: return self.tile0 == 0 or self.tile1 == 0
    @property
    def is_tensor(self) -> bool: return self.tile1 is None
    @property
    def is_subtile(self) -> bool: return self.subtile1 != self.tile1 or self.subtile0 != self.tile0
    @property
    def is_sparse(self) -> bool: return self.sparse_m > 1
    @property
    def is_multiscale(self) -> bool: return self._tenscale is not None
    @property
    def sparse_ratio(self) -> int: return cdiv(self.sparse_m, self.sparse_n)
    @property
    def tenscale(self) -> NumberSpec: return self._tenscale or self.scale
    # fmt: on

    def get_tile_info(self, size0: int = None, size1: int = None, virtual: bool = True) -> tuple[int]:
        """Since tile=0 is channel and possible transpose, return specific values given a tensor shape."""
        tile0, tile1, subtile0, subtile1 = self.tile0, self.tile1, self.subtile0, self.subtile1
        if tile0 == 0 or tile1 == 0:
            if tile0 == 0:
                if size0 is None:
                    raise ValueError("ScaleSpec: tile0=0 requires size0 to be specified.")
                tile0 = size0
            if tile1 == 0:
                if size1 is None:
                    raise ValueError("ScaleSpec: tile1=0 requires size1 to be specified.")
                tile1 = size1
        if self.is_sparse and virtual:
            # modify scale tile sizes to account for sparsity
            tile1 *= self.sparse_ratio
            subtile1 *= self.sparse_ratio
        return tile0, tile1, subtile0, subtile1, self.sparse_m, self.sparse_n

    def _set_tile_info(self, tval: str, sval: str, nval: str, mval: str):
        """Sets tile-related fields based on input string values."""
        prefix = f"ScaleSpec: '{self.name}'"
        if tval is None:
            # this is a tensor scale
            if self.zero is not None:
                self._tenscale, self.zero = self.zero, None
            return
        tile = int(tval)
        subtile = int(sval[1:]) if sval else None
        sparse_n = int(nval[1:]) if nval else None
        sparse_m = int(mval[1:]) if mval else None
        if tile > 1024 or not (tile == 0 or is_power_of_2(tile)):
            raise ValueError(f"{prefix} '{tile}' is not a supported tile size.")
        if subtile is None:
            subtile = tile
        elif tile == 0:
            raise ValueError(f"{prefix} tile=0 channel scale does not support subtiles.")
        elif subtile < 2 or not is_power_of_2(subtile) or subtile >= tile or tile % subtile != 0:
            raise ValueError(f"{prefix} '{subtile}' is not a supported subtile size.")
        if self.tile1 is None:
            self.tile1, self.subtile1 = tile, subtile
            self.tile0 = self.subtile0 = 1
        else:
            self.tile0, self.subtile0 = self.tile1, self.subtile1
            self.tile1, self.subtile1 = tile, subtile

        # sparsity can only be 1D
        if (sparse_n is None) != (sparse_m is None):
            raise ValueError(f"{prefix} sparse M and N must both be specified (or not)")
        if sparse_n is not None:
            if self.sparse_m != 1:
                raise ValueError(f"{prefix} sparsity already specified")
            if tile == 0:
                raise NotImplementedError(f"{prefix} sparsity specified for channel scale ")
            if self.is_2d:
                raise NotImplementedError(f"{prefix} sparsity is for 1D scales only")
            if (
                sparse_m % sparse_n != 0
                or not is_power_of_2(sparse_m)
                or sparse_m > 16
                or tile % sparse_m != 0
                or subtile % sparse_m != 0
            ):
                raise ValueError(
                    f"{prefix} sparse N '{sparse_n}' and sparse M '{sparse_m}' must be 0 < N < M"
                    f" and M must be a power of 2 <= 16 and divide evenly into tile '{tile}'"
                )
            self.sparse_n, self.sparse_m = sparse_n, sparse_m

    def _decode(self, code: str) -> None:
        """Sets fields based on input string code."""
        self.name = code = code.lower()
        for segment in code.split("_"):
            if NumberSpec.valid(segment):
                if self.scale is None:
                    self.scale = NumberSpec(segment)
                elif self.zero is None:
                    self.zero = NumberSpec(segment)
                else:
                    raise ValueError(f"ScaleSpec: more than two NumberSpecs provided in string code '{self.name}'.")
            elif m := re.fullmatch(r"t(\d*)(s\d+)?(n\d+)?(m\d+)?", segment):
                self._set_tile_info(*m.group(1, 2, 3, 4))
            else:
                raise ValueError(f"ScaleSpec: '{segment}' is neither a valid number or tile specification.")

    def _check(self):
        prefix = f"ScaleSpec: '{self.name}'"
        if not self.scale:
            raise ValueError(f"{prefix} does not specify a scale.")
        if self.zero:
            if not self.scale.is_float:
                raise ValueError(f"{prefix} asymmetric scaling requires a float scale")
            if self.zero.is_exponent or self.zero.is_uint:
                raise ValueError(f"{prefix} asymmetric scaling requires a float or int zero point type")

    @classmethod
    def valid(cls, code: str) -> bool:
        """Checks validity without raising an exception."""
        try:
            cls(code)
            return True
        except (ValueError, NotImplementedError):
            return False
