#!/usr/bin/env python
# tcast/scale.py: scaling format specification
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

from dataclasses import dataclass
import re

from .number import NumberSpec
from .utils import cdiv, get_logger, is_power_of_2

logger = get_logger()


@dataclass
class TileSpec:
    """Specifies tile characterisics for one axis.  Ridiculous error checking for a tuple."""

    tsize: int | str | None
    ssize: int | str | None
    sparse_n: int | str | None
    sparse_m: int | str | None
    axis: int = None

    # fmt: off
    @property
    def is_tensor(self) -> bool: return self.axis == 1 and self.tsize is None
    @property
    def is_channel(self) -> bool: return self.tsize == 0
    @property
    def is_tile(self) -> bool: return self.tsize is not None and self.tsize > 0
    @property
    def is_subtile(self) -> bool: return self.ssize is not None and self.ssize != self.tsize
    @property
    def is_sparse(self) -> bool: return self.sparse_m not in [None, 1]
    @property
    def is_dummy(self) -> bool: return all(v is None for v in (self.tsize, self.ssize, self.sparse_n, self.sparse_m))
    # fmt: on

    def description(self) -> str:
        """Returns a string description of the tile specification."""
        desc = f"axis {self.axis} "
        if self.is_tensor:
            desc += "tensor"
        if self.is_channel:
            desc += "channel "
        elif self.is_tile:
            desc += f"tile {self.tsize} "
            if self.is_subtile:
                desc += f"subtile {self.ssize} tile {self.tsize} "
        if self.is_sparse:
            desc += f"sparse {self.sparse_n}:{self.sparse_m} "
        return desc.strip()

    def __post_init__(self):
        if isinstance(self.tsize, str):
            self.tsize = int(self.tsize[1:])
        if self.tsize and not (is_power_of_2(self.tsize) and self.tsize in range(4, 512)):
            raise ValueError(f"TileSpec: '{self.tsize}' is not a supported tile size.")
        if isinstance(self.ssize, str):
            self.ssize = int(self.ssize[1:])
        if self.is_tile and self.ssize is None:
            self.ssize = self.tsize
        if not self.is_tile and self.ssize is not None:
            raise ValueError("TileSpec: subtile size cannot be specified for non-tile.")
        if self.ssize is not None and not (self.is_tile and is_power_of_2(self.ssize) or self.tsize % self.ssize != 0):
            raise ValueError("TileSpec: subtile size {'ssize'} must be a power of 2 and divide tile size {'tsize'}.")
        if isinstance(self.sparse_n, str):
            self.sparse_n = int(self.sparse_n[1:])
        if isinstance(self.sparse_m, str):
            self.sparse_m = int(self.sparse_m[1:])
        if (self.sparse_n is None) != (self.sparse_m is None):
            raise ValueError("TileSpec: sparse M and N must both be specified (or not).")
        if self.is_sparse and (self.axis != 1 or not self.is_tile):
            raise ValueError("TileSpec: sparsity only valid for tile on last axis.")
        if self.sparse_n is not None:
            if self.sparse_m not in (2, 4, 8, 16):
                raise ValueError(f"ScaleSpec: sparse M '{self.sparse_m}' must be in (2, 4, 8, 16).")
            if self.sparse_n not in range(1, self.sparse_m):
                raise ValueError(f"ScaleSpec: sparse N '{self.sparse_n}' must be in [1, sparse_m) '{self.sparse_m}'.")
            if self.ssize and self.ssize % self.sparse_m != 0:
                raise ValueError(f"ScaleSpec: sparse M '{self.sparse_m}' must divide subtile '{self.ssize}'.")
            if self.tsize and self.tsize % self.sparse_m != 0:
                raise ValueError(f"ScaleSpec: sparse M '{self.sparse_m}' must divide tile '{self.tsize}'.")


@dataclass
class ScaleSpec:
    """Specifies scaling method for a given NumberSpec."""

    name: str = None
    tile0: int = None
    tile1: int = None
    subtile0: int = None
    subtile1: int = None
    sparse_n: int = None  # per the literature, M is the total values and N is the dense values
    sparse_m: int = None  # even if that makes no sense to me whatsoever
    scale: NumberSpec = None
    zero: NumberSpec = None
    _tenscale: NumberSpec = None
    desc: str = ""

    def __init__(self, code: str):
        self._decode(code)
        self._check()

    # fmt: off
    @property
    def is_tile(self) -> bool: return self.tile1 is not None and self.tile1 > 1
    @property
    def is_exponent(self) -> bool: return self.scale.is_exponent
    @property
    def is_zero_float(self) -> bool: return self.zero.is_float if self.zero else False
    @property
    def is_2d(self) -> bool: return self.is_tile and self.tile0 != 1
    @property
    def is_channel(self) -> bool: return self.tile0 == 0 or self.tile1 == 0
    @property
    def is_tensor(self) -> bool: return self.tile1 is None
    @property
    def is_subtile(self) -> bool: return self.is_tile and self.subtile1 != self.tile1
    @property
    def is_sparse(self) -> bool: return self.sparse_m not in [None, 1]
    @property
    def is_square(self) -> bool: return self.is_2d and self.tile0 == self.tile1
    @property
    def is_multiscale(self) -> bool: return self._tenscale is not None
    @property
    def sparse_ratio(self) -> int: return cdiv(self.sparse_m, self.sparse_n) if self.is_sparse else 1
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

    def _decode(self, code: str) -> str:
        """Sets fields based on input string code."""
        # 1 to 3 NumberSpecs, 0 to 2 tile specs
        # 1 nspec, 1 or 2 tilespecs: channel or tile-scaled (float, int) self.scale
        # the next are ambiguous without knowing the nspec type
        # 2 nspecs, 1 or 2 tilespecs: multiscale (float, int) channel or tile-scaled self.tenspec, self.scale
        # 2 nspecs, 1 or 2 tilespecs: channel or tile-scaled (uint) self.scale and self.zero
        # 3 nspecs, 0 tilespecs: invalid
        # 3 nspecs, 1 or 2 tilespecs: multiscale (uint) self.tenscale, self.scale, self.zero

        ### parse the string code into NumberSpecs and TileSpecs

        self.name = code = code.lower()
        multiscale_if_ambiguous = False
        if code.startswith("multi_"):
            multiscale_if_ambiguous = True
            code = code[6:]
        nspecs, tspecs, segments = [], [], code.split("_")
        for i, segment in enumerate(segments):
            if NumberSpec.valid(segment):
                if len(nspecs) > 2:
                    raise ValueError(f"ScaleSpec: more than three NumberSpecs provided in string code '{self.name}'.")
                if len(tspecs) > 0:
                    raise ValueError(f"ScaleSpec: NumberSpec found after tile specification in '{self.name}'.")
                nspecs.append(NumberSpec(segment))
            elif m := re.fullmatch(r"(t\d*)(s\d+)?(n\d+)?(m\d+)?", segment):
                axis = 0 if i == len(segments) - 2 else 1
                if len(nspecs) == 0:
                    raise ValueError(f"ScaleSpec: tile specification found before NumberSpec in '{self.name}'.")
                if len(tspecs) > 2:
                    raise ValueError(f"ScaleSpec: more than two tile specifications provided in string code '{self.name}'.")
                tspecs.append(TileSpec(*m.group(1, 2, 3, 4), axis))
            else:
                raise ValueError(f"ScaleSpec: '{segment}' is neither a valid number or tile specification.")

        ### check the parsed values and set the fields

        nspecs = tuple(nspecs)
        tspecs = tuple(tspecs)
        num_nspecs, num_tspecs = len(nspecs), len(tspecs)  # explicit tspecs
        if num_tspecs == 2 and tspecs[0].is_channel and tspecs[1].is_channel:
            raise ValueError(f"ScaleSpec: both tile specifications cannot be channel scales '{self.name}'.")
        if num_nspecs == 0:
            raise ValueError(f"ScaleSpec: no NumberSpecs provided in string code '{self.name}'.")
        if num_tspecs == 0 and num_nspecs == 3:
            raise ValueError(f"ScaleSpec: more than two NumberSpecs provided with no tile specs '{self.name}'.")
        # make sure we have two tspecs to work with
        if num_tspecs == 0:
            tspecs = (TileSpec(None, None, None, None, 0), TileSpec(None, None, None, None, 1))
        elif num_tspecs == 1:
            tspecs = (TileSpec(None, None, None, None, 0), tspecs[0])

        # capture the tile and subtile sizes and sparsity

        t0, t1 = tspecs
        self.tile1 = t1.tsize
        self.subtile1 = t1.ssize if t1.is_subtile else self.tile1
        self.sparse_n = t1.sparse_n if t1.is_sparse else 1
        self.sparse_m = t1.sparse_m if t1.is_sparse else 1
        if t0.is_dummy:
            self.tile0 = 1
            self.subtile0 = 1
        else:
            self.tile0 = t0.tsize
            self.subtile0 = t0.ssize

        # add the scale nspecs

        if num_nspecs == 1:
            self.scale = nspecs[0]
        elif num_nspecs == 2:
            if num_tspecs == 0:
                self.scale, self.zero = nspecs
            elif multiscale_if_ambiguous:
                self._tenscale, self.scale = nspecs
            else:
                self.scale, self.zero = nspecs
        else:
            self._tenscale, self.scale, self.zero = nspecs

        # set the description

        desc = "multiscale " if self.is_multiscale else ""
        if self.is_2d:
            assert self.is_tile
            t0 = "channel" if self.tile0 == 0 else str(self.tile0)
            t1 = "channel" if self.tile1 == 0 else str(self.tile1)
            desc += f"tile ({t0}x{t1}) "
            if self.is_subtile:
                assert not self.is_channel
                assert self.subtile0 != self.tile0 and self.subtile1 != self.tile1
                desc += f"subtile ({self.subtile0}x{self.subtile1}) "
        else:
            if self.is_tensor:
                desc += "tensor "
            elif self.is_channel:
                desc += "channel "
            elif self.is_tile:
                desc += f"tile {self.tile1} "
                if self.is_subtile:
                    desc += f"subtile {self.subtile1} "
        desc += "uint data " if self.zero else "float/int data "
        desc += f"sparse {self.sparse_n}:{self.sparse_m} " if self.is_sparse else ""
        self.desc = desc.strip()
        return self.desc

    def _check(self):
        prefix = f"ScaleSpec: '{self.name}'"
        if not self.scale:
            logger.info(self.desc)
            raise ValueError(f"{prefix} does not specify a scale.")
        if self.zero:
            if not self.scale.is_float:
                logger.info(self.desc)
                raise ValueError(f"{prefix} asymmetric scaling requires a float scale")
            if self.zero.is_exponent or self.zero.is_uint:
                logger.info(self.desc)
                raise ValueError(f"{prefix} asymmetric scaling requires a float or int zero point type")

    @classmethod
    def valid(cls, code: str) -> bool:
        """Checks validity without raising an exception."""
        try:
            cls(code)
            return True
        except (ValueError, NotImplementedError):
            return False
