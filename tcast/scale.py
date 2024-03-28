"""TensorCast: Conversion and compression of arbitrary datatypes."""
# tcast/scale.py: scaling format specification

from dataclasses import dataclass
import re
from typing import NamedTuple

import torch

from .number import NumberSpec


class ScaleData(NamedTuple):
    """Scale data tensors."""

    scale: torch.Tensor = None
    zero: torch.Tensor = None
    lookup: torch.Tensor = None
    offset: torch.Tensor = None


class ScaledTensor(NamedTuple):
    """Combined tensor and/or scaledata container."""

    tensor: torch.Tensor = None
    scaledata: ScaleData = None


@dataclass
class ScaleSpec:
    """Specifies scaling method for a given NumberSpec."""

    name: str = None
    tile: int = None
    subtile: int = None
    offset: int = None
    dim: int = None
    tile2: int = None
    subtile2: int = None
    dim2: int = None

    is_tensor: bool = False
    is_channel: bool = False
    is_tile: bool = False
    is_subtile: bool = False
    is_offset: bool = False
    is_2d: bool = False

    scale: NumberSpec = None
    zero: NumberSpec = None
    shape: tuple[int] = None

    def __init__(self, code: str):
        self._decode(code)
        self._check()
        if self.is_2d:
            raise NotImplementedError("ScaleSpec: two dimensional scaling is not yet supported.")

    def reshape_tensor(self, tensor: torch.Tensor, subtile: bool = False) -> torch.Tensor:
        """Reshape and/or transpose tensor for scaling."""
        # TODO(ericd) 2D reshape
        self.shape = tensor.shape
        if not self.is_tile:
            return tensor
        tensor = tensor.transpose(self.dim, -1)
        if subtile and self.is_subtile:
            tensor = tensor.reshape(-1, self.tile // self.subtile, self.subtile)
        else:
            tensor = tensor.reshape(-1, self.tile)
        return tensor

    def revert_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Revert tensor shape after scaling."""
        # TODO(ericd) 2D revert
        if not (self.is_tile and self.shape):
            return tensor
        tensor = tensor.reshape(self.shape)
        self.shape = None
        return tensor.transpose(self.dim, -1)

    def _set_nspec(self, nspec: str):
        if self.scale is None:
            self.scale = NumberSpec(nspec)
        elif self.zero is None:
            self.zero = NumberSpec(nspec)
        else:
            raise ValueError(f"ScaleSpec: more than two NumberSpecs provided in string code '{self.name}'.")

    def _valid_tile(self, tile: int):
        return tile in (0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)

    def _set_tile(self, tval: str, sval: str, oval: str, dval: str):
        tile = int(tval)
        if not self._valid_tile(tile):
            raise ValueError(f"ScaleSpec: '{tile}' is not a supported tile size.")
        # subtile, if any, must be > 1 and divide evenly into tile, which must not be an entire channel
        subtile = int(sval[1:]) if sval else None
        if subtile is not None:
            if tile == 0:
                raise ValueError(f"ScaleSpec: subtile '{subtile}' specified for channel scale in '{self.name}'")
            if subtile < 2 or float(tile // subtile) != tile / subtile:
                raise ValueError(f"ScaleSpec: tile '{tile}' must be a multiple of subtile '{subtile}' in '{self.name}'")
            self.is_subtile = True
        # offset is only specified once, and only then with a subtile
        offset = int(oval[1:]) if oval else None
        if offset is not None:
            if offset not in [1, 2]:
                raise ValueError(f"ScaleSpec: offset {oval} must be 1 or 2 in code '{self.name}'")
            if subtile is None:
                raise ValueError(f"ScaleSpec: offset '{offset}' specified without subtile in '{self.name}'")
            if self.offset is not None and self.offset != offset:
                raise ValueError(f"ScaleSpec: offset specified more than once in code '{self.name}'")
            self.offset = offset
            self.is_offset = True
        # a second dim must be different from the first
        dim = int(dval[1:]) if dval else -1
        if self.dim == dim:
            raise ValueError(f"ScaleSpec: dim '{dim}' specified more than once in code '{self.name}'")
        if self.tile is None:
            self.tile, self.subtile, self.offset, self.dim = tile, subtile, offset, dim
        else:
            self.tile2, self.subtile2, self.dim2, self.is_2d = tile, subtile, dim, True

    def _decode(self, code: str) -> None:
        """Sets fields based on input string code."""
        self.name = code = code.lower()
        for segment in code.split("_"):
            if NumberSpec.valid(segment):
                self._set_nspec(segment)
            elif m := re.fullmatch(r"t(\d+)(s\d+)?(o\d+)?(d\d+)?", segment):
                self._set_tile(*m.group(1, 2, 3, 4))
            else:
                raise ValueError(f"ScaleSpec: '{segment}' is neither a valid number or tile specification.")
        if self.tile is None:
            self.is_tensor = True
        else:
            self.is_tile = self.tile != 0 or self.tile2 != 0
            self.is_channel = not self.is_tile

    def _check(self):
        prefix = f"ScaleSpec: '{self.name}'"
        if not self.scale:
            raise ValueError(f"{prefix} does not specify a scale.")
        if self.zero:
            if not self.scale.is_float:
                raise ValueError(f"{prefix} asymmetric scaling requires a float scale")
            if self.zero.is_exponent or self.zero.is_uint:
                raise ValueError(f"{prefix} asymmetric scaling requires a float or int zero point type")
        if self.is_subtile and (self.zero or not self.scale.is_exponent):
            raise ValueError(f"{prefix} subtiles require an exponent scale")

    @classmethod
    def valid(cls, code: str) -> bool:
        """Checks validity without raising an exception."""
        try:
            cls(code)
            return True
        except (ValueError, NotImplementedError):
            return False
