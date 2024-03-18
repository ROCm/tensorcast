"""TensorCast: Conversion and compression of arbitrary datatypes."""
# tcast/scale.py: scaling format specification

from dataclasses import dataclass
import re

import torch

from .number import NumberSpec


@dataclass
class ScaleSpec:
    """Specifies scaling method for a given NumberSpec."""

    name: str = None
    tile: int = None
    dim: int = None
    is_tensor: bool = False
    is_channel: bool = False
    is_tile: bool = False
    scale: NumberSpec = None
    zero: NumberSpec = None
    shape: tuple[int] = None

    def __init__(self, code: str):
        self._decode(code)
        self._check()

    def _valid_tile(self, tile: int):
        return tile in (0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)

    def _set_nspec(self, nspec: str):
        if self.scale is None:
            self.scale = NumberSpec(nspec)
        elif self.zero is None:
            self.zero = NumberSpec(nspec)
        else:
            raise ValueError(f"ScaleSpec: more than two NumberSpecs provided in string code '{self.name}'.")

    def _set_tile(self, tile: int, dim: int = None):
        if not self._valid_tile(tile):
            raise ValueError(f"ScaleSpec: '{tile}' is not a supported tile size.")
        if self.tile is not None:
            raise ValueError(f"ScaleSpec: second tile spec found in '{self.name}'")
        self.tile = tile
        self.dim = -1 if dim is None else dim
        self.is_channel, self.is_tile = self.tile == 0, self.tile != 0

    def _decode(self, code: str) -> None:
        """Sets fields based on input string code."""
        self.name = code = code.lower()
        # The string is one or two NumberSpec codes (scale and optional zero point) and
        # an optional tile spec code, separated by underscores.  The tilespec is tXdY, where X is the tile size
        # and Y is the dimension of the tile. If omitted, this is a tensor scale; if present with X=0, it is a
        # channel scale.  If the dimension is omitted, it defaults to -1, or the last dimension of the tensor.
        for segment in code.split("_"):
            if NumberSpec.valid(segment):
                self._set_nspec(segment)
            elif m := re.fullmatch(r"t(\d+)(d\d+)?", segment):
                if self.tile is not None:
                    raise ValueError(f"ScaleSpec: spurious code segment '{segment}' found after tile spec.")
                self._set_tile(int(m.group(1)), int(m.group(2)[1:]) if m.group(2) else -1)
            else:
                raise ValueError(f"ScaleSpec: '{segment}' is neither a valid number or tile specification.")
        if self.tile is None:
            self.is_tensor = True
        if self.scale is None:
            raise ValueError(f"ScaleSpec: scale spec '{code}' has no valid NumberSpec name for scale.")

    def _check(self):
        # TODO(ericd): additional checks for bad/unsupported combinations of values that parsed correctly
        prefix = f"ScaleSpec: '{self.name}'"
        if not self.scale:
            raise ValueError(f"{prefix} does not specify a scale.")
        if self.zero:
            if not self.scale.is_float:
                raise ValueError(f"{prefix} asymmetric scaling requires a float scale")
            if self.zero.is_exponent or self.zero.is_uint:
                raise ValueError(f"{prefix} asymmetric scaling requires a float or int zero point type")

    def reshape_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape and/or transpose tensor for scaling."""
        assert self.shape is None
        if not self.is_tile:
            return tensor
        tensor = tensor.transpose(self.dim, -1)
        self.shape = tensor.shape
        return tensor.reshape(-1, self.tile)

    def revert_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Revert tensor shape after scaling."""
        if not self.is_tile:
            return tensor
        assert self.shape
        tensor = tensor.reshape(self.shape)
        self.shape = None
        return tensor.transpose(self.dim, -1)

    @classmethod
    def valid(cls, code: str) -> bool:
        """Checks validity without raising an exception."""
        try:
            cls(code)
            return True
        except (ValueError, NotImplementedError):
            return False

