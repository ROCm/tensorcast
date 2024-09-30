"""TensorCast: Conversion and compression of arbitrary datatypes."""
# tcast/datatype.py: combines number spec and scale spec to define a datatype

from dataclasses import dataclass

from .number import NumberSpec
from .scale import ScaleSpec


@dataclass
class DataType:
    """Everything needed to define a scaled or unscaled datatype."""

    _name: str = None
    nspec: NumberSpec = None
    sspec: ScaleSpec = None
    is_unscaled: bool = False
    is_tensor: bool = False
    is_channel: bool = False
    is_tile: bool = False
    export: bool = False

    def __init__(self, nspec: str | NumberSpec, sspec: str | ScaleSpec = None, name: str = None, export: bool = False):
        self.nspec = nspec if isinstance(nspec, NumberSpec) else NumberSpec(nspec)
        self._name = name
        self.sspec = sspec if isinstance(sspec, ScaleSpec) else ScaleSpec(sspec) if sspec else None
        self.is_unscaled = self.sspec is None
        self.is_tensor = self.sspec is not None and self.sspec.is_tensor
        self.is_channel = self.sspec is not None and self.sspec.is_channel
        self.is_tile = self.sspec is not None and self.sspec.is_tile
        assert int(self.is_unscaled) + int(self.is_tensor) + int(self.is_channel) + int(self.is_tile) == 1
        self._check()
        self.export = export

    def _check(self):
        prefix = f"DataType: '{self.name}'"
        if self.is_unscaled:
            if not self.nspec.is_float:
                raise ValueError(f"{prefix} only float data can be cast unscaled.")
            return
        if self.nspec.is_exponent:
            raise ValueError(f"{prefix} exponent number spec is only permitted as a scale.")
        if self.sspec.zero and not self.nspec.is_uint:
            raise ValueError(f"{prefix} zero spec in scale is incompatible with float or signed int data spec.")
        if self.nspec.is_uint and not self.sspec.zero:
            raise ValueError(f"{prefix} uint data requires a zero point.")
        if self.nspec.is_float and not self.sspec.scale.is_exponent:
            raise NotImplementedError(f"{prefix} only exponent scaling is supported for float data.")
        if self.nspec.is_int and not (self.sspec.scale.is_exponent or self.sspec.scale.is_float):
            raise ValueError(f"{prefix} int data requires either a float or exponent scale.")

    @property
    def name(self):
        if self._name:
            return self._name
        return str(self)

    def bits_per_value(self) -> float:
        """Given scaling metadata, how many bits per value?"""
        if not self.is_tile:
            return float(self.nspec.bits)
        bits = self.nspec.bits * self.sspec.tile + self.sspec.scale.bits + (self.sspec.zero.bits if self.sspec.zero else 0)
        return bits / self.sspec.tile

    def __str__(self):
        s = self.nspec.name
        if self.sspec:
            s += "_" + self.sspec.name
        return s

    @classmethod
    def valid(cls, ncode: str, scode: str = None) -> bool:
        """Returns True if the code generates a valid datatype."""
        try:
            cls(ncode, scode)
            return True
        except ValueError:
            return False
