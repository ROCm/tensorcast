"""TensorCast: Conversion and compression of arbitrary datatypes."""
# tcast/lookup.py: tools for making make lookup table NumberSpecs

import math

from .datatype import DataType
from .number import NumberSpec
from .scale import ScaleSpec
from .utils import is_power_of_2


class LookupBuilder:
    """Create lookup table datatypes."""

    def __init__(self, cspec: str | NumberSpec, bits: int = 4):
        self.cspec = cspec if isinstance(cspec, NumberSpec) else NumberSpec(cspec)
        self.bits = bits
        self.base_mappings = {}
        self.base_scales = {}
        self.make_base_scales()
        self.make_base_mappings()

    @property
    def emax(self):
        """Get the compute emax to use for scaling."""
        return self.cspec.emax - int(self.cspec.infnan == "fn")

    def scale_emax(self, mapping: list[int], from_emax: int = 0):
        """Scale to the compute datatype."""
        return [v * 2 ** (self.emax - from_emax) for v in mapping]

    def mirror(self, mapping: list[int]):
        """Add zero and mirror to get the full mapping."""
        if mapping[0] != 0:
            mapping.insert(0, 0.0)
        return [-v for v in reversed(mapping)] + mapping

    def make_lookup_mapping(
        self, mapping: str | NumberSpec | list[float], index_shift: int = 0, power_shift: int = 0, mirror: bool = False
    ) -> list[float]:
        """Create a mapping from a number spec or a list of floats, with optional modifications."""
        if isinstance(mapping, str):
            mapping = NumberSpec(mapping)
        if isinstance(mapping, NumberSpec):
            vals = mapping.get_number_line()
            # scale values to compute spec
            vals = [v * 2 ** (self.emax - mapping.emax) for v in vals]
        else:
            vals = mapping  # these values need to be valid in cspec
        if mirror:
            vals = [-v for v in reversed(vals)] + vals
        if power_shift:
            vals = [v / 2**power_shift for v in vals]
        if index_shift:
            # index shifts down when positive
            indices = [
                i - index_shift if v > 0.0 else i + index_shift if v < 0.0 else i
                for v, i in zip(vals, self.cspec.indices_from_vals(vals), strict=True)
            ]
            vals = self.cspec.vals_from_indices(indices)
        return vals

    def make_base_scales(self):
        """Create the scales we will use to build datatypes."""
        for s in [
            "e8m0_t32",
            "e8m0_t32s8",
            "e8m0_t32s8o1",
            "e8m0_t32s8o2",
            "e8m0_t32s4",
            "e8m0_t32s4o1",
            "e8m0_t32s4o2",
            # "e8m0_t32d0_t32",
            # "e8m0_t32s8d0_t32s8",
            # "e8m0_t32s8o2d0_t32s8",
            # "e8m0_t32s4d0_t32s4",
            # "e8m0_t32s4o2d0_t32s4",
        ]:
            ssplit = s.replace("e8m0_t", "e").split("_")
            sname = ssplit[0] if len(ssplit) == 1 else f"{ssplit[0]}_2D"
            self.base_scales[sname] = ScaleSpec(s)

    def make_base_mappings(self):
        """Create some building block mappings.  By 'some', I mean 256.  Or 112."""
        prog4 = self.scale_emax(self.mirror([0.1015625, 0.1875, 0.3125, 0.46875, 0.6875, 0.875, 1.0]))
        out2 = self.scale_emax(self.mirror([0.04296875, 0.0625, 0.09375000, 0.125, 0.1875, 0.25, 1.0]))
        out3 = self.scale_emax(self.mirror([0.02343750, 0.03125000, 0.04296875, 0.0625, 0.09375000, 0.125, 1.0]))
        ei = self.scale_emax(self.mirror([0.2343750, 0.34375, 0.46875, 0.6875, 0.9375, 1.375, 1.875]))
        ef = self.scale_emax(self.mirror([0.6875, 0.8125, 0.9375, 1.125, 1.375, 1.625, 1.875]))
        self.base_mappings["ei7"] = self.make_lookup_mapping(ei, index_shift=0)
        self.base_mappings["ei6"] = self.make_lookup_mapping(ei, index_shift=1)
        self.base_mappings["ei7s1"] = self.make_lookup_mapping(ei, index_shift=0, power_shift=1)
        self.base_mappings["ei6s1"] = self.make_lookup_mapping(ei, index_shift=1, power_shift=1)
        self.base_mappings["ei7s2"] = self.make_lookup_mapping(ei, index_shift=0, power_shift=2)
        self.base_mappings["ei6s2"] = self.make_lookup_mapping(ei, index_shift=1, power_shift=2)
        self.base_mappings["ei7s3"] = self.make_lookup_mapping(ei, index_shift=0, power_shift=3)
        self.base_mappings["ei6s3"] = self.make_lookup_mapping(ei, index_shift=1, power_shift=3)
        self.base_mappings["ei7s4"] = self.make_lookup_mapping(ei, index_shift=0, power_shift=4)
        self.base_mappings["ei6s4"] = self.make_lookup_mapping(ei, index_shift=1, power_shift=4)
        self.base_mappings["ei7s5"] = self.make_lookup_mapping(ei, index_shift=0, power_shift=5)
        self.base_mappings["ei6s5"] = self.make_lookup_mapping(ei, index_shift=1, power_shift=5)
        self.base_mappings["ei7s6"] = self.make_lookup_mapping(ei, index_shift=0, power_shift=6)
        self.base_mappings["ei6s6"] = self.make_lookup_mapping(ei, index_shift=1, power_shift=6)
        self.base_mappings["ef7"] = self.make_lookup_mapping(ef, index_shift=0)
        self.base_mappings["ef6"] = self.make_lookup_mapping(ef, index_shift=1)
        self.base_mappings["ef5"] = self.make_lookup_mapping(ef, index_shift=2)
        self.base_mappings["ef4"] = self.make_lookup_mapping(ef, index_shift=3)
        self.base_mappings["ef7s1"] = self.make_lookup_mapping(ef, index_shift=0, power_shift=1)
        self.base_mappings["ef6s1"] = self.make_lookup_mapping(ef, index_shift=1, power_shift=1)
        self.base_mappings["ef5s1"] = self.make_lookup_mapping(ef, index_shift=2, power_shift=1)
        self.base_mappings["ef4s1"] = self.make_lookup_mapping(ef, index_shift=3, power_shift=1)
        self.base_mappings["ef7s2"] = self.make_lookup_mapping(ef, index_shift=0, power_shift=2)
        self.base_mappings["ef6s2"] = self.make_lookup_mapping(ef, index_shift=1, power_shift=2)
        self.base_mappings["ef5s2"] = self.make_lookup_mapping(ef, index_shift=2, power_shift=2)
        self.base_mappings["ef4s2"] = self.make_lookup_mapping(ef, index_shift=3, power_shift=2)
        self.base_mappings["ef7s3"] = self.make_lookup_mapping(ef, index_shift=0, power_shift=3)
        self.base_mappings["ef6s3"] = self.make_lookup_mapping(ef, index_shift=1, power_shift=3)
        self.base_mappings["ef5s3"] = self.make_lookup_mapping(ef, index_shift=2, power_shift=3)
        self.base_mappings["ef4s3"] = self.make_lookup_mapping(ef, index_shift=3, power_shift=3)
        self.base_mappings["ef7s4"] = self.make_lookup_mapping(ef, index_shift=0, power_shift=4)
        self.base_mappings["ef6s4"] = self.make_lookup_mapping(ef, index_shift=1, power_shift=4)
        self.base_mappings["ef5s4"] = self.make_lookup_mapping(ef, index_shift=2, power_shift=4)
        self.base_mappings["ef4s4"] = self.make_lookup_mapping(ef, index_shift=3, power_shift=4)
        self.base_mappings["ef7s5"] = self.make_lookup_mapping(ef, index_shift=0, power_shift=5)
        self.base_mappings["ef6s5"] = self.make_lookup_mapping(ef, index_shift=1, power_shift=5)
        self.base_mappings["ef5s5"] = self.make_lookup_mapping(ef, index_shift=2, power_shift=5)
        self.base_mappings["ef4s5"] = self.make_lookup_mapping(ef, index_shift=3, power_shift=5)
        self.base_mappings["ef7s6"] = self.make_lookup_mapping(ef, index_shift=0, power_shift=6)
        self.base_mappings["ef6s6"] = self.make_lookup_mapping(ef, index_shift=1, power_shift=6)
        self.base_mappings["ef5s6"] = self.make_lookup_mapping(ef, index_shift=2, power_shift=6)
        self.base_mappings["ef4s6"] = self.make_lookup_mapping(ef, index_shift=3, power_shift=6)
        for i in range(8):
            self.base_mappings[f"f{i}"] = self.make_lookup_mapping("e2m1fnuz", index_shift=4 - i)
            self.base_mappings[f"i{i}"] = self.make_lookup_mapping("e1m2fnuz", index_shift=6 - i)
            self.base_mappings[f"p{i}"] = self.make_lookup_mapping(prog4, index_shift=-i)
            self.base_mappings[f"o2{i}"] = self.make_lookup_mapping(out2, index_shift=-i)
            self.base_mappings[f"o3{i}"] = self.make_lookup_mapping(out3, index_shift=-i)
            for j in range(1, 5):
                self.base_mappings[f"f{i}s{j}"] = self.make_lookup_mapping("e2m1fnuz", power_shift=j)
                self.base_mappings[f"i{i}s{j}"] = self.make_lookup_mapping("e1m2fnuz", power_shift=j)
                self.base_mappings[f"p{i}s{j}"] = self.make_lookup_mapping(prog4, power_shift=j)

    def make_lspec(self, base_names: str, uname: str = None) -> NumberSpec:
        """Create an lspec from a hyphen-separated string of base mapping names."""
        names = base_names.split("-")
        if not is_power_of_2(len(names)):
            raise ValueError(f"make_lspec: mapping name list has length {len(names)}; need a power of 2.")
        if not uname:
            uname = base_names.replace("-", "")
        lprefix = f"l{self.bits}{int(math.log2(len(names)))}"
        if not uname.startswith(lprefix):
            uname = lprefix + uname
        lspec = NumberSpec(f"{uname}_{self.cspec.name}")
        for n in names:
            mapping = self.base_mappings.get(n, None)
            if mapping is None:
                raise ValueError(f"mapping '{n}' is not a base mapping.")
            lspec.add_mapping(mapping, n)
        return lspec

    def make_datatype(self, lspec: NumberSpec, sspec: str = None) -> DataType | list[DataType]:
        """Create a dtype from lookup tables described by existing numpecs."""
        dtypes = []
        for s in [sspec] if sspec else list(self.base_scales.keys()):
            assert isinstance(s, str)
            if s in self.base_scales:
                sspec, sname = self.base_scales[sspec], s
            else:
                ssplit = s.replace("e8m0_t", "e").split("_")
                sname = ssplit[0] if len(ssplit) == 1 else f"{ssplit[0]}_2D"
                sspec = ScaleSpec(s)
            dtypes.append(DataType(lspec, sspec, lspec.name + "_" + sname))
        return dtypes if len(dtypes) > 1 else dtypes[0]
