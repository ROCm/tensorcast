"""TensorCast: Conversion and compression of arbitrary datatypes."""
# tcast/cast.py: casting methods

from collections.abc import Callable
from typing import ClassVar, Literal

import torch

from .datatype import DataType
from .extension import Extension
from .number import NumberSpec
from .utils import check_literal

RoundMode = Literal["even", "away", "zero", "stochastic", "nearest"]  # "nearest" is an alias for "away"
ScaleMode = Literal["max", "midmax", "auto"] # "auto" is an alias for "midmax"
ComputeMode = Literal["cpu", "gpu", "torch"]


class Cast:
    """Static class with implementations in PyTorch."""

    roundmode: ClassVar[RoundMode] = "even"  # the current rounding mode
    scalemode: ClassVar[ScaleMode] = "max"  # the current scale selection mode
    compmode: ClassVar[ComputeMode] = "torch"  # use PyTorch operators for both CPU and GPU
    extension: ClassVar[Extension] = None

    @classmethod
    def _round(cls, x: torch.Tensor) -> torch.Tensor:
        if cls.roundmode == "stochastic":
            return torch.sign(x) * torch.trunc(torch.abs(x) + torch.rand_like(x))
        if cls.roundmode == "even":
            return torch.round(x)
        if cls.roundmode in ("away", "nearest"):
            # return torch.trunc(x + x.sign() * 0.5) torch thinks 0.4999999701976776123046875 + 0.5 is 1.0
            return torch.where(x.abs().frac() == 0.5, torch.trunc(x + x.sign() * 0.5), x.round())
        return torch.where(x.abs().frac() == 0.5, torch.trunc(x), x.round())

    @classmethod
    def _run_extension(cls, x: torch.Tensor, dtype: DataType, op: str, **kwargs) -> torch.Tensor | None:
        """Run an extension operator if it exists and we need it."""
        if cls.compmode != "torch":
            if cls.extension is None:
                cls.extension = Extension()
            if cls.extension.has_operation(op, cls.compmode):
                return cls.extension.exec_operation(x, dtype, op, cls.compmode, **kwargs)
        return None

    @classmethod
    def _safe_frexp(cls, x: torch.Tensor) -> torch.Tensor:
        return x.float().add(torch.finfo(torch.float32).eps * (x == 0.0)).frexp().exponent

    @classmethod
    def _cast_unscaled(cls, x: torch.Tensor, nspec: NumberSpec) -> torch.Tensor:
        assert x.is_floating_point and nspec.is_float
        valexp = (cls._safe_frexp(x) - 1).clamp_min(nspec.emin)
        rscale = (nspec.mbits - valexp).exp2()
        x = cls._round(x * rscale).div(rscale).clamp(-nspec.maxfloat, nspec.maxfloat)
        return x

    @classmethod
    def _get_scales(cls, x: torch.Tensor, dtype: DataType, noreshape: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Find and return the scale factors."""
        assert dtype and dtype.sspec
        if ret := cls._run_extension(x, dtype, "get_scales") is not None:
            return ret if isinstance(ret, tuple) else (ret, None)
        dim = None if dtype.sspec.is_tensor else -1
        x = x.clone()
        if not (dtype.sspec.is_tensor or noreshape):
            x = dtype.sspec.reshape_tensor(x)
        zero: torch.Tensor = None
        if dtype.nspec.is_uint:
            assert dtype.sspec.scale.is_float
            tmin, tmax = torch.aminmax(x, dim=dim, keepdim=True)
            tmin, tmax = tmin.clamp_max(0.0), tmax.clamp_min(0.0)
            scale = cls._cast_unscaled((tmax - tmin) / dtype.nspec.maxint, dtype.sspec.scale)
            zero = cls._cast_unscaled(-tmin, dtype.sspec.zero) if dtype.sspec.zero.is_float else cls._round(-tmin / scale)
        elif dtype.sspec.scale.is_float:
            scale = (
                cls._cast_unscaled(1.0 / x.abs().amax(dim=dim, keepdim=True), dtype.sspec.scale)
                if dtype.nspec.is_float
                else cls._cast_unscaled(x.abs().amax(dim=dim, keepdim=True) / dtype.nspec.maxint, dtype.sspec.scale)
            )
        else:
            maxexp = cls._safe_frexp(x).amax(dim=dim, keepdim=True) - 1 - dtype.nspec.emax
            if dtype.sspec.subtile:
                # reshape with subtile as last dim so we can do subtiled lookups or offsets or both
                if not noreshape:
                    x = dtype.sspec.reshape_tensor(dtype.sspec.revert_tensor(x), True)
                submaxexp = cls._safe_frexp(x).amax(dim=dim, keepdim=True) - 1 - dtype.nspec.emax
                # this looks like a kudge, but for non-offset lookups, we either need to expand the scale to the right shape
                # or we can do this and simple not allow the scale to change across the subtiles in a tile
                offset = 2**dtype.sspec.offset - 1 if dtype.sspec.is_offset else 0
                maxexp = submaxexp.clamp_min(maxexp - offset)
            elif dtype.nspec.ebits > 1 and cls.scalemode == "midmax":
                maxexp[(x * (-maxexp).exp2()).abs().amax(dim=dim) > dtype.nspec.midmax] += 1
            if dtype.sspec.scale.is_exponent:
                # e8m0 is actual exponent + scale bias - nspec emax
                maxexp += dtype.sspec.scale.bias
            scale = maxexp
        return scale, zero

    @classmethod
    def _apply_scales(
        cls,
        x: torch.Tensor,
        dtype: DataType,
        scale: torch.Tensor,
        zero: torch.Tensor = None,
        lookup: int = None,
        noreshape: bool = False,
    ) -> torch.Tensor:
        """Given scales, return vcast tensor."""
        assert dtype and not dtype.is_unscaled
        if not dtype.is_tensor:
            if ret := cls._run_extension(x, dtype, "apply_scales", lookup=lookup) is not None:
                return ret
            if not noreshape:
                x = dtype.sspec.reshape_tensor(x, dtype.sspec.is_subtile)
        eps = torch.finfo(x.dtype).eps
        if hasattr(dtype.nspec, "is_lookup") and dtype.nspec.is_lookup:
            assert dtype.is_tile and lookup is not None and dtype.sspec.scale.is_exponent
            scale = (scale - dtype.sspec.scale.bias).exp2()
            tmap = dtype.nspec.get_mapping(lookup, torch_dtype=x.dtype, device=x.device)
            tmid = dtype.nspec.get_midpoints(lookup, torch_dtype=x.dtype, device=x.device)
            t = x.div(scale)
            out = torch.zeros_like(x)
            out[t <= tmid[0]] = tmap[0]
            for i in range(tmap.numel() - 1):
                if tmid[i] < 0:
                    out[t > tmid[i]] = tmap[i + 1]
                else:
                    out[t >= tmid[i]] = tmap[i + 1]
            x = out.clamp(min=tmap[0], max=tmap[-1]) * scale
        elif dtype.nspec.is_uint:
            assert not dtype.sspec.is_subtile
            if dtype.sspec.zero.is_float:
                x = scale * cls._round((x + zero) / scale.clamp_min(eps)).clamp(0, dtype.nspec.maxint) - zero
            else:
                x = scale * ((cls._round(x / scale.clamp_min(eps)) + zero).clamp(0, dtype.nspec.maxint) - zero)
        elif dtype.sspec.scale.is_float:
            x = cls._round(x / scale.clamp_min(eps)).clamp(dtype.nspec.minint, dtype.nspec.maxint) * scale
        else:
            assert dtype.sspec.scale.is_exponent
            scale = (scale - dtype.sspec.scale.bias).exp2()
            x /= scale  # scale x into range of the target dtype
            valexp = (cls._safe_frexp(x) - 1).clamp_min(dtype.nspec.emin)  # get the independent exponents, clipped to emin
            rscale = (dtype.nspec.mbits - valexp).exp2()
            x = cls._round(x * rscale).div(rscale).clamp(-dtype.nspec.maxfloat, dtype.nspec.maxfloat) * scale
        if not (dtype.is_tensor or noreshape):
            x = dtype.sspec.revert_tensor(x)
        return x

    @classmethod
    def _vcast(cls, x: torch.Tensor, dtype: DataType) -> torch.Tensor:
        """Virtual cast."""
        if ret := cls._run_extension(x, dtype, "vcast"):
            return ret
        xtype, x = x.dtype, x.clone()
        if dtype.is_unscaled:
            return cls._cast_unscaled(x, dtype.nspec).to(xtype)
        scale, zero = cls._get_scales(x, dtype)
        return cls._apply_scales(x, dtype, scale, zero)

    @classmethod
    def _vcast_lookup(cls, x: torch.Tensor, dtype: DataType, better: Callable = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Cast a tensor to multiple datatypes based on tile/subtile content."""

        def _better(q1: torch.Tensor, q2: torch.Tensor, f: torch.Tensor):
            return torch.linalg.vector_norm(q1 - f, dim=-1) < torch.linalg.vector_norm(q2 - f, dim=-1)

        assert dtype.is_tile and dtype.nspec.is_lookup and dtype.sspec.scale.is_exponent
        for i in range(dtype.nspec.num_mappings):
            if i == 0:
                scale, _ = cls._get_scales(x, dtype)
                x = dtype.sspec.reshape_tensor(x, dtype.sspec.is_subtile)
                choices = torch.zeros(x.shape[0], dtype=torch.int8, device=x.device)
                q = cls._apply_scales(x, dtype, scale, None, lookup=i, noreshape=True)
                out = q.clone()
            else:
                if not dtype.sspec.subtile:
                    scale, _ = cls._get_scales(x, dtype, noreshape=True)
                q = cls._apply_scales(x, dtype, scale, None, lookup=i, noreshape=True)
                choice = better(q, out, x) if better else _better(q, out, x)
                choices[choice] = i
                out[choice] = q[choice]
        out = dtype.sspec.revert_tensor(out)
        return out, choices

    @classmethod
    def sparse(cls, tensor: torch.Tensor, stile: int, dense: int, dim: int = -1) -> torch.Tensor:
        """Simple structured sparsity, M of N, where M is dense values retained out of N."""
        if tensor.shape[-1] % stile != 0:
            raise NotImplementedError(
                f"Last tensor dim ({tensor.shape[-1]}) must be evenly divisible by sparse tile size ({stile})"
            )
        assert dense > 0 and dense < stile
        tshape = tensor.shape
        t = tensor.clone().transpose(dim, -1).reshape(-1, stile).abs()
        idx = t.argsort(dim=-1, descending=True)
        premask = torch.full(t.shape, True, dtype=torch.bool, device=tensor.device)
        mask = torch.empty_like(premask)
        premask[..., :dense] = False
        mask.scatter_(-1, idx, premask)
        return t.masked_fill_(mask, 0.0).reshape(tshape).transpose(dim, -1)

    @classmethod
    def cast(
        cls,
        x: torch.Tensor,
        dtype: DataType,
        roundmode: RoundMode = None,
        scalemode: ScaleMode = None,
        compmode: ComputeMode = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Generic cast interface."""
        # currently not so generic as we are only doing virtual cast
        # roundmode and scalemode are optional overrides
        check_literal(roundmode, RoundMode, True)
        check_literal(scalemode, ScaleMode, True)
        check_literal(compmode, ComputeMode, True)
        saveround, savescale, savecomp = cls.roundmode, cls.scalemode, cls.compmode
        cls.roundmode = roundmode if roundmode else saveround
        cls.scalemode = scalemode if scalemode else savescale
        cls.compmode = compmode if compmode else savecomp
        if dtype.is_lookup:
            x, choices = cls._vcast_lookup(x, dtype)
            # for i in range(dtype.nspec.num_mappings):
            #     share, name = (choices == i).sum() / choices.numel(), dtype.nspec.mapnames[i]
            #     print(f"{name}: {share * 100.0:6.3f}%")
        else:
            x = cls._vcast(x, dtype)
        cls.roundmode = saveround
        cls.scalemode = savescale
        cls.compmode = savecomp
        return (x, choices) if dtype.is_lookup else x
