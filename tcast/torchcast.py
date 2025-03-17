#!/usr/bin/env python
# tcast/torchcast.py: casting methods using torch
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

from collections.abc import Callable

import torch

from .common import CastMode, ComputeMode, Modes, RoundMode, ScaleMode
from .number import NumberSpec
from .tensor import Tensor
from .utils import get_logger

logger = get_logger("tcast")


class TorchCast:
    """Static class with implementations in PyTorch."""

    @staticmethod
    def round(x: torch.Tensor) -> torch.Tensor:
        if Modes.round == RoundMode.STOCHASTIC:
            return torch.sign(x) * torch.trunc(torch.abs(x) + torch.rand_like(x))
        if Modes.round == RoundMode.EVEN:
            return torch.round(x)
        if Modes.round == RoundMode.AWAY:
            return torch.trunc(x + x.sign() * 0.5)
        return torch.where(x.abs().frac() == 0.5, torch.trunc(x), x.round()).to(x.dtype)

    @staticmethod
    def get_exponents(x: torch.Tensor) -> torch.Tensor:
        """torch.frexp is broken."""
        return (x.abs() + torch.finfo(x.dtype).smallest_normal).log2().floor()

    @staticmethod
    def _cast_unscaled(x: torch.Tensor, nspec: NumberSpec) -> torch.Tensor:
        assert x.is_floating_point and nspec.is_float
        valexp = TorchCast.get_exponents(x).clamp_min(nspec.emin)
        rscale = (nspec.mbits - valexp).exp2()
        return TorchCast.round(x * rscale).div(rscale).clamp(-nspec.finfo.maxfloat, nspec.finfo.maxfloat).to(x.dtype)

    @staticmethod
    def cast_unscaled(tensor: Tensor):
        x = TorchCast._cast_unscaled(tensor.input, tensor.dtype.nspec)
        tensor.update(output=x)

    @staticmethod
    def select_codebook_meta(tensor: Tensor) -> None:
        """Cast a tensor to multiple datatypes based on tile/subtile content."""

        def _better(q1: torch.Tensor, q2: torch.Tensor, f: torch.Tensor):
            return torch.linalg.vector_norm(q1 - f, dim=-1) < torch.linalg.vector_norm(q2 - f, dim=-1)

        # start with everything being mapping zero
        master_meta = torch.zeros_like(tensor.meta)
        for i in range(tensor.dtype.nspec.num_mappings):
            tmp_tensor = Tensor(tensor.tensor, tensor.dtype, tensor.scale, meta=torch.full_like(tensor.meta, i))
            TorchCast.apply_codebook(tmp_tensor)
            if i == 0:
                output = tmp_tensor.tensor.clone()
            else:
                choice = _better(tmp_tensor.tensor, output, tensor.tensor)
                master_meta[choice] = i
                output[choice] = tmp_tensor.tensor[choice]
            del tmp_tensor
        tensor.update(meta=master_meta)

    @staticmethod
    def apply_codebook(tensor: Tensor, better: Callable = None) -> None:
        """Cast to codebooks in tensor.meta.  Create meta if not already selected."""
        assert tensor.dtype.is_codebook
        nspec, sspec = tensor.dtype.nspec, tensor.dtype.sspec
        x = tensor.input
        TorchCast.select_codebook_meta(tensor, better)
        scale = (tensor.scale - sspec.scale.bias).exp2()
        tmap, tmid = nspec.get_codebook(torch_dtype=x.dtype, device=x.device)
        t = x / scale  # scale x into range of the compute dtype, which is where the codebooks are
        out = torch.zeros_like(x)
        if nspec.symmetric:
            t = t.abs()
        tmap = tmap[tensor.meta]
        tmid = tmid[tensor.meta]
        for i in range(tmap.shape[-1]):
            ge = t >= tmid[:, i]
            out[ge] = tmap[ge, i]
        if nspec.symmetric:
            out = out * x.sign()
        tensor.update(output=out * scale)

    @staticmethod
    def apply_scales(tensor: Tensor) -> None:
        """Given scales, return cast tensor."""
        nspec, sspec = tensor.dtype.nspec, tensor.dtype.sspec
        assert tensor.has_scales and not nspec.is_codebook and not sspec.is_subtile
        x, scale, zero = tensor.input, tensor.scale, tensor.zero
        minint, maxint, maxfloat = nspec.finfo.minint, nspec.finfo.maxint, nspec.finfo.maxfloat
        if nspec.is_uint:
            assert zero is not None
            eps = torch.finfo(torch.float32).eps
            if sspec.zero.is_float:
                x = scale * TorchCast.round((x + zero) / scale.clamp_min(eps)).clamp(0, maxint) - zero
            else:
                x = scale * ((TorchCast.round(x / scale.clamp_min(eps)) + zero).clamp(0, maxint) - zero)
        elif sspec.scale.is_float:
            if nspec.is_float:
                x = TorchCast.round(x * scale).clamp(-maxfloat, maxfloat) / scale
            else:
                x = TorchCast.round(x / scale.clamp_min(eps)).clamp(minint, maxint) * scale
        else:
            if sspec.scale.is_exponent:
                scale = (scale - sspec.scale.bias).exp2()
            elif sspec.scale.is_int:
                scale = scale.exp2()
            x /= scale  # scale x into range of the target dtype
            valexp = TorchCast.get_exponents(x).clamp_min(nspec.emin)  # get the independent exponents, clipped to emin
            rscale = (nspec.mbits - valexp).exp2()
            x = TorchCast.round(x * rscale).div(rscale).clamp(-maxfloat, maxfloat) * scale
        tensor.update(output=x)

    @staticmethod
    def apply_sparsity_mask(tensor: Tensor) -> None:
        """Simple structured sparsity, M of N, where M is dense values retained out of N."""
        assert not tensor.has_mask and not tensor.is_compressed and tensor.dtype.is_sparse
        x = tensor.input
        idx = x.abs().argsort(dim=-1, descending=True)
        premask = torch.full(x.shape, True, dtype=torch.bool, device=tensor.device)
        mask = torch.empty_like(premask)
        premask[..., : tensor.dtype.sspec.dense] = False
        mask.scatter_(-1, idx, premask)
        if Modes.cast == CastMode.COMPRESS:
            x = x[not mask]
        else:
            x[mask] = 0.0
        tensor.update(output=x, mask=mask)

    @staticmethod
    def select_scales(tensor: Tensor) -> None:
        """Find the scales for a tensor."""
        nspec, sspec = tensor.dtype.nspec, tensor.dtype.sspec
        assert nspec and sspec
        dim = None if sspec.is_tensor else -1
        x = tensor.input
        zero = None
        if nspec.is_uint:
            assert sspec.scale.is_float and sspec.zero
            tmin, tmax = torch.aminmax(x, dim=dim, keepdim=True)
            tmin, tmax = tmin.clamp_max(0.0), tmax.clamp_min(0.0)
            scale = TorchCast._cast_unscaled((tmax - tmin) / nspec.maxint, sspec.scale)
            zero = TorchCast._cast_unscaled(-tmin, sspec.zero) if sspec.zero.is_float else TorchCast.round(-tmin / scale)
        elif sspec.scale.is_float:
            scale = (
                TorchCast._cast_unscaled(nspec.finfo.maxfloat / x.abs().amax(dim=dim, keepdim=True), sspec.scale)
                if nspec.is_float
                else TorchCast._cast_unscaled(x.abs().amax(dim=dim, keepdim=True) / nspec.finfo.maxint, sspec.scale)
            )
        else:
            if nspec.ebits == 1 or Modes.scale == ScaleMode.FLOOR:
                maxexp = TorchCast.get_exponents(x).amax(dim=dim, keepdim=True)
            else:
                absmax = x.abs().amax(dim=dim, keepdim=True)
                maxexp = TorchCast.get_exponents(absmax)  # this constitutes the FLOOR method
                if Modes.scale == ScaleMode.CEIL:
                    maxexp = (absmax + torch.finfo(x.dtype).eps).log2().ceil().int()
                elif Modes.scale == ScaleMode.OPTION3:
                    # scale absmax into pure mantissa range [0, 2), then up by mbits to RNE
                    ascale = (maxexp - nspec.mbits).exp2()
                    maxexp = TorchCast.get_exponents((absmax / ascale).round() * ascale)
                elif Modes.scale == ScaleMode.MIDMAX:
                    # scale midmax to [0, 2)
                    midmax = nspec.finfo.midmax / (2**nspec.emax)
                    maxexp[absmax / maxexp.exp2() > midmax] += 1
                elif Modes.scale == ScaleMode.TOPBINADE:
                    # scale maxfloat and absmax into pure mantissa range [0, 2)
                    maxfloat = nspec.finfo.maxfloat / (2**nspec.emax)
                    maxexp[(absmax / maxexp.exp2()) > maxfloat] += 1
            if sspec.scale.is_exponent:
                # e8m0 is actual exponent + scale bias - nspec emax
                scale = (maxexp - nspec.emax + sspec.scale.bias).to(torch.uint8)
            else:
                # int scale is the actual unbiased exponent
                scale = maxexp.to(torch.int8)
        assert (zero is None) != nspec.is_uint
        tensor.update(scale=scale, zero=zero)

    @staticmethod
    def supports(tensor: Tensor) -> bool:
        """Check if the cast operation is supported by in the torch code."""
        cast_virtual = Modes.cast == CastMode.VIRTUAL
        compute_torch = Modes.compute == ComputeMode.TORCH
        unscaled = tensor.dtype.is_unscaled
        not_2d = unscaled or not tensor.dtype.sspec.is_2d
        supports_all = cast_virtual and compute_torch and not_2d
        assert supports_all is True, f"{cast_virtual}, {compute_torch}, {not_2d}"
        return supports_all
        # return (
        #     Modes.cast == CastMode.VIRTUAL
        #     and Modes.compute == "torch"
        #     and (tensor.dtype.is_unscaled or not tensor.dtype.sspec.is_2d)
        # )

    @staticmethod
    def cast(tensor: Tensor) -> bool:
        """Cast interface using PyTorch ops."""
        if tensor.dtype.is_unscaled:
            TorchCast.cast_unscaled(tensor)
            return True
        else:
            TorchCast.select_scales(tensor)
            if tensor.dtype.is_sparse:
                TorchCast.apply_sparsity_mask(tensor)
            if tensor.dtype.is_codebook:
                TorchCast.apply_codebook(tensor)
            else:
                TorchCast.apply_scales(tensor)
        return tensor.scale is not None
