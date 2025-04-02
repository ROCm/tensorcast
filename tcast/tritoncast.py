#!/usr/bin/env python
# tcast/tritoncast.py: triton methods for casting
# SPDX-License-Identifier: MIT
# ruff: noqa: D103

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import torch
import triton
import triton.language as tl

from . import snippets as lp
from .common import FP8_DTYPES, STD_DTYPES, CastMode, Modes, RoundMode, ScaleMode
from .tensor import Tensor
from .utils import is_triton_available


# fmt: on
class TritonCast:
    """Triton: accelerated methods for casting."""

    @staticmethod
    def supports(tensor: Tensor) -> bool:
        """Check if the cast operation is supported by in the triton code."""
        if not is_triton_available():
            return False
        if Modes.cast == CastMode.UPCAST:
            return False  # TODO(ericd) implement upcast, then change this
        else:
            return (
                Modes.cast != CastMode.COMPRESS
                and tensor.input.dim() == 2
                and tensor.input.dtype in STD_DTYPES
                and tensor.dtype.nspec.torch_dtype in FP8_DTYPES
                and not tensor.needs_pad()
                and tensor.dtype.is_square
            )

    @staticmethod
    def cast(tensor: Tensor, transpose_scale: bool = False) -> bool:
        """Cast data to the specified dtype."""

        def get_tensor_info(tensor: Tensor, name: str):
            if (t := getattr(tensor, name, None)) is not None:
                return (t, t.size(0), t.size(1), t.stride(0), t.stride(1))
            else:
                return (None, 0, 0, 0, 0)

        # nspec, sspec = tensor.dtype.nspec, tensor.dtype.sspec
        # t = tensor.input
        # input_info = (t, t.size(0), t.size(1), t.stride(0), t.stride(1))
        # output_info = get_tensor_info(tensor, "tensor")
        # scale_info = get_tensor_info(tensor, "scale")
        # # zero_info = get_tensor_info(tensor, "zero")
        # # meta_info = get_tensor_info(tensor, "meta")
        # mask_info = get_tensor_info(tensor, "mask")
        # nspec_info = (nspec.emax, nspec.emin, nspec.mbits, nspec.finfo.maxfloat, nspec.finfo.midmax)
        # M, N = input_info[1:3]
        # sspec_info = sspec.get_tile_info(M, N)[2:]
        # if not nspec.is_codebook:
        #     sspec_info = (sspec_info[0], sspec_info[1], sspec_info[4], sspec_info[5]) # no subtiles needed

        # grid = (tl.cdiv(M, sspec_info[0]) * tl.cdiv(N, sspec_info[1]),)
        # if sspec.scale.is_exponent:
        #     exponent_cast_kernel[grid](
        #         *input_info, *output_info, *scale_info, *mask_info, Modes.round, Modes.scale, Modes.cast,
        #         *nspec_info, nspec.bits, sspec.scale.bits, *sspec_info,
        #     )
        # else:
        #     raise ValueError("Unsupported cast operation.")
        tensor.postcast(transpose_scale)
        # fmt: on

    @staticmethod
    def upcast(tensor: Tensor, torch_dtype: torch.dtype = None):
        """Upcast data to the specified dtype."""
        # TODO(ericd): must implement
        if torch_dtype is None:
            torch_dtype = tensor.original_dtype
        if tensor.original_dtype.is_floating() != torch_dtype.is_floating():
            raise ValueError("Upcast requires the same dtype family (both float or both int.")
        if not tensor.quantized:
            raise ValueError("Upcast requires a quantized tensor.")
        raise NotImplementedError("Upcast not implemented for TritonCast.")
