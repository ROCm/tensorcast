#!/usr/bin/env python
# tcast/test_bfp_export.py: block floating point
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import pytest
import torch

import tcast

from .utils import compare_2, tensor_to_bfp


@pytest.mark.parametrize("datatype", ["bfp16", "bfp15", "bfp14", "bfp13"])
@pytest.mark.parametrize("roundmode", ["even", "away"])
@pytest.mark.parametrize("tile_size", ["8", "16", "32"])
def test_bfp(datatype: str, roundmode: str, tile_size: int):
    tensor = (torch.randint(-2048, 2048, (16, 1024)) * torch.randn(16, 1024)).float()
    tcast_dt = tcast.datatype(f"int{int(datatype.removeprefix('bfp'))- 8}", f"e8m0_t{tile_size}")
    tensor_bfp = tensor_to_bfp(tensor, 1, tcast_dt, roundmode)
    tensor_tcast = tcast.cast(tensor, dtype=tcast_dt, castmode="virtual", roundmode=roundmode)
    compare_2(tensor_bfp, tensor_tcast)
