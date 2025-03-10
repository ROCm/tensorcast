#!/usr/bin/env python
# tcast/test_bfp_export.py: block floating point
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import pytest
import torch

import tcast

from .utils import compare_2, tensor_to_bfp


@pytest.mark.parametrize("datatype", ["bfp16", "bfp15", "bfp14", "bfp13"])
@pytest.mark.parametrize("roundmode", ["even", "nearest"])
@pytest.mark.parametrize("block_size", ["8", "16", "32"])
def test_bfp(datatype, roundmode, block_size):
    tensor = (torch.randint(-2048, 2048, (16, 1024)) * torch.randn(16, 1024)).float()
    tcast_dt = tcast.datatype(f"int{int(datatype[3:]) - 8}", f"e8m0_t{block_size}")
    tensor_bfp = tensor_to_bfp(tensor, 1, tcast_dt, roundmode)
    tensor_tcast_d = tcast.cast(tensor, dtype=tcast_dt, roundmode=roundmode)
    tensor_tcast = tensor_tcast_d["x"]
    compare_2(tensor_bfp, tensor_tcast)
