#!/usr/bin/env python
# tcast/test_bfp.py: block floating point
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import pytest
import torch

import tcast

from .utils import compare_2, tensor_to_bfp


@pytest.mark.parametrize("datatype", ["bfp16", "bfp15", "bfp14", "bfp13"])
@pytest.mark.parametrize("roundmode", ["even", "away"])
@pytest.mark.parametrize("block_size", ["8", "16", "32"])
def test_bfp(datatype, roundmode, block_size):
    tensor = torch.randn(16, 1024).float()
    p1 = "int" + str(int(datatype[3:]) - 8)
    p2 = "e8m0_t" + block_size
    tcast_dt = tcast.datatype(p1, p2)
    tensor_bfp = tensor_to_bfp(tensor, 1, tcast_dt, roundmode)
    tensor_tcast = tcast.cast(tensor, dtype=tcast_dt, castmode="virtual", roundmode=roundmode)
    compare_2(tensor_bfp, tensor_tcast)
