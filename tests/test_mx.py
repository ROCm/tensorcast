#!/usr/bin/env python
# tests/test_mx.py: test quantization against MX library
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import pytest

import tcast
import tests.base_tests_mx as B

from .utils import MX_AVAILABLE, dependency_assert

logger = tcast.get_logger("tcast_unittest")

dependency_assert("mx")


@pytest.mark.parametrize("shape", [(1024, 1024), (8, 128, 64)])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("roundmode", ["even", "away"])
@pytest.mark.parametrize("computemode", ["torch", "triton"])
def test_mx_unscaled_datatypes(shape: tuple[int], dtype: str | tcast.DataType, roundmode: str, computemode: str):
    B.test_mx_mxfp_datatypes_base(shape, dtype, roundmode, computemode)


@pytest.mark.parametrize("shape", [(1024, 1024), (8, 128, 64)])
@pytest.mark.parametrize("dtype", ["mxfp8e5", "mxfp8e4", "mxfp6e3", "mxfp6e2", "mxfp4e2"])
@pytest.mark.parametrize("roundmode", ["even", "away"])
@pytest.mark.parametrize("computemode", ["torch", "triton"])
@pytest.mark.skipif(not MX_AVAILABLE, reason="MX library is not available. github.com/microsoft/microxcaling")
def test_mx_mxfp_datatypes(shape: tuple[int], dtype: str | tcast.DataType, roundmode: str, computemode: str):
    B.test_mx_mxfp_datatypes_base(shape, dtype, roundmode, computemode)
