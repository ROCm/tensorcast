#!/usr/bin/env python
# tests/test_config.py: test LP configuration
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import tcast

logger = tcast.get_logger("tcast_unittest")


def test_config(shape: tuple[int], dtype: str | tcast.DataType, computemode: str):
    errors = tcast.config.test_lpconfig(logger)
    assert errors == 0, f"{errors} found in LP configuration"
