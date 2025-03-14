#!/usr/bin/env python
# tests/test_config.py: test LP configuration
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import tcast

logger = tcast.get_logger("tcast_unittest")


def test_config():
    errors = tcast.config.test_config(logger)
    assert errors == 0, f"{errors} found in LP configuration"
