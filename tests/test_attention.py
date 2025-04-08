#!/usr/bin/env python
# tests/test_attention.py: test all things required by FA FP8 project (duplicate tests included)
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import pytest

import tests.base_attention as A

# TODO(ericd): add tests for:
# scale_and_quantize
# needs_icp (PyTorch i/f vs Triton i/f)
# quant_dtype
# scale_dtype
# descale
# code match (attn.code == attn.to_code())


# fmt: off
@pytest.mark.parametrize("index1, index2", [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
        (2, 3), (2, 4), (2, 5), (2, 6), (2, 7),
        (3, 4), (3, 5), (3, 6), (3, 7),
        (4, 5), (4, 6), (4, 7),
        (5, 6), (5, 7),
        (6, 7)
    ]
)
def test_attention_configuration(index1, index2):
    A.base_attention_configuration(index1, index2)
#fmt on
