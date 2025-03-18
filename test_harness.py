#!/usr/bin/env python
# tensorcast/test_harness.py: test entry point external to tcast package
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import tcast
import tests.base_tests_misc as M
import tests.base_tests_v2 as V2
import tests.utils as U

logger = tcast.get_logger()


def run_base_misc():
    tcount = 0
    errors, methods = 0, tcast.LPConfig.methods()
    num_methods = len(methods)
    for i in range(num_methods - 1):
        for j in range(i + 1, num_methods):
            errors += int(not M.test_lpconfig_base(tcast.get_logger(), i, j))
            tcount += 1
    assert errors == 0, f"LPConfig test failed with {errors} errors"


def run_base_v2_unscaled():
    tcount = 0
    for dtype in tcast.DataType.gather_registered(lambda x: x.is_unscaled and x.nspec.bits <= 16):
        for torch_dtype in U.TORCH_DTYPES_32_16:
            for shape in U.SHAPES_234D:
                for roundmode in ["even"]:
                    for computemode in ["torch"]:
                        V2.test_torchcast_virtual_unscaled_base(dtype, torch_dtype, shape, roundmode, computemode)
                        tcount += 1


def run_base_v2_unscaled_torch():
    tcount = 0
    for dtype in tcast.DataType.gather_registered(lambda x: x.nspec.torch_dtype is not None and x.is_unscaled):
        for torch_dtype in U.TORCH_DTYPES_32_16_8:
            for shape in U.SHAPES_234D:
                for roundmode in ["even"]:
                    for computemode in ["torch"]:
                        V2.test_torchcast_virtual_unscaled_torch_base(dtype, torch_dtype, shape, roundmode, computemode)
                        tcount += 1

# for every unscaled dtype that has a torch dtype
# for every 32 or 16 bit torch dtype
# create a tensor of the torch dtype
# cast that tensor to the nspec dtype via tcast
# cast that tensor to to the converted nspec torch_dtype using ".to"

if __name__ == "__main__":
    tcast.set_seed(19)
    run_base_v2_unscaled()
    run_base_v2_unscaled_torch()
    run_base_misc()
