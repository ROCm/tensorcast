#!/usr/bin/env python
# examples/export_i8.py: example of exporting a tensor to interleaved BFP16 format
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import numpy as np
import torch

import tcast


def interleave_bfp16(mantissas, exponents, block):
    num_blocks = tcast.cdiv(mantissas.size(0), block)
    array_size = num_blocks * (block + 1)
    i8_array = np.zeros(array_size, dtype=np.int8)
    indx = 0
    for b in range(num_blocks):
        e = int(exponents[b * block])
        i8_array[indx] = np.array(e).astype(np.int8)
        indx += 1
        for v in range(block):
            m = int(mantissas[b * block + v])
            i8_array[indx] = m
            indx += 1
    return i8_array


if __name__ == "__main__":
    blockmap = {8: tcast.bfp16, 16: tcast.bfp16b16}
    block, blocktype = 16, blockmap[16]
    tensor = (torch.randint(-2048, 2048, (1, block)) * torch.randn(1, block)).float()
    tensor_tcast_d = tcast.cast(tensor, dtype=blocktype, roundmode="even")
    tensor_tcast_m = tensor_tcast_d["x_export"].view(-1)
    tensor_tcast_e = tensor_tcast_d["meta_export"].view(-1)
    i8_array = interleave_bfp16(tensor_tcast_m, tensor_tcast_e, block)
    print(tensor_tcast_d["x"])
    print("values: ", end="")
    exp = np.array(i8_array[0]).astype(np.uint8)
    for i in range(1, i8_array.size):
        print(i8_array[i] * (2 ** (exp - 127)), end=", ")
    print("\n")
