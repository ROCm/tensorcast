#!/usr/bin/env python
# examples/linear_bfp16.py: example using bfp16 for Linear layers
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import copy

import torch

import tcast


# Manual method using hooks
def manual_inject(in_fp32, layer_fp32, tcast_specs):
    layer_q = copy.deepcopy(layer_fp32)

    if "weight_dtype" in tcast_specs:
        with torch.no_grad():
            layer_q.weight = torch.nn.parameter.Parameter(tcast.cast(layer_fp32.weight, dtype=tcast_specs["weight_dtype"]))

    if "input_dtype" in tcast_specs:

        def forward_input_hook(module, input):
            input[0].copy_(tcast.cast(input[0], dtype=tcast_specs["input_dtype"]))

        layer_q.register_forward_pre_hook(forward_input_hook)

    if "output_dtype" in tcast_specs:

        def forward_output_hook(module, input, output):
            output.copy_(tcast.cast(output, dtype=tcast_specs["output_dtype"]))

        layer_q.register_forward_hook(forward_output_hook)

    return layer_q(in_fp32)


if __name__ == "__main__":
    bfp16ebs8_t = tcast.DataType("int8", "e8m0_t8", "bfp16ebs8_t")
    layer_fp32 = torch.nn.Linear(64, 64)
    input_fp32 = torch.randn(64, 64)

    output_fp32 = layer_fp32(input_fp32)

    # Manual Method
    tcast_specs = {}
    output_bfp16_1 = manual_inject(input_fp32, layer_fp32, tcast_specs)
    print(f"l2 norm error none: {torch.norm(output_fp32 - output_bfp16_1)}")

    tcast_specs = {"weight_dtype": bfp16ebs8_t}
    output_bfp16_1 = manual_inject(input_fp32, layer_fp32, tcast_specs)
    print(f"l2 norm error weights-only: {torch.norm(output_fp32 - output_bfp16_1)}")

    tcast_specs = {"weight_dtype": bfp16ebs8_t, "input_dtype": bfp16ebs8_t}
    output_bfp16_1 = manual_inject(input_fp32, layer_fp32, tcast_specs)
    print(f"l2 norm error weights and input: {torch.norm(output_fp32 - output_bfp16_1)}")

    tcast_specs = {"weight_dtype": bfp16ebs8_t, "input_dtype": bfp16ebs8_t, "output_dtype": bfp16ebs8_t}
    output_bfp16_1 = manual_inject(input_fp32, layer_fp32, tcast_specs)
    print(f"l2 norm error weight, input, and output: {torch.norm(output_fp32 - output_bfp16_1)}")

    # Modify the pytorch modules
    tcast.torch_injector(tcast_specs)
    layer_fp32_2 = torch.nn.Linear(64, 64)
    # same weights and biases
    with torch.no_grad():
        layer_fp32_2.weight = torch.nn.parameter.Parameter(tcast.cast(layer_fp32.weight, dtype=bfp16ebs8_t))
        layer_fp32_2.bias = layer_fp32.bias

    output_bfp16_2 = layer_fp32_2(input_fp32)
    print(f"Method2\nl2 norm error weight, input, and output: {torch.norm(output_fp32 - output_bfp16_2)}")
