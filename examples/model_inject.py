#!/usr/bin/env python
# examples/model_inject.py: example of injecting mixed precision into a model
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import torch

import tcast


class NonTcastModel(torch.nn.Module):
    """A simple model with a convolutional layer and two fully connected layers."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3)
        self.fc1 = torch.nn.Linear(16, 32)
        self.fc2 = torch.nn.Linear(32, 8)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = NonTcastModel()
    input_fp32 = torch.randn(3, 8, 18)
    output_fp32 = model(input_fp32)
    bfp16ebs8_t = tcast.DataType("int8", "e8m0_t8", "bfp16ebs8_t")
    bfp16ebs16_t = tcast.DataType("int8", "e8m0_t16", "bfp16ebs16_t")
    tcast_specs = {
        "fc1": {"weight_dtype": bfp16ebs8_t},
        "fc2": {"weight_dtype": bfp16ebs16_t},
        "conv": {"weight_dtype": bfp16ebs8_t},
    }
    model_mixed = tcast.MixedPrecisionInjector(model, tcast_specs)
    output_bfp16 = model_mixed(input_fp32)
    print("l2 norm error: ", torch.norm(output_fp32 - output_bfp16).item())
