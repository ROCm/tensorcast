#!/usr/bin/env python
# tcast/modules.py: quant versions of modules
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import torch

import tcast


class Linear(torch.nn.Linear):
    """Linear subclass that supports custom accumulation and datatype conversion."""
    def __init__(self, in_features, out_features, bias=True, tcast_specs=None, pre_weights=None, pre_bias=None):
        super().__init__(in_features, out_features, bias=bias)

        self.specs = tcast_specs

        if pre_weights is not None:
            with torch.no_grad():
                self.weight = pre_weights
        if pre_bias is not None:
            with torch.no_grad():
                self.bias = pre_bias

        if "weight_dtype" in self.specs:
            with torch.no_grad():
                self.weight = torch.nn.parameter.Parameter(tcast.cast(self.weight, dtype=self.specs["weight_dtype"]))

        if "bias_dtype" in self.specs:
            with torch.no_grad():
                self.bias = torch.nn.parameter.Parameter(tcast.cast(self.bias, dtype=self.specs["bias_dtype"]))

    def forward(self, inputs):
        if "input_dtype" in self.specs:
            inputs = tcast.cast(inputs, dtype=self.specs["input_dtype"])

        if "custom_accumulation" in self.specs:
            # the following could be modified by a method.
            outputs = torch.nn.functional.linear(inputs, self.weight, bias=self.bias)
        else:
            outputs = super().forward(inputs)

        if "output_dtype" in self.specs:
            outputs = tcast.cast(outputs, dtype=self.specs["output_dtype"])

        return outputs


class Conv2d(torch.nn.Conv2d):
    """Conv2d subclass that supports custom accumulation and datatype conversion."""
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        tcast_specs=None,
        pre_weights=None,
        pre_bias=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        self.specs = tcast_specs

        if pre_weights is not None:
            with torch.no_grad():
                self.weight = pre_weights
        if pre_bias is not None:
            with torch.no_grad():
                self.bias = pre_bias

        if "weight_dtype" in self.specs:
            with torch.no_grad():
                self.weight = torch.nn.parameter.Parameter(tcast.cast(self.weight, dtype=self.specs["weight_dtype"]))

        if "bias_dtype" in self.specs:
            with torch.no_grad():
                self.bias = torch.nn.parameter.Parameter(tcast.cast(self.bias, dtype=self.specs["bias_dtype"]))

    def forward(self, inputs):
        if "input_dtype" in self.specs:
            inputs = tcast.cast(inputs, dtype=self.specs["input_dtype"])

        if "custom_accumulation" in self.specs:
            # the following could be modified by a method.
            # return super()._conv_forward(inputs, self.weight, self.bias)
            outputs = torch.nn.functional.conv2d(
                inputs,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dialation=self.dilation,
                groups=self.groups,
            )
        else:
            outputs = super().forward(inputs)

        if "output_dtype" in self.specs:
            outputs = tcast.cast(outputs, dtype=self.specs["output_dtype"])

        return outputs
