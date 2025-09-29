import torch
import tcast
import copy

class Linear(torch.nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        tcast_specs=None,
        pre_weights=None,
        pre_bias=None,
    ):
        super().__init__(
            in_features,
            out_features,
            bias=bias)

        self.specs = tcast_specs
        self.weight = pre_weights
        self.bias = pre_bias

        if 'weight_dtype' in self.specs:
            self.weight.data = tcast.cast(self.weight, dtype=self.specs['weight_dtype']).tensor

        if 'bias_dtype' in self.specs:
            self.bias.data = tcast.cast(self.bias, dtype=self.specs['bias_dtype']).tensor

    def forward(self, inputs):
        if 'input_dtype' in self.specs:
            inputs = tcast.cast(inputs, dtype=self.specs['input_dtype']).tensor

        if 'custom_accumulation' in self.specs:
            # the following could be modified by a method.
            outputs = torch.nn.functional.linear(
                inputs,
                self.weight,
                bias=self.bias,
            )
        else:
            outputs = super().forward(inputs)

        if 'output_dtype' in self.specs:
            outputs = tcast.cast(outputs, dtype=self.specs['output_dtype']).tensor

        return outputs


class Conv2d(torch.nn.Conv2d):
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
        padding_mode='zeros',
        tcast_specs=None,
        pre_weights=None,
        pre_bias=None,
    ):

        super(Conv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        )

        self.specs = tcast_specs

        self.weight = pre_weights
        self.bias = pre_bias

        if 'weight_dtype' in self.specs:
            self.weight.data = tcast.cast(self.weight, dtype=self.specs['weight_dtype']).tensor

        if 'bias_dtype' in self.specs:
            self.bias.data = tcast.cast(self.bias, dtype=self.specs['bias_dtype']).tensor

    def forward(self, inputs):

        if 'input_dtype' in self.specs:
            inputs = tcast.cast(inputs, dtype=self.specs['input_dtype']).tensor.detach().clone()

        if 'custom_accumulation' in self.specs:
            # the following could be modified by a method.
            #return super()._conv_forward(inputs, self.weight, self.bias)
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

        if 'output_dtype' in self.specs:
            outputs = tcast.cast(outputs, dtype=self.specs['output_dtype']).tensor.detach().clone()

        return outputs
