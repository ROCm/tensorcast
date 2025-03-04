<!-- markdownlint-disable MD033 MD041 -->

# Examples

A collection of examples on how to use Tensorcast.

## Manual method

User has the most flexibility to quantize each tensor. [linear_bfp16](linear_bfp16.py) shows the usage of **tcast.
cast** class to quantize a linear layer's weights directly and its input and output using torch hooks.

## Replacing torch modules with customized modules

User can replace torch modules with customized modules with required hooks to quantize layer's weights, input,
and output.
[linear_bfp16](linear_bfp16.py) shows how to use the **tcast.TorchInjector** for a single **torch.linear** layer
and [model_custom](model_custom.py) shows an example for a torchvision model. In both examples, user can pass a
dictionary, **tcast_specs**, to describe the data types.

## Replacing model layers with customized layers

User can replace a model's layers with customized layers with required hooks to quantize weights, input, and output.
[model_inject](model_inject.py) shows how to use the **tcast.MixedPrecisionInjector**. In this example, user can
pass a dictionary, **tcast_specs**, to describe the data types for each layer's weoghts, input, and output.
