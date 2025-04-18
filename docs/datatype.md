<!-- markdownlint-disable MD033 MD041 -->

# Datatypes

> TODO(ericd) *this needs proofreading*

A datatype is specified by a [DataType](../tcast/datatype.py), which comprises a NumberSpec and an
optional ScaleSpec.  Normally this is created by passing a number spec and a scale spec, except when
the latter is omitted because it is an unscaled datatype.  Both [NumberSpec](./number.md)
([source](../tcast/number.py)) and [ScaleSpec](./scale.md) ([source](../tcast/number.py)) can be
specified by a string, and those strings can be substituted for the class instances when creating a
`DataType` instance.

The DataType also has a name, which is defined through the name parameter.  This is used to save
the datatype to a registry, so that a datatype does not need to be created twice.  The name can
aso be used by itself to create a datatype, if the number spec and scale specs can be inferred.

## Predefined DataTypes

PyTorch has dtypes in the torch namespace; TensorCast has predefined dtypes in the tcast namespace.

These are standard datatypes that are expected to be commonly used.  Unscaled dtypes include the standard
torch floating point types, including float8 types if supported in your Pytorch installation.  Also included
are unscaled versions of the MXFP types: `e3m2fnuz`, `e2m3fnuz`, and `e2m1fnuz`.

Tensor scaled types include uint16, int16, uint8, and int8 as well as the MXFP 8 and 6-bit numberspecs. The naming
convention for the dtype names is the numberspec and a scale indicator, encoded as "f" for float16, "b" for bfloat16,
"e" for exponent scales. The uint types have two such indicators, the second being for the zero point, and the zero
point has an "i" to indicate an int8 zero point number spec instead of the disallowed "e".  Floating point dtypes all
have the "e" designation.

Tile scaled predefined types are the MXFP and MXINT types: mxfp8e5, mxfp8e4, mxfp6e3, mxfp6e2, mxfp4e2, mxint8, and
mxint4, all of which have a tile size of 32.  Also included is `bfp16`, which is like mxint8 but with a block size
of 8.  Other tile scaled dtypes are the uint8 and uint4 variants of ff, bb, fi, bi with tile size 32 and int8/int4
with float16 and bfloat16 scales.

---

[Documentation](./README.md)
</br>
[Testing](../tests/README.md)
</br>
[Examples](../examples/README.md)
</br>
[TensorCast Home](../README.md)
