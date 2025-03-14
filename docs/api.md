<!-- markdownlint-disable MD033 MD041 -->

# TensorCast Interface

> TODO(ericd): *needs proofreading and code snippets need to be verified*

The `tcast` interface includes consists of:

- initialization to set defaults for modes of rounding, scale selection, compute, and casting
- simple functions to create number specifications, scaling specifications, and datatypes
- casting functions, including cast and upcast
- configuration functions for applying low precision to attention and GEMM kernels
- various utilities
- predefined datatypes

## API Functions

### `initialize`

The initialize function sets the default modes for round, scale, cast, and compute. The code below shows
the defaults used if `initialize` is not called.

```python
import tcast
tcast.initialize(roundmode="even", scalemode="floor", computemode="torch", castmode="actual")
```

### `configuration`

This creates and returns an [LPConfig](./attention) instance that specifies what tensors in attention or
GEMM kernels get quantized to to what datatypes, specifies incoherence processing and block scaling shapes.
This can be done by passing an integer code that completely defines the configuration, a json path
that contains the attribute settings ([example json](../tcast/tests/config_attrs.json)) for `LPConfig`, a string
that is the name of a shortcut containing predefined configurations, or just pass the parameters that
would be used to create `LPConfig`.

This can also be accomplished by setting environment variables ([example bash](../tcast/tests/config_attrs.sh)).

```python
import tcast
# code
cfg = tcast.configuration(0xa929204c)
# or json
cfg = tcast.configuration(Path("my_test_config.json"))
# or shortcut
cfg = tcast.configuration("split_match_e4m3fnuz_e5m2fnuz_icpqk32_32x32")
# or parameters
cfg = tcast.configuration(
    block_size=(32, 32), block_axes=(0, 1), scale_dtype=None,
    q_dtype="float8_e4m3fnuz", k_dtype="float8_e4m3fnuz", v_dtype="float8_e4m3fnuz",
    p_dtype="float8_e5m2fnuz", ds_dtype="float8_e5m2fnuz", do_dtype="float8_e5m2fnuz",
    icp_qk=True, icp_pv=False, icp_fp32=True
)
```

### `number`

This function, given a valid code string, returns a NumberSpec, which can then be used to create a DataType.

```python
import tcast
nspec = tcast.number("e5m6") # fp12, an abbreviated version of fp16
```

### `scale`

This function, given a valid code string, returns a ScaleSpec, which can then be used to create a DataType.

```python
import tcast
# power of 2 scaling on the last dimension with tile size 32
sspec = tcast.scale("e8m0_t32")
```

### `datatype`

This function, given a number spec (NumberSpec or valid numspec code string), an optional scale
(ScaleSpec or valid scale spec code string), and an optional name for the datatype,
returns a DataType, which can be passed to a cast function. If the name is omitted, one is manufactured.

```python
import tcast
nspec, sspec = number("e5m6"), scale("e8m0_t32")
dtype = tcast.datatype(nspec, sspec, name="e5m6_e32")
# or
dtype = tcast.datatype("e5m6", "e8m0_t32", name="e5m6_e32")
```

### `cast`

This is intended to be a universal interface to the TritonCast and TorchCast classes.
The input `torch.Tensor` and `DataType` are required with optional overrides for modes,
and optional transpose_scale with exchanges the two tile descriptors fo that a transposed
quantization can occur without transposing the tensor.

```python
import tcast
x = tcast.cast(
        torch.randn(1024, 1024, device="cuda", dtype=torch.float16),
        tcast.datatype("e5m6", "e8m0_t32", name="e5m6_e32"),
        castmode="actual",
        computemode="triton",
        roundmode="even",
        scalemode="floor",
        transpose_scale=False
    )
```

Many common datatypes are predefined, exhaustively described below, which simplifies the calls:

```python
import tcast
x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
c = tcast.cast(x, tcast.mxfp6e2)
```

### `upcast`

When an *actual* or *compress* mode is used in a cast, a `tcast.Tensor` is returned that includes
a scaled and possibly compressed output that cannot be directly used as a GEMM (or `tl.dot`) input.
This occurs when the storage datatype is not a compute datatype, and the tcast.Tensor need to be
upcast to a compute datatype (scaled or unscaled).  The upcast function takes a `tcast.Tensor`
a `torch.dtype`, and a bool descale as inputs, returning a `torch.Tensor` with the input torch_dtype.

```python
import tcast
x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
c = tcast.cast(x, tcast.mxfp6e2)
x_up = tcast.upcast(c, x.dtype, descale=True)
```

## Predefined Datatypes

Just as Pytorch and Triton have datatypes defined in the `torch` and `triton.languge` namespaces,
so does TensorCast in the `tcast` namespace.  Not only are standard datatypes defined, but so are
variants of those standards, including varied scale types.

Here we have a lot of redundancy, which violates the DRY principle (don't repeat yourself), but may
make it easier to use. There is precendent with PyTorch having torch.float and torch.float32, butby they
seem to be moving away from that.

### Standard Unscaled Datatypes

#### PyTorch

PyTorch is becoming overloaded with dtypes under the hood, such as torch.bits1x8, which may become
very useful at compressing and decompressing data via casting, but they don't seem functional as
of 2.6.  From a casting perspective, unscaled integers don't appear to be that interesting.

- float32 (alias fp32)
- float16 (alias fp16)
- bfloat16 (alias bf16)
- float8_e5m2 (alises bf8, e5m2)
- float8_e5m2fnuz (aliases bf8n, e5m2fnuz)
- float8_e4m3fn (alias fp8, e4m3fn)
- float8_e4m3fnuz (aliases fp8n, e4m3fnuz)

#### IEEE P3109 8-bit unscaled datatypes

Not sure what the status of these is as of now.  They are here just in case anyone ever uses them.

- binary8p1, binary8p2, binary8p3, binary8p4, binary8p5, binary8p6

### Tensor- and Channel-scaled Integers

The notation used to abbreviate scaling uses **F** for float32, **f** for float16, **b** for bfloat16,
**e** for e8m0, **n** for unsigned e4m3, **I** for int16, and **i** for int8.  Signed integers get
one scale letter. Unsigned integers get two letters, the second of which is the zero point.
A third letter (**c**) indicates that the type is channel-scaled in the last dimension.  Not all
combinations are included.  Once we get to scaling, we generally want low precision.  You can
still use 16 bits, but you have to use tcast.datatype("uint16", "float32_int8") or similar.

#### Unsigned

- uint8_FF, uint8_FFc (fp32 scale and zero point)
- uint8_Fi, uint8_Fic (fp32 scale, int8 zero point)
- uint8_ff, uint8_ffc (fp16 scale and zero point)
- uint8_fi, uint8_fic (fp16 scale, int8 zero point)
- uint8_bb, uint8_bbc (bf16 scale and zero point)
- uint8_bi, uint8_bic (bf16 scale, int8 zero point)
- uint8_ni, uint8_bic (e4m3 scale, int8 zero point)
- uint8_ei, uint8_bic (e8m0 scale, int8 zero point)
- uint4_FFc (fp32 scale and zero point)
- uint4_Fic (fp32 scale, int8 zero point)
- uint4_ffc (fp16 scale and zero point)
- uint4_fic (fp16 scale, int8 zero point)
- uint4_bbc (bf16 scale and zero point)
- uint4_bic (bf16 scale, int8 zero point)
- uint4_nic (e4m3 scale, int8 zero point)
- uint4_eic (e8m0 scale, int8 zero point)

#### Integer

- int8_F, int8_Fc (fp32 scale)
- int8_f, int8_fc (fp16 scale)
- int8_b, int8_bc (bf16 scale)
- int8_n, int8_nc (e4m3 scale)
- int8_e, int8_ec (e8m0 scale)
- int4_Fc (fp32 scale)
- int4_fc (fp16 scale)
- iint4_bc (bf16 scale)
- int4_nc (e4m3 scale)
- int4_ec (e8m0 scale)

#### FP8

- bf8_F, bf8_Fc, bf8n_F, bf8n_Fc (fp32 scale bf8 and bf8 nanoo, tensor and channel)
- bf8_f, bf8_fc, bf8n_f, bf8n_fc (fp16 scale bf8 and bf8 nanoo, tensor and channel)
- bf8_b, bf8_bc, bf8n_b, bf8n_bc (bf16 scale bf8 and bf8 nanoo, tensor and channel)
- bf8_n, bf8_nc, bf8n_n, bf8n_nc (e4m3 scale bf8 and bf8 nanoo, tensor and channel)
- bf8_e, bf8_ec, bf8n_e, bf8n_ec (e8m0 scale bf8 and bf8 nanoo, tensor and channel)
- fp8_F, fp8_Fc, fp8n_F, fp8n_Fc (fp32 scale fp8 and fp8 nanoo, tensor and channel)
- fp8_f, fp8_fc, fp8n_f, fp8n_fc (fp16 scale fp8 and fp8 nanoo, tensor and channel)
- fp8_b, fp8_bc, fp8n_b, fp8n_bc (bf16 scale fp8 and fp8 nanoo, tensor and channel)
- fp8_n, fp8_nc, fp8n_n, fp8n_nc (e4m3 scale fp8 and fp8 nanoo, tensor and channel)
- fp8_e, fp8_ec, fp8n_e, fp8n_ec (e8m0 scale fp8 and fp8 nanoo, tensor and channel)

### Tile Scaled

#### OCP MXFP and MXINT, tile size 32, scale e8m0

- mxbf8 (alias mxfp8e5)
- mxfp8 (alias mxfp8e4)
- mxbf6 (alias mxfp6e3)
- mxbfp (alias mxfp6e2)
- mxfp4 (alias mxfp4e2)
- mxint8
- mxint4

#### OCP MXFP and MXINT, tile size 16, scale e8m0

- mxbf8t16, mxfp8t16, mxbf6t16, mxbfpt16, mxfp4t16, mxint8t16, mxint4t16

#### NVF4, tile size 16, scale e4m3

- nvf4 (maybe others to come?)

#### MSFP (old school Microsoft), tile size 16, exponent scale, subtile size 2, implemented as codebooks

- mx9, mx6, mx4

#### BFP (old school Microsoft), tile size 8 or 16, exponent scale

- bfp16, bfp16t16

### Implicit Codebooks

- mxfi4 (alternates MXINT4 and MXFP4 patterns, tile 32, subtile 4, cb41fi_e2m2fnuz)
- mxfp4m (MXFP4 patterns with additional trailing mantissa bit, tile 32, subtile 4, cb41f01_e2m2fnuz)
- mxfp4e (MXFP4 patterns four different starting exponents, tile 32, subtile 8, cb42fe0123_e3m2fnuz)
- mxfp4f4 (mxfp4 shifted 4 times up or down the number line, tile 32, subtile 8, cb42f1346_e2m3fnuz)

---

[Documentation](./README.md)
</br>
[TensorCast Home](../README.md)
