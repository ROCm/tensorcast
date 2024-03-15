import pytest
import tcast
import torch
from tests.utils import compare_2
try:
    from mx.mx_ops import quantize_mx_op
    from mx.elemwise_ops import quantize_elemwise_op
    from mx.specs import MxSpecs
    MX_AVAILABLE = True

except ImportError:
    MX_AVAILABLE = False

@pytest.mark.parametrize("datatype", ['float16', 'bfloat16'])
@pytest.mark.parametrize("roundmode", ["even", "nearest"])
@pytest.mark.skipif(not MX_AVAILABLE, reason="MX library is not available. github.com/microsoft/microxcaling")

def test_mx_unscaled_datatypes(datatype, roundmode):
    tensor = torch.randn(1024, 1024).float()
    mx_specs = MxSpecs()
    if datatype == 'float16':
        mx_specs['fp'] = 16
    elif datatype == 'bfloat16':
        mx_specs['bfloat'] = 16
    mx_specs['round'] = roundmode
    tensor_mx = quantize_elemwise_op(tensor, mx_specs)
    if 'fp' in datatype:
        datatype = datatype[4:]
    tcast_dt = tcast.datatype(datatype)
    tensor_tcast = tcast.cast(tensor, dtype=tcast_dt, roundmode=roundmode)
    compare_2(tensor_mx, tensor_tcast)

@pytest.mark.parametrize("datatype", ['int8', 'int4', 'fp8_e5m2', 'fp8_e4m3', 'fp6_e3m2', 'fp6_e2m3', 'fp4_e2m1'])
@pytest.mark.parametrize("roundmode", ["even", "nearest"])
@pytest.mark.skipif(not MX_AVAILABLE, reason="MX library is not available. github.com/microsoft/microxcaling")

def test_mx_scaled_datatypes(datatype, roundmode):
    tensor = torch.randn(1024, 1024).float()
    mx_specs = MxSpecs()
    mx_specs['block_size'] = 32
    mx_specs['round'] = roundmode
    tensor_mx = quantize_mx_op(tensor, mx_specs, elem_format=datatype, axes=-1, round=roundmode)
    if 'fp' in datatype:
        datatype = datatype[4:]
    if 'e4' in datatype:
        datatype += 'fn'
    elif 'e3' in datatype or 'e2' in datatype:
        datatype += 'fnuz'
    tcast_dt = tcast.datatype(datatype, "e8m0_t32")
    tensor_tcast = tcast.cast(tensor, dtype=tcast_dt, roundmode=roundmode, scalemode="max")
    compare_2(tensor_mx, tensor_tcast)

if __name__ == "__main__":
    test_mx_scaled_datatypes('fp4_e2m1')