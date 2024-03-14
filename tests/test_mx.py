import pytest
import tcast
import torch
from .utils import compare_2
try:
    from mx.mx_ops import quantize_mx_op
    from mx.elemwise_ops import quantize_elemwise_op
    from mx.specs import MxSpecs
    MX_AVAILABLE = True

except ImportError:
    MX_AVAILABLE = False

@pytest.mark.parametrize("datatype", ['float16', 'bfloat16'])
@pytest.mark.skipif(not MX_AVAILABLE, reason="MX library is not available. github.com/microsoft/microxcaling")

def test_torch_datatypes(datatype):
    tensor = torch.randn(1000, 1000).float()
    mx_specs = MxSpecs()
    if datatype == 'float16':
        mx_specs['fp'] = 16
    elif datatype == 'bfloat16':
        mx_specs['bfloat'] = 16
    mx_specs['round'] = 'even'
    tensor_mx = quantize_elemwise_op(tensor, mx_specs)
    if 'fp' in datatype:
        datatype = datatype[4:]
    tcast_dt = tcast.datatype(datatype)
    tensor_tcast = tcast.cast(tensor, dtype=tcast_dt)
    compare_2(tensor_mx, tensor_tcast)