import pytest
import tcast
import torch
from tests.utils import compare_2


@pytest.mark.parametrize("datatype", ['float16', 'bfloat16','float8_e5m2', 'float8_e5m2fnuz', 'float8_e4m3fn', 'float8_e4m3fnuz'])

def test_torch_datatypes(datatype):
    tensor = torch.randn(1024, 1024).float()
    tensor_torch = tensor.to(getattr(torch, datatype))
    tensor_torch = tensor_torch.float()
    if 'float8_' in datatype:
        if not tcast.utils.is_float8_available():
            pytest.skip("Skipping because float8 is not available")
        datatype = datatype[7:]
        if 'fnuz' in datatype: # TODO: is there a better way to do this?
            e = int(datatype[1])
            bias = str(2**(e-1))
            datatype = datatype[:-4]+'b'+bias+'fnuz'
    tcast_dt = tcast.datatype(datatype)
    tensor_tcast = tcast.cast(tensor, dtype=tcast_dt, roundmode="even")
    compare_2(tensor_torch, tensor_tcast)

