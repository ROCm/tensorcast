import pytest
import tcast
import torch
from tests.utils import compare_2, tensor_to_bfp

@pytest.mark.parametrize("datatype", ['bfp16', 'bfp15', 'bfp14'])
@pytest.mark.parametrize("roundmode", ["even", "nearest"])
@pytest.mark.parametrize("block_size", ["8", "16"])

def test_bfp(datatype, roundmode, block_size):
    tensor = torch.randn(16, 1024).float()
    p1 = "int"+str(int(datatype[3:])-8)
    p2 = "e8m0_t"+block_size
    tcast_dt = tcast.datatype(p1, p2)
    tensor_bfp = tensor_to_bfp(tensor, 1, tcast_dt, roundmode)
    tensor_tcast = tcast.cast(tensor, dtype=tcast_dt, roundmode=roundmode)
    compare_2(tensor_bfp, tensor_tcast)

