import pytest
import tcast
import torch
from tests.utils import compare_2, tensor_to_bfp
from struct import pack, unpack
import numpy as np

@pytest.mark.parametrize("datatype", ['bfp16', 'bfp15', 'bfp14', 'bfp13'])
@pytest.mark.parametrize("roundmode", ["even", "nearest"])
@pytest.mark.parametrize("block_size", ["8", "16", "32"])

def test_bfp(datatype, roundmode, block_size):
    tensor = (torch.randint(-2048, 2048, (16, 1024))*torch.randn(16, 1024)).float()
    p1 = "int"+str(int(datatype[3:])-8)
    p2 = "e8m0_t"+block_size
    tcast_dt = tcast.datatype(p1, p2, export=True)
    tensor_bfp = tensor_to_bfp(tensor, 1, tcast_dt, roundmode)
    tensor_tcast_d = tcast.cast(tensor, dtype=tcast_dt, roundmode=roundmode)
    tensor_tcast = tensor_tcast_d["x"]
    compare_2(tensor_bfp, tensor_tcast)


tensor = torch.randn(16, 1024).float()
p1 = "int8"
p2 = "e8m0_t8"
tcast_dt = tcast.datatype(p1, p2, export=True)
tensor_bfp = tensor_to_bfp(tensor, 1, tcast_dt, "even")
tensor_tcast_d = tcast.cast(tensor, dtype=tcast_dt, roundmode="even")
tensor_tcast = tensor_tcast_d["x"]
compare_2(tensor_bfp, tensor_tcast)
x_int = tensor_tcast_d["x_export"]
x_po2 = tensor_tcast_d["meta_export"]
print(f"x.shape: {tensor_tcast.shape}, x_int.shape: {x_int.shape}, x_po2.shape: {x_po2.shape}")

# number of private bits = tcast_dt.nspec.mbits+2
for _i in range(1):
    private = x_int[_i].numpy().astype(np.int8) # this is int
    shared = x_po2[_i,0].numpy().astype(np.uint8) # this is int
    print(f"private: {private}, shared: {shared}")
    print(f"private: {pack('=8b', *private).hex()}, shared: {pack('=B', shared).hex()}")

