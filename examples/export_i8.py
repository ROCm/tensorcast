import tcast
import torch
from tests.utils import compare_2, tensor_to_bfp
from struct import pack, unpack
import numpy as np

def interleave_bfp16(mantissas, exponents, block):
    num_blocks = int(np.ceil(mantissas.numel()/block))
    array_size = num_blocks*(block+1)
    i8_array = np.zeros(array_size, dtype=np.int8)
    indx = 0
    for b in range(num_blocks):
        e = int(exponents[b*block])
        i8_array[indx] = np.array(e).astype(np.int8)
        indx += 1
        for v in range(block):
            m = int(mantissas[b*block+v])
            i8_array[indx] = m
            indx += 1
    return i8_array

if __name__ == "__main__":
    block = 8
    tensor = (torch.randint(-2048, 2048, (1, block))*torch.randn(1, block)).float()
    tcast_dt = tcast.datatype("int8", "e8m0_t"+str(block), export=True)
    tensor_tcast_d = tcast.cast(tensor, dtype=tcast_dt, roundmode="even")
    tensor_tcast_m = tensor_tcast_d["x_export"].view(-1)
    tensor_tcast_e = tensor_tcast_d["meta_export"].view(-1)
    i8_array = interleave_bfp16(tensor_tcast_m, tensor_tcast_e, block)
    print(tensor_tcast_d["x"])
    print("values: ", end="")
    exp = np.array(i8_array[0]).astype(np.uint8)
    for i in range(1, i8_array.size):
        print(i8_array[i]*(2**(exp-127)), end=", ")
    print("\n")
