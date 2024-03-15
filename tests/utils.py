import torch


def compare_2(tensor1, tensor2):
        if torch.allclose(tensor1, tensor2):
            return
        else:
            print("tensor1: ", tensor1)
            print("tensor2: ", tensor2)
            raise ValueError