import torch


def compare_2(tensor1, tensor2):
        if torch.allclose(tensor1, tensor2):
            return
        else:
            raise ValueError