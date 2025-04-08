#!/usr/bin/env python
# tcast/incoherence.py: incoherence processing utilities
# SPDX-License-Identifier: MIT

"""TensorCast: Specification, conversion and compression of arbitrary datatypes."""

import torch

from .common import STD_DTYPES
from .datatype import DataType


class ICP:
    """Static Incoherence processing class."""

    ICP_MATRICES = {}

    @classmethod
    def randomize_imatrix(cls, imatrix: torch.Tensor) -> torch.Tensor:
        """Randomize a Walsh-Hadamard matrix while preserving orthogonality."""
        diag = torch.diag(torch.randint(0, 2, (imatrix.shape[0],), dtype=imatrix.dtype, device=imatrix.device) * 2 - 1)
        return diag @ imatrix

    @classmethod
    def create_imatrix(
        cls,
        size: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device = "cuda",
        walsh: bool = True,
        randomize: bool = True,
    ) -> torch.Tensor:
        def sign_changes(matrix):
            """Count the number of changes in sign across a row."""
            return [sum(int(matrix[j, i] != matrix[j, i + 1]) for i in range(size - 1)) for j in range(size)]
        # create the Hadamard matrix
        # the Hadamard matrix is defined recursively as H(2n) = H(n) ⊗ [[1, 1], [1, -1]]
        # where ⊗ is the Kronecker product
        imatrix = torch.tensor([[1, 1], [1, -1]], dtype=dtype).cuda()
        device = imatrix.device
        while imatrix.size(0) < size:
            imatrix = torch.kron(imatrix, torch.tensor([[1, 1], [1, -1]], dtype=dtype, device=device))
        # scale the Hadamard matrix
        imatrix /= torch.tensor(size, dtype=dtype, device=device).sqrt()
        if walsh:
            # convert to Walsh-Hadamard matrix, ordering the rows ascending by the number of sign changes per row
            changes = sign_changes(imatrix)
            order = torch.tensor(changes, dtype=dtype, device=device).argsort()
            imatrix = imatrix[order, :]
        if randomize:
            imatrix = cls.randomize_imatrix(imatrix)
        return imatrix.requires_grad_(False)

    @classmethod
    def get_imatrix(
        cls,
        size: int,
        torch_dtype: str | torch.dtype = torch.float32,
        device: torch.device = None,
        walsh: bool = True,
        randomize: bool = False,
    ) -> torch.Tensor:
        """Get, create, or randomize the incoherence matrix for the given size and dtype."""

        def get_key(size: int, torch_dtype: str | torch.dtype, walsh: bool, randomize: bool) -> str:
            key = f"{str(torch_dtype)[6:]}_{size}_"
            if randomize:
                key += "R"
            if walsh:
                key += "W"
            key += "H"
            return key

        if not isinstance(torch_dtype, torch.dtype):
            torch_dtype = DataType(name=torch_dtype).torch_dtype
        if torch_dtype not in STD_DTYPES:
            raise ValueError(f"Invalid imatrix dtype (must be float32, float16, or bfloat16): {torch_dtype}")
        if not any(size == 2**x for x in range(2, 8)):
            raise ValueError(f"Invalid size for incoherence matrix: {size} (must power of 2 in [8, 128]")
        key = get_key(size, torch_dtype, walsh, randomize)
        if randomize:
            if key in cls.ICP_MATRICES:
                # re-randomize but don't replace (might have implications on testing assumptions)
                return cls.randomize_imatrix(cls.ICP_MATRICES[key])
            nr_key = get_key(size, torch_dtype, walsh, False)
            if nr_key in cls.ICP_MATRICES:
                # there is a non-randomized version in the cache so randomize it and store it under the new key
                cls.ICP_MATRICES[key] = cls.randomize_imatrix(cls.ICP_MATRICES[nr_key])
            else:
                # it doesn't, so create a new matrix
                cls.ICP_MATRICES[key] = cls.create_imatrix(size, torch_dtype, device, walsh=walsh, randomize=randomize)
        # non-randomized case: create one if it doesn't exist
        elif key not in cls.ICP_MATRICES:
            cls.ICP_MATRICES[key] = cls.create_imatrix(size, torch_dtype, device, walsh=walsh, randomize=randomize)
        return cls.ICP_MATRICES[key]
