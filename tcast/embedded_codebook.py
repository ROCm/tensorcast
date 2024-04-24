"""TensorCast: Conversion and compression of arbitrary datatypes."""
# cast/embedded_codebook.py: 2 bit indices into 4 compute values embedded in the tile metadata

import torch

from .cast import Cast
from .datatype import DataType


@torch.inference_mode()
def find_codebook0(x: torch.Tensor, cdtype: DataType, importance: torch.Tensor = None, n_iterations: int = 10):
    """Find all embedded codebooks for the given tensor."""
    # Initialize three centroids to max, min, and zero values
    cx = Cast.cast(x, cdtype).tensor
    xmin, xmax = cx.aminmax(dim=-1, keepdim=True)
    # initialize the fourth centroid to the midpoint between zero and the further of min and max
    # xmid = torch.where(xmax.abs() > xmin.abs(), xmax / 2.0, xmin / 2.0)
    xneg = torch.where(cx < 0, cx.abs(), 0).sum(-1, keepdim=True)
    xpos = torch.where(cx > 0, cx, 0).sum(-1, keepdim=True)
    xmid = torch.where(xneg > xpos, xmin / 2.0, xmax / 2.0)
    centroids = torch.cat((torch.zeros_like(xmax), xmin, xmid, xmax), dim=1)
    for iteration in range(n_iterations):
        # Assign each weight to the nearest centroid
        assignments = (cx.unsqueeze(-1) - centroids.unsqueeze(1)).square().argmin(-1)
        qx = torch.gather(centroids, 1, assignments)
        # Update each centroid to be the weighted mean of the weights assigned to it
        for n in range(cx.shape[0]):
            for i in range(1, 4):
                assigned = cx[n, assignments[n] == i]
                if assigned.numel() > 0:
                    centroids[n, i] = assigned.mean(dim=-1)
        centroids = Cast.cast(centroids, cdtype).tensor
        qx = Cast._vcast(qx, cdtype).tensor
        loss = torch.nn.functional.mse_loss(qx, x)
        print(f"iter {iteration}: loss {loss:8.5f}")
    return centroids

@torch.inference_mode()
def find_codebook(x: torch.Tensor, cdtype: DataType, importance: torch.Tensor = None, n_iterations: int = 10):
    """Find all embedded codebooks for the given tensor."""
    # Initialize three centroids to max, min, and zero values
    cx = Cast.cast(x, cdtype).tensor
    xmin, xmax = cx.aminmax(dim=-1, keepdim=True)
    xmidmin, xmidmax = xmin / 2, xmax / 2
    # initialize the fourth centroid to the midpoint between zero and the further of min and max
    # xmid = torch.where(xmax.abs() > xmin.abs(), xmax / 2.0, xmin / 2.0)
    # xneg = torch.where(cx < 0, cx.abs(), 0).sum(-1, keepdim=True)
    # xpos = torch.where(cx > 0, cx, 0).sum(-1, keepdim=True)
    # xmid = torch.where(xneg > xpos, xmin / 2.0, xmax / 2.0)
    # centroids = torch.cat((torch.zeros_like(xmax), xmin, xmid, xmax), dim=1)
    centroids = torch.cat((xmin, xmidmin, xmidmax, xmax), dim=1)
    for iteration in range(n_iterations):
        # Assign each weight to the nearest centroid
        assignments = (cx.unsqueeze(-1) - centroids.unsqueeze(1)).square().argmin(-1)
        qx = torch.gather(centroids, 1, assignments)
        # Update each centroid to be the weighted mean of the weights assigned to it
        for n in range(cx.shape[0]):
            for i in range(4):
                assigned = cx[n, assignments[n] == i]
                if assigned.numel() > 0:
                    centroids[n, i] = assigned.mean(dim=-1)
        centroids = Cast.cast(centroids, cdtype).tensor
        qx = Cast._vcast(qx, cdtype).tensor
        loss = torch.nn.functional.mse_loss(qx, x)
        print(f"iter {iteration}: loss {loss:8.5f}")
    return centroids
