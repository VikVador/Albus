r"""Fréchet distances for comparing the distributions of two sets of samples."""

__all__ = [
    "frechet_distance",
    "frechet_inception_distance",
]

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from functools import cache
from torch import Tensor

from albus.tools import apply_mask, axes


def sqrtm(sigma: Tensor) -> Tensor:
    r"""Compute the square root of symmetric positive semi-definite matrices.

    Formula:
        √Σ = Q √Λ Qᵀ, with Σ = Q Λ Qᵀ

    Arguments:
        sigma : Symmetric positive semi-definite matrices, with shape (*, D, D).

    Returns:
        Square roots of `sigma`, with shape (*, D, D).
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(sigma)
    return eigenvectors @ (eigenvalues.relu().sqrt()[..., None] * eigenvectors.mT)


def frechet_distance(
    x: Tensor,
    y: Tensor,
    dims: str,
    samples: str,
    features: str,
) -> Tensor:
    r"""Compute the Fréchet distance between the Gaussian fits of two sets of samples.

    Formula:
        FD = ‖μₓ - μᵧ‖² + tr(Σₓ + Σᵧ - 2 √(√Σᵧ Σₓ √Σᵧ))

    Arguments:
        x        : First set of samples.
        y        : Second set of samples, with the same shape as `x` except along `samples`.
        dims     : Space-separated axis names describing the shape of `x`, e.g. "T N D".
        samples  : Space-separated subset of `dims` indexing the samples, e.g. "N".
        features : Space-separated subset of `dims` spanning the feature space, e.g. "D".

    Returns:
        Fréchet distance, with the axes in `samples` and `features` collapsed.

    Example:
        >>> x = torch.randn(7, 1000, 64)  # (T, N, D)
        >>> y = torch.randn(7, 500, 64)   # (T, N, D)
        >>> frechet_distance(
        ...     x=x,
        ...     y=y,
        ...     dims="T N D",
        ...     samples="N",
        ...     features="D",
        ... ).shape
        torch.Size([7])
    """
    samples_axes, features_axes = axes(dims, samples), axes(dims, features)
    tail = tuple(range(-len(samples_axes) - len(features_axes), 0))

    # Samples and features are each flattened into one axis, giving (*, N, D)
    def flatten(z: Tensor) -> Tensor:
        z = z.movedim(samples_axes + features_axes, tail).flatten(-len(features_axes))
        return z.flatten(-len(samples_axes) - 1, -2)

    # Matrix square roots are imprecise in single precision
    x, y, dtype = flatten(x).double(), flatten(y).double(), x.dtype
    assert min(x.shape[-2], y.shape[-2]) > 1, "ERROR - Each set needs at least two samples."

    mu_x, mu_y = x.mean(dim=-2), y.mean(dim=-2)
    x_centered, y_centered = x - mu_x[..., None, :], y - mu_y[..., None, :]
    sigma_x = x_centered.mT @ x_centered / (x.shape[-2] - 1)
    sigma_y = y_centered.mT @ y_centered / (y.shape[-2] - 1)
    sigma_y_half = sqrtm(sigma_y)

    trace = lambda m: m.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    fd = (
        (mu_x - mu_y).pow(2).sum(dim=-1)
        + trace(sigma_x)
        + trace(sigma_y)
        - 2 * trace(sqrtm(sigma_y_half @ sigma_x @ sigma_y_half))
    )
    return fd.to(dtype)


@cache
def inception() -> nn.Module:
    r"""Load the Inception-v3 network pretrained on ImageNet, without its classification layer.

    Returns:
        network : Network mapping images (B, 3, 299, 299) in [-1, 1] to features (B, 2048).
    """
    network = torchvision.models.inception_v3(weights="DEFAULT", transform_input=False)
    network.fc = nn.Identity()
    return network.eval().requires_grad_(False)


@torch.no_grad()
def frechet_inception_distance(
    x: Tensor,
    y: Tensor,
    dims: str,
    samples: str,
    spatial: str,
    mask: Tensor | None = None,
    batch_size: int = 64,
) -> Tensor:
    r"""Compute the Fréchet inception distance (FID) between two sets of 2D fields.

    Formula:
        FID = FD(Inception(x), Inception(y))

    Arguments:
        x          : First set of fields, e.g. the ground truth.
        y          : Second set of fields, with the same shape as `x` except along `samples`.
        dims       : Space-separated axis names describing the shape of `x`, e.g. "N C Y X".
        samples    : Space-separated subset of `dims` indexing the samples, e.g. "N".
        spatial    : Space-separated names of the two spatial axes forming an image, e.g. "Y X".
        mask       : Boolean-like tensor, broadcastable with `x` and `y`, zero where invalid.
        batch_size : Number of images passed at once through the Inception network.

    Returns:
        FID, with the axes in `samples` and `spatial` collapsed.

    Example:
        >>> x = torch.randn(64, 2, 128, 256)  # (N, C, Y, X)
        >>> y = torch.randn(32, 2, 128, 256)  # (N, C, Y, X)
        >>> frechet_inception_distance(
        ...     x=x,
        ...     y=y,
        ...     dims="N C Y X",
        ...     samples="N",
        ...     spatial="Y X",
        ... ).shape
        torch.Size([2])
    """
    samples_axes, spatial_axes = axes(dims, samples), axes(dims, spatial)
    assert len(spatial_axes) == 2, f"ERROR - spatial must name two axes, got '{spatial}'."
    tail = tuple(range(-len(samples_axes) - 2, 0))

    # Both sets are scaled to [0, 1] with the range of `x`, so that a value has the same color
    x, y = apply_mask(x, mask), apply_mask(y, mask)
    lower = x.nan_to_num(torch.inf).amin(dim=samples_axes + spatial_axes, keepdim=True)
    upper = x.nan_to_num(-torch.inf).amax(dim=samples_axes + spatial_axes, keepdim=True)
    network = inception().to(x.device)

    # Each field becomes a grey image, masked entries being black
    def features(z: Tensor) -> Tensor:
        z = ((z - lower) / (upper - lower).clamp(min=1e-8)).clamp(0, 1).nan_to_num(0.0)
        z = z.movedim(samples_axes + spatial_axes, tail).flatten(-len(samples_axes) - 2, -3)
        images = z.reshape(-1, 1, *z.shape[-2:]).float()
        out = []
        for batch in images.split(batch_size):
            # The network was trained on 299 x 299 images with values in [-1, 1]
            batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
            out.append(network(2 * batch.expand(-1, 3, -1, -1) - 1))
        return torch.cat(out).reshape(*z.shape[:-2], -1)

    kept = " ".join(name for name in dims.split() if name not in f"{samples} {spatial}".split())
    return frechet_distance(
        features(x),
        features(y),
        dims=f"{kept} samples features",
        samples="samples",
        features="features",
    )
