r"""Metrics borrowed from image quality assessment, applied to 2D fields."""

__all__ = [
    "frechet_distance",
    "frechet_inception_distance",
    "kernel_distance",
    "kernel_inception_distance",
    "peak_signal_to_noise_ratio",
    "structural_similarity",
]

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from functools import cache
from torch import Tensor

from albus.metrics import mean_square_error
from albus.tools import apply_mask, axes, drop


def extrema(x: Tensor, dim: tuple[int, ...]) -> tuple[Tensor, Tensor]:
    r"""Compute the minimum and the maximum of a tensor over the given axes, ignoring NaNs.

    Arguments:
        x   : Input tensor.
        dim : Positional indices of the axes to reduce over.

    Returns:
        lower : Minimum of `x` over `dim`, with the reduced axes kept with size one.
        upper : Maximum of `x` over `dim`, with the reduced axes kept with size one.
    """
    lower = x.nan_to_num(torch.inf).amin(dim=dim, keepdim=True)
    upper = x.nan_to_num(-torch.inf).amax(dim=dim, keepdim=True)
    return lower, upper


def flatten(x: Tensor, dims: str, samples: str, features: str) -> Tensor:
    r"""Flatten the samples and the features of a tensor into its last two axes.

    Arguments:
        x        : Input tensor.
        dims     : Space-separated axis names describing the shape of `x`, e.g. "T N D".
        samples  : Space-separated subset of `dims` indexing the samples, e.g. "N".
        features : Space-separated subset of `dims` spanning the feature space, e.g. "D".

    Returns:
        Tensor with shape (*, N, D), the other axes kept in their order.
    """
    samples_axes, features_axes = axes(dims, samples), axes(dims, features)
    tail = tuple(range(-len(samples_axes) - len(features_axes), 0))
    x = x.movedim(samples_axes + features_axes, tail).flatten(-len(features_axes))
    return x.flatten(-len(samples_axes) - 1, -2)


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
    # Matrix square roots are imprecise in single precision
    dtype = x.dtype
    x, y = (flatten(z, dims, samples, features).double() for z in (x, y))
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


def kernel_distance(
    x: Tensor,
    y: Tensor,
    dims: str,
    samples: str,
    features: str,
) -> Tensor:
    r"""Compute the unbiased squared maximum mean discrepancy between two sets of samples.

    Formula:
        KD = 𝔼[k(x, x')] + 𝔼[k(y, y')] - 2 𝔼[k(x, y)], with k(a, b) = (aᵀb / D + 1)³

    Arguments:
        x        : First set of samples.
        y        : Second set of samples, with the same shape as `x` except along `samples`.
        dims     : Space-separated axis names describing the shape of `x`, e.g. "T N D".
        samples  : Space-separated subset of `dims` indexing the samples, e.g. "N".
        features : Space-separated subset of `dims` spanning the feature space, e.g. "D".

    Returns:
        Kernel distance, with the axes in `samples` and `features` collapsed.

    Example:
        >>> x = torch.randn(7, 1000, 64)  # (T, N, D)
        >>> y = torch.randn(7, 500, 64)   # (T, N, D)
        >>> kernel_distance(
        ...     x=x,
        ...     y=y,
        ...     dims="T N D",
        ...     samples="N",
        ...     features="D",
        ... ).shape
        torch.Size([7])
    """
    # Sums over all pairs of samples are imprecise in single precision
    dtype = x.dtype
    x, y = (flatten(z, dims, samples, features).double() for z in (x, y))
    (m, d), n = x.shape[-2:], y.shape[-2]
    assert min(m, n) > 1, "ERROR - Each set needs at least two samples."

    kernel = lambda a, b: (a @ b.mT / d + 1).pow(3)
    k_xx, k_yy, k_xy = kernel(x, x), kernel(y, y), kernel(x, y)

    # The pairs of a sample with itself are left out, so that the estimate is unbiased
    off_diagonal = lambda k: k.sum(dim=(-2, -1)) - k.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    kd = (
        off_diagonal(k_xx) / (m * (m - 1))
        + off_diagonal(k_yy) / (n * (n - 1))
        - 2 * k_xy.mean(dim=(-2, -1))
    )
    return kd.to(dtype)


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
def inception_features(
    x: Tensor,
    y: Tensor,
    dims: str,
    samples: str,
    spatial: str,
    mask: Tensor | None = None,
    batch_size: int = 64,
) -> tuple[Tensor, Tensor]:
    r"""Compute the Inception-v3 features of two sets of 2D fields, seen as grey images.

    Arguments:
        x          : First set of fields, e.g. the ground truth.
        y          : Second set of fields, with the same shape as `x` except along `samples`.
        dims       : Space-separated axis names describing the shape of `x`, e.g. "N C Y X".
        samples    : Space-separated subset of `dims` indexing the samples, e.g. "N".
        spatial    : Space-separated names of the two spatial axes forming an image, e.g. "Y X".
        mask       : Boolean-like tensor, broadcastable with `x` and `y`, zero where invalid.
        batch_size : Number of images passed at once through the Inception network.

    Returns:
        features_x : Features of `x`, with shape (*, N, 2048), the other axes kept in their order.
        features_y : Features of `y`, with shape (*, N, 2048), the other axes kept in their order.
    """
    samples_axes, spatial_axes = axes(dims, samples), axes(dims, spatial)
    assert len(spatial_axes) == 2, f"ERROR - spatial must name two axes, got '{spatial}'."
    tail = tuple(range(-len(samples_axes) - 2, 0))

    # Both sets are scaled to [0, 1] with the range of `x`, so that a value has the same color
    x, y = apply_mask(x, mask), apply_mask(y, mask)
    lower, upper = extrema(x, samples_axes + spatial_axes)
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

    return features(x), features(y)


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
    features_x, features_y = inception_features(x, y, dims, samples, spatial, mask, batch_size)
    return frechet_distance(
        features_x,
        features_y,
        dims=f"{drop(dims, f'{samples} {spatial}')} samples features",
        samples="samples",
        features="features",
    )


def kernel_inception_distance(
    x: Tensor,
    y: Tensor,
    dims: str,
    samples: str,
    spatial: str,
    mask: Tensor | None = None,
    batch_size: int = 64,
) -> Tensor:
    r"""Compute the kernel inception distance (KID) between two sets of 2D fields.

    Formula:
        KID = KD(Inception(x), Inception(y))

    Arguments:
        x          : First set of fields, e.g. the ground truth.
        y          : Second set of fields, with the same shape as `x` except along `samples`.
        dims       : Space-separated axis names describing the shape of `x`, e.g. "N C Y X".
        samples    : Space-separated subset of `dims` indexing the samples, e.g. "N".
        spatial    : Space-separated names of the two spatial axes forming an image, e.g. "Y X".
        mask       : Boolean-like tensor, broadcastable with `x` and `y`, zero where invalid.
        batch_size : Number of images passed at once through the Inception network.

    Returns:
        KID, with the axes in `samples` and `spatial` collapsed.

    Example:
        >>> x = torch.randn(64, 2, 128, 256)  # (N, C, Y, X)
        >>> y = torch.randn(32, 2, 128, 256)  # (N, C, Y, X)
        >>> kernel_inception_distance(
        ...     x=x,
        ...     y=y,
        ...     dims="N C Y X",
        ...     samples="N",
        ...     spatial="Y X",
        ... ).shape
        torch.Size([2])
    """
    features_x, features_y = inception_features(x, y, dims, samples, spatial, mask, batch_size)
    return kernel_distance(
        features_x,
        features_y,
        dims=f"{drop(dims, f'{samples} {spatial}')} samples features",
        samples="samples",
        features="features",
    )


def peak_signal_to_noise_ratio(
    x: Tensor,
    y: Tensor,
    dims: str,
    spatial: str,
    mask: Tensor | None = None,
) -> Tensor:
    r"""Compute the peak signal-to-noise ratio (PSNR) of 2D fields against the ground truth.

    Formula:
        PSNR = 10 log₁₀(L² / 𝔼[(x - y)²]), with L = max(x) - min(x) over each field

    Arguments:
        x       : Ground truth tensor.
        y       : Predicted tensor, broadcastable with `x`.
        dims    : Space-separated axis names describing the shape of `x`, e.g. "T C Y X".
        spatial : Space-separated names of the two spatial axes forming a field, e.g. "Y X".
        mask    : Boolean-like tensor, broadcastable with `x`, zero where entries are invalid.

    Returns:
        PSNR [dB], with the axes in `spatial` collapsed.

    Example:
        >>> x = torch.randn(7, 2, 128, 256)  # (T, C, Y, X)
        >>> y = torch.randn(7, 2, 128, 256)  # (T, C, Y, X)
        >>> peak_signal_to_noise_ratio(
        ...     x=x,
        ...     y=y,
        ...     dims="T C Y X",
        ...     spatial="Y X",
        ... ).shape
        torch.Size([7, 2])
    """
    spatial_axes = axes(dims, spatial)
    lower, upper = extrema(apply_mask(x, mask), spatial_axes)
    mse = mean_square_error(x, y, dims, spatial, mask)
    return 10 * torch.log10((upper - lower).squeeze(spatial_axes).pow(2) / mse)


def structural_similarity(
    x: Tensor,
    y: Tensor,
    dims: str,
    spatial: str,
    mask: Tensor | None = None,
) -> Tensor:
    r"""Compute the structural similarity (SSIM) of 2D fields against the ground truth.

    Formula:
        SSIM = 𝔼[(2μₓμᵧ + c₁)(2σₓᵧ + c₂) / ((μₓ² + μᵧ² + c₁)(σₓ² + σᵧ² + c₂))]

    Arguments:
        x       : Ground truth tensor.
        y       : Predicted tensor, broadcastable with `x`.
        dims    : Space-separated axis names describing the shape of `x`, e.g. "T C Y X".
        spatial : Space-separated names of the two spatial axes forming a field, e.g. "Y X".
        mask    : Boolean-like tensor, broadcastable with `x`, zero where entries are invalid.

    Returns:
        SSIM, with the axes in `spatial` collapsed (1 for a perfect prediction).

    Example:
        >>> x = torch.randn(7, 2, 128, 256)  # (T, C, Y, X)
        >>> y = torch.randn(7, 2, 128, 256)  # (T, C, Y, X)
        >>> structural_similarity(
        ...     x=x,
        ...     y=y,
        ...     dims="T C Y X",
        ...     spatial="Y X",
        ... ).shape
        torch.Size([7, 2])
    """
    spatial_axes = axes(dims, spatial)
    assert len(spatial_axes) == 2, f"ERROR - spatial must name two axes, got '{spatial}'."

    # Both fields are scaled with the range of `x`, like images with values in [0, 1]
    x, y = torch.broadcast_tensors(apply_mask(x, mask), apply_mask(y, mask))
    lower, upper = extrema(x, spatial_axes)
    x, y = ((z - lower) / (upper - lower).clamp(min=1e-8) for z in (x, y))
    x, y = (z.movedim(spatial_axes, (-2, -1)) for z in (x, y))
    shape = x.shape
    x, y = (z.reshape(-1, 1, *shape[-2:]) for z in (x, y))
    valid = (~x.isnan() & ~y.isnan()).to(x.dtype)
    x, y = (z.masked_fill(valid == 0, 0.0) for z in (x, y))

    # Gaussian window of the original SSIM, 11 x 11 with a standard deviation of 1.5
    gaussian = torch.exp(-(torch.arange(11, dtype=x.dtype, device=x.device) - 5).pow(2) / 4.5)
    window = (gaussian[:, None] * gaussian[None, :] / gaussian.sum().pow(2))[None, None]

    # Local statistics only weigh the valid entries of each window, near the coasts and edges too
    weight = F.conv2d(valid, window, padding=5).clamp(min=1e-8)
    local = lambda z: F.conv2d(z, window, padding=5) / weight

    mu_x, mu_y = local(x), local(y)
    sigma_xx = local(x * x) - mu_x.pow(2)
    sigma_yy = local(y * y) - mu_y.pow(2)
    sigma_xy = local(x * y) - mu_x * mu_y

    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_xx + sigma_yy + c2)
    )
    ssim = ssim.masked_fill(valid == 0, torch.nan)
    return ssim.reshape(shape).nanmean(dim=(-2, -1))
