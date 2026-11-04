r"""Tests for albus.frechet."""

import pytest
import torch

from scipy.linalg import sqrtm as scipy_sqrtm

from albus.frechet import *


def test_frechet_distance_matches_scipy() -> None:
    r"""Matches the reference formula computed with the matrix square root of scipy."""
    torch.manual_seed(0)
    x = torch.randn(200, 8, dtype=torch.float64)
    y = 2 * torch.randn(100, 8, dtype=torch.float64) + 1

    mu_x, mu_y = x.mean(dim=0), y.mean(dim=0)
    sigma_x, sigma_y = torch.cov(x.T), torch.cov(y.T)
    covmean = torch.from_numpy(scipy_sqrtm((sigma_x @ sigma_y).numpy()).real)
    expected = (mu_x - mu_y).pow(2).sum() + (sigma_x + sigma_y - 2 * covmean).trace()

    fd = frechet_distance(x, y, dims="N D", samples="N", features="D")
    assert torch.allclose(fd, expected, atol=1e-8)


def test_frechet_distance_zero_for_identical_sets() -> None:
    r"""Is zero when both sets of samples are the same."""
    x = torch.randn(100, 16)
    fd = frechet_distance(x, x.clone(), dims="N D", samples="N", features="D")
    assert torch.isclose(fd, torch.tensor(0.0), atol=1e-4)


def test_frechet_distance_keeps_other_axes() -> None:
    r"""Keeps the axes that are neither samples nor features, whatever their position."""
    torch.manual_seed(0)
    x = torch.randn(50, 3, 4, 5)  # (N, C, Y, X)
    y = torch.randn(30, 3, 4, 5)  # (N, C, Y, X)
    fd = frechet_distance(x, y, dims="N C Y X", samples="N", features="Y X")
    assert fd.shape == (3,)
    for c in range(3):
        fd_c = frechet_distance(
            x[:, c].flatten(1), y[:, c].flatten(1), dims="N D", samples="N", features="D"
        )
        assert torch.isclose(fd[c], fd_c, atol=1e-4)


def test_frechet_distance_rejects_single_sample() -> None:
    r"""Raises an error when a set has a single sample, as its covariance is undefined."""
    with pytest.raises(AssertionError):
        frechet_distance(
            torch.randn(1, 4), torch.randn(5, 4), dims="N D", samples="N", features="D"
        )


@pytest.mark.integration
def test_frechet_inception_distance_typical() -> None:
    r"""Is zero for identical sets and grows when the second set is corrupted by noise."""
    torch.manual_seed(0)
    x = torch.randn(16, 2, 32, 48).cumsum(dim=-1)  # (N, C, Y, X), smooth along X
    mask = torch.rand(32, 48) > 0.2
    kwargs = dict(dims="N C Y X", samples="N", spatial="Y X", mask=mask, batch_size=8)

    fid_same = frechet_inception_distance(x, x.clone(), **kwargs)
    fid_noisy = frechet_inception_distance(x, x + 5 * torch.randn_like(x), **kwargs)
    assert fid_same.shape == (2,)
    assert torch.allclose(fid_same, torch.zeros(2), atol=1e-2)
    assert (fid_noisy > 1.0).all()
