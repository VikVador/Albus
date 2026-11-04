r"""Tests for albus.visuals."""

import math
import pytest
import torch

from scipy.linalg import sqrtm as scipy_sqrtm

from albus.visuals import *


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


def test_kernel_distance_matches_brute_force() -> None:
    r"""Matches a double-loop implementation of the unbiased squared MMD."""
    torch.manual_seed(0)
    x = torch.randn(12, 5, dtype=torch.float64)
    y = torch.randn(9, 5, dtype=torch.float64) + 0.5
    k = lambda a, b: (a @ b / 5 + 1) ** 3

    m, n = len(x), len(y)
    k_xx = sum(k(x[i], x[j]) for i in range(m) for j in range(m) if i != j) / (m * (m - 1))
    k_yy = sum(k(y[i], y[j]) for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
    k_xy = sum(k(x[i], y[j]) for i in range(m) for j in range(n)) / (m * n)

    kd = kernel_distance(x, y, dims="N D", samples="N", features="D")
    assert torch.isclose(kd, k_xx + k_yy - 2 * k_xy)


def test_kernel_distance_detects_shift() -> None:
    r"""Is close to zero for two sets of the same distribution, and grows with a shift."""
    torch.manual_seed(0)
    x, y = torch.randn(2, 500, 8)  # two sets of 500 samples (N, D)
    kd_same = kernel_distance(x, y, dims="N D", samples="N", features="D")
    kd_shift = kernel_distance(x, y + 1, dims="N D", samples="N", features="D")
    assert kd_same.abs() < 0.05
    assert kd_shift > 20 * kd_same.abs()


def test_peak_signal_to_noise_ratio_manual() -> None:
    r"""Matches a hand-computed PSNR, ignoring masked entries."""
    x = torch.tensor([[0.0, 1.0, 2.0, 3.0, 100.0]])  # (Y, X), range 3 once masked
    y = x + torch.tensor([[1.0, -1.0, 1.0, -1.0, 50.0]])  # mean square error 1 once masked
    mask = torch.tensor([1, 1, 1, 1, 0])
    psnr = peak_signal_to_noise_ratio(x, y, dims="Y X", spatial="Y X", mask=mask)
    assert torch.isclose(psnr, torch.tensor(10 * math.log10(9)))


def test_peak_signal_to_noise_ratio_keeps_other_axes() -> None:
    r"""Collapses only the spatial axes, wherever they are."""
    x, y = torch.randn(4, 6, 3, 5), torch.randn(4, 6, 3, 5)  # (Y, X, T, C)
    assert peak_signal_to_noise_ratio(x, y, dims="Y X T C", spatial="Y X").shape == (3, 5)


def test_structural_similarity_one_for_perfect_prediction() -> None:
    r"""Is one when the prediction equals the ground truth."""
    x = torch.randn(3, 2, 20, 30)  # (T, C, Y, X)
    ssim = structural_similarity(x, x.clone(), dims="T C Y X", spatial="Y X")
    assert ssim.shape == (3, 2)
    assert torch.allclose(ssim, torch.ones(3, 2), atol=1e-4)


def test_structural_similarity_matches_brute_force() -> None:
    r"""Matches a pixel-by-pixel loop over Gaussian windows restricted to the valid entries."""
    torch.manual_seed(0)
    x = torch.randn(9, 10, dtype=torch.float64)
    y = x + 0.5 * torch.randn(9, 10, dtype=torch.float64)
    mask = torch.rand(9, 10) > 0.2

    lower, upper = x[mask].min(), x[mask].max()
    xs, ys = (x - lower) / (upper - lower), (y - lower) / (upper - lower)
    g = torch.exp(-((torch.arange(11, dtype=torch.float64) - 5) ** 2) / 4.5)
    c1, c2 = 0.01**2, 0.03**2

    values = []
    for i, j in mask.nonzero().tolist():
        w, a, b = [], [], []
        for di in range(-5, 6):
            for dj in range(-5, 6):
                k, l = i + di, j + dj
                if 0 <= k < 9 and 0 <= l < 10 and mask[k, l]:
                    w.append(g[di + 5] * g[dj + 5])
                    a.append(xs[k, l])
                    b.append(ys[k, l])
        w, a, b = torch.stack(w), torch.stack(a), torch.stack(b)
        w = w / w.sum()
        mu_a, mu_b = (w * a).sum(), (w * b).sum()
        var_a, var_b = (w * (a - mu_a) ** 2).sum(), (w * (b - mu_b) ** 2).sum()
        cov = (w * (a - mu_a) * (b - mu_b)).sum()
        num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
        values.append(num / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)))

    ssim = structural_similarity(x, y, dims="Y X", spatial="Y X", mask=mask)
    assert torch.isclose(ssim, torch.stack(values).mean())


def test_structural_similarity_ignores_masked_entries() -> None:
    r"""Is unchanged when the prediction changes where the mask is zero."""
    torch.manual_seed(0)
    x = torch.randn(2, 16, 16)
    y = x + 0.3 * torch.randn(2, 16, 16)
    mask = torch.rand(16, 16) > 0.3
    y_changed = torch.where(mask, y, torch.full_like(y, 1e3))
    ssim = structural_similarity(x, y, dims="T Y X", spatial="Y X", mask=mask)
    ssim_changed = structural_similarity(x, y_changed, dims="T Y X", spatial="Y X", mask=mask)
    assert torch.allclose(ssim, ssim_changed)


def test_structural_similarity_ignores_nan_predictions() -> None:
    r"""Treats NaNs in the prediction alone as masked entries, and stays within [-1, 1]."""
    torch.manual_seed(0)
    x = torch.randn(2, 16, 16)
    x[:, 0, 0], x[:, 0, 1] = -10.0, 10.0  # same range of `x` with both masks
    y = x + 0.3 * torch.randn(2, 16, 16)
    mask = torch.rand(16, 16) > 0.3
    mask[0, :2] = True
    nan = torch.rand(16, 16) > 0.8
    nan[0, :2] = False

    ssim_nan = structural_similarity(
        x, y.masked_fill(nan, torch.nan), dims="T Y X", spatial="Y X", mask=mask
    )
    ssim = structural_similarity(x, y, dims="T Y X", spatial="Y X", mask=mask & ~nan)
    assert torch.allclose(ssim_nan, ssim)
    assert (ssim_nan.abs() <= 1).all()


@pytest.mark.integration
def test_inception_distances_typical() -> None:
    r"""Are close to zero for identical sets and grow when the second set is corrupted by noise."""
    torch.manual_seed(0)
    x = torch.randn(16, 2, 32, 48).cumsum(dim=-1)  # (N, C, Y, X), smooth along X
    noisy = x + 5 * torch.randn_like(x)
    mask = torch.rand(32, 48) > 0.2
    kwargs = dict(dims="N C Y X", samples="N", spatial="Y X", mask=mask, batch_size=8)

    fid_same = frechet_inception_distance(x, x.clone(), **kwargs)
    fid_noisy = frechet_inception_distance(x, noisy, **kwargs)
    assert fid_same.shape == (2,)
    assert torch.allclose(fid_same, torch.zeros(2), atol=1e-2)
    assert (fid_noisy > 1.0).all()

    kid_same = kernel_inception_distance(x, x.clone(), **kwargs)
    kid_noisy = kernel_inception_distance(x, noisy, **kwargs)
    assert kid_same.shape == (2,)
    assert (kid_noisy > 10 * kid_same.abs()).all()
