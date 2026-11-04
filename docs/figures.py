r"""Figures of the tutorials, computed on the sample data in docs/data.

Run from the root of the repository: python docs/figures.py
"""

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from pathlib import Path
from torch import Tensor

from albus.metrics import (
    continuous_ranked_probability_score,
    power_spectrum,
    root_mean_square_error,
    spread_skill_ratio,
    standard_deviation,
)
from albus.ocean.oxygen import hypoxia
from albus.visuals import (
    frechet_inception_distance,
    kernel_inception_distance,
    peak_signal_to_noise_ratio,
    structural_similarity,
)

# fmt: off
#
# ----- Style of the figures: colourblind-safe colours, one per depth, and mathematics in Computer Modern
#
DATA    = Path("docs/data")
IMAGES  = Path("docs/images")
DEPTHS  = ["0.25 m", "9 m"]
COLORS  = ["#0072B2", "#E69F00"]
MUTED   = "#9a9a9a"
OXYGEN  = r"\mathrm{mmol\,m^{-3}}"
# fmt: on

plt.rcParams.update({
    "font.size": 10,
    "mathtext.fontset": "cm",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "axes.labelpad": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.constrained_layout.use": True,
    "svg.hashsalt": "albus",
})


def plot_lead_time(ax: plt.Axes, values: Tensor, ylabel: str, **kwargs) -> None:
    r"""Plot a metric against the lead time, one line per depth.

    Arguments:
        ax     : Axis to plot on.
        values : Metric, with shape (T, C).
        ylabel : Label of the y axis, with the unit in brackets.
        kwargs : Keyword arguments passed to the plot (e.g. the line style).
    """
    lead_times = range(1, values.shape[0] + 1)
    for c, depth in enumerate(DEPTHS):
        ax.plot(
            lead_times,
            values[:, c],
            color=COLORS[c],
            marker="o",
            markersize=3,
            label=depth,
            **kwargs,
        )
    ax.set_xticks(lead_times)
    ax.set_ylabel(ylabel)


def save(fig: plt.Figure, name: str) -> None:
    r"""Save a figure as SVG in the images of the documentation.

    Arguments:
        fig  : Figure to save.
        name : Name of the file, without extension.
    """
    fig.savefig(IMAGES / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)


def main() -> None:
    r"""Compute every metric of the tutorials on the sample data and draw their figures."""

    IMAGES.mkdir(exist_ok=True)
    mask = torch.load(DATA / "mask.pt", weights_only=True)
    truth = torch.load(DATA / "o-truth.pt", weights_only=True)
    forecast = torch.load(DATA / "o-forecast.pt", weights_only=True)

    # Fields: the truth and two members at the surface, on the last day
    fields = [truth[-1, 0], forecast[-1, 0, 0], forecast[-1, 1, 0]]
    vmin, vmax = torch.stack(fields).nanquantile(torch.tensor([0.02, 0.98])).tolist()
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.2), sharex=True, sharey=True)
    for ax, field, title in zip(axes, fields, ["Truth", "Member 1", "Member 2"], strict=True):
        image = ax.imshow(field, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel(r"Grid point $X$ [$-$]")
    axes[0].set_ylabel(r"Grid point $Y$ [$-$]")
    cax = axes[-1].inset_axes([1.05, 0, 0.05, 1])
    fig.colorbar(image, cax=cax, label=rf"Oxygen [${OXYGEN}$]", extend="both")
    save(fig, "fields")

    # Metrics: errors and spread of the forecast
    kwargs = dict(dims="T N C Y X", mask=mask)
    rmse = root_mean_square_error(x=truth.unsqueeze(1), y=forecast, reduce="N Y X", **kwargs)
    std = standard_deviation(x=forecast, reduce="N Y X", **kwargs)
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.6), sharex=True)
    plot_lead_time(axes[0], rmse, rf"Root mean square error [${OXYGEN}$]")
    plot_lead_time(axes[1], std, rf"Standard deviation [${OXYGEN}$]")
    axes[0].legend()
    for ax in axes:
        ax.set_xlabel(r"Lead time [$\mathrm{days}$]")
    save(fig, "errors")

    # Metrics: skill, spread and CRPS of the ensemble
    kwargs = dict(dims="T N C Y X", ensemble="N", reduce="Y X", mask=mask)
    skill, spread, ratio = spread_skill_ratio(x=truth, y=forecast, **kwargs)
    crps = continuous_ranked_probability_score(x=truth, y=forecast, **kwargs)
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.6), sharex=True)
    plot_lead_time(axes[0], skill, rf"Skill and spread [${OXYGEN}$]")
    plot_lead_time(axes[0], spread, rf"Skill and spread [${OXYGEN}$]", linestyle="--")
    axes[0].annotate(
        "skill", (3, skill[2, 0]), xytext=(0, 8), textcoords="offset points", ha="center"
    )
    axes[0].annotate(
        "spread", (3, spread[2, 0]), xytext=(0, -14), textcoords="offset points", ha="center"
    )
    plot_lead_time(axes[1], ratio, r"Spread-skill ratio [$-$]")
    axes[1].axhline(1.0, color=MUTED, linewidth=1, zorder=0)
    plot_lead_time(axes[2], crps, rf"Continuous ranked" "\n" rf"probability score [${OXYGEN}$]")
    axes[2].legend()
    for ax in axes:
        ax.set_xlabel(r"Lead time [$\mathrm{days}$]")
    save(fig, "ensemble")

    # Metrics: power spectra of the truth and of the ensemble mean, on the last day
    kwargs = dict(dims="T C Y X", spatial="Y X", mask=mask)
    spectrum_truth = power_spectrum(x=truth, **kwargs)
    spectrum_forecast = power_spectrum(x=forecast.nanmean(dim=1), **kwargs)
    # The wavenumber zero is left out, as the mean of each field is removed
    wavenumbers = torch.arange(1, spectrum_truth.shape[-1])
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for c, depth in enumerate(DEPTHS):
        truth_c, forecast_c = spectrum_truth[-1, c, 1:], spectrum_forecast[-1, c, 1:]
        ax.loglog(wavenumbers, truth_c, color=COLORS[c], label=f"Truth, {depth}")
        ax.loglog(wavenumbers, forecast_c, color=COLORS[c], linestyle="--", label=f"Mean, {depth}")
    ax.set_xlabel(r"Wavenumber [$-$]")
    ax.set_ylabel(rf"Power spectral density [$({OXYGEN})^2$]")
    ax.legend(loc="lower left")
    save(fig, "spectrum")

    # Ocean: hypoxia scores, with a threshold inside the range of the sample
    out = hypoxia(
        x=truth.unsqueeze(1),
        y=forecast,
        dims="T N C Y X",
        reduce="N Y X",
        thresholds=[287.0, 287.0],
        thresholds_curve=torch.linspace(190.0, 380.0, 20).tolist(),
        mask=mask,
    )
    names = ["accuracy", "balanced_accuracy", "f1", "precision", "recall"]
    fig, axes = plt.subplots(1, 5, figsize=(10, 2.4), sharex=True, sharey=True)
    for ax, name in zip(axes, names, strict=True):
        plot_lead_time(ax, out[name], "Score [$-$]")
        ax.set_title(name.replace("_", " ").capitalize())
        ax.set_xlabel(r"Lead time [$\mathrm{days}$]")
        ax.set_xticks([1, 4, 7])
    for ax in axes[1:]:
        ax.set_ylabel("")
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc="lower left")
    save(fig, "hypoxia-scores")

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    for c, depth in enumerate(DEPTHS):
        roc = f"{depth}, area {out['roc_auc'][-1, c]:.2f}"
        pr = f"{depth}, area {out['pr_auc'][-1, c]:.2f}"
        axes[0].plot(
            out["roc_fpr"][:, -1, c], out["roc_tpr"][:, -1, c], "o-", ms=3, c=COLORS[c], label=roc
        )
        axes[1].plot(
            out["pr_recall"][:, -1, c],
            out["pr_precision"][:, -1, c],
            "o-",
            ms=3,
            c=COLORS[c],
            label=pr,
        )
    axes[0].plot([0, 1], [0, 1], color=MUTED, linewidth=1, zorder=0)
    axes[0].set(xlabel=r"False positive rate [$-$]", ylabel=r"True positive rate [$-$]")
    axes[1].set(xlabel=r"Recall [$-$]", ylabel=r"Precision [$-$]")
    for ax in axes:
        ax.legend(loc="lower right" if ax is axes[0] else "lower left")
    save(fig, "hypoxia-curves")

    # Visuals: PSNR and SSIM of each member, and their mean
    kwargs = dict(x=truth.unsqueeze(1), y=forecast, dims="T N C Y X", spatial="Y X", mask=mask)
    psnr = peak_signal_to_noise_ratio(**kwargs)
    ssim = structural_similarity(**kwargs)
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.6), sharex=True)
    for ax, values, ylabel in zip(
        axes,
        [psnr, ssim],
        [r"Peak signal-to-noise ratio [$\mathrm{dB}$]", r"Structural similarity [$-$]"],
        strict=True,
    ):
        lead_times = torch.arange(1, values.shape[0] + 1)
        for c in range(len(DEPTHS)):
            for n in range(values.shape[1]):
                ax.scatter(
                    lead_times, values[:, n, c], s=6, color=COLORS[c], alpha=0.4, linewidth=0
                )
        plot_lead_time(ax, values.mean(dim=1), ylabel)
        ax.set_xlabel(r"Lead time [$\mathrm{days}$]")
    axes[0].legend()
    save(fig, "visuals-scores")

    # Visuals: FID and KID of the forecast and of a smoothed truth, every day and member being a sample
    smoothed = F.avg_pool2d(truth, 5, stride=1, padding=2).unsqueeze(1)
    kwargs = dict(dims="T N C Y X", samples="T N", spatial="Y X", mask=mask)
    for name, y in (
        ("truth", truth.unsqueeze(1)),
        ("forecast", forecast),
        ("smoothed truth", smoothed),
    ):
        fid = frechet_inception_distance(x=truth.unsqueeze(1), y=y, **kwargs)
        kid = kernel_inception_distance(x=truth.unsqueeze(1), y=y, **kwargs)
        print(f"{name:15s} FID {fid.tolist()} KID {kid.tolist()}")


if __name__ == "__main__":
    main()
