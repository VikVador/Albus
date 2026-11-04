r"""Helper tools for metric implementations."""

__all__ = [
    "apply_mask",
    "axes",
    "drop",
    "nanmean",
]

from torch import Tensor


def apply_mask(
    x: Tensor,
    mask: Tensor | None,
    value: float = float("nan"),
) -> Tensor:
    r"""Replace masked-out entries with a given value.

    Arguments:
        x     : Input tensor.
        mask  : Boolean-like tensor, broadcastable with `x`, zero where entries are invalid.
        value : Value to replace masked-out entries with.

    Returns:
        Tensor with masked-out entries replaced by `value`.

    Example:
        >>> apply_mask(torch.ones(3), torch.tensor([1, 0, 1]))
        tensor([1., nan, 1.])
    """
    return x if mask is None else x.masked_fill(mask == 0, value)


def axes(
    dims: str,
    names: str,
) -> tuple[int, ...]:
    r"""Resolve named axes to their positional indices.

    Arguments:
        dims  : Space-separated axis names describing the shape of a tensor, e.g. "T N C Y X".
        names : Space-separated subset of `dims` to resolve, e.g. "Y X".

    Returns:
        Positional indices of `names` within `dims`.

    Example:
        >>> axes("T N C Y X", "Y X")
        (3, 4)
    """
    positions = {name: axis for axis, name in enumerate(dims.split())}
    for name in names.split():
        assert name in positions, f"ERROR - Axis '{name}' is not in dims '{dims}'."
    return tuple(positions[name] for name in names.split())


def drop(
    dims: str,
    names: str,
) -> str:
    r"""Remove named axes from a string of axis names.

    Arguments:
        dims  : Space-separated axis names describing the shape of a tensor, e.g. "T N C Y X".
        names : Space-separated axis names to remove, e.g. "N" or "N Y X".

    Returns:
        `dims` without `names`.

    Example:
        >>> drop("T N C Y X", "N")
        'T C Y X'
        >>> drop("T N C Y X", "N Y X")
        'T C'
    """
    return " ".join(axis for axis in dims.split() if axis not in names.split())


def nanmean(
    x: Tensor,
    dim: tuple[int, ...],
    keepdim: bool = False,
) -> Tensor:
    r"""Average over the given axes ignoring NaNs, or return `x` unchanged if there are none.

    Arguments:
        x       : Input tensor.
        dim     : Positional indices of the axes to average over, possibly empty.
        keepdim : Whether to keep the reduced axes with size one.

    Returns:
        Mean of `x` over `dim`, or `x` itself if `dim` is empty.

    Example:
        >>> nanmean(torch.randn(7, 2, 128, 256), dim=(2, 3)).shape
        torch.Size([7, 2])
        >>> nanmean(torch.randn(7, 2, 128, 256), dim=()).shape
        torch.Size([7, 2, 128, 256])
    """
    # Unlike an empty `reduce`, which asks for pointwise values, `torch.nanmean(dim=())`
    # reduces over all axes.
    return x if len(dim) == 0 else x.nanmean(dim=dim, keepdim=keepdim)
