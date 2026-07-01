"""Matplotlib style for pruning convergence plots."""

from __future__ import annotations

import matplotlib as mpl

ALPHA_COLORS = {
    0.0: "#2166ac",
    0.25: "#4393c3",
    0.5: "#92c5de",
    0.75: "#f4a582",
    1.0: "#b2182b",
}

MPL_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}


def alpha_label(alpha: float) -> str:
    return f"{alpha:.2f}".rstrip("0").rstrip(".") if alpha % 1 else f"{int(alpha)}"


def apply_style() -> None:
    mpl.rcParams.update(MPL_RC)
