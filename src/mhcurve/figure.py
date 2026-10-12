"""Publication figures: matplotlib -> tightly trimmed vector PDF for \\includegraphics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # never touch the Qt event loop
import matplotlib as mpl  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .io import Dataset  # noqa: E402
from .palette import PAD, SERIES, dash_for  # noqa: E402
from .view import PdfSettings  # noqa: E402

MM = 1 / 25.4
FigureStyle = PdfSettings  # backwards-compatible name

FONTS = {
    "STIX Two (Times 系)": (["STIX Two Text", "STIXGeneral"], "stix"),
    "DejaVu Serif": (["DejaVu Serif"], "dejavuserif"),
    "DejaVu Sans": (["DejaVu Sans"], "dejavusans"),
}
_JP_FALLBACK = ["Hiragino Mincho ProN", "Hiragino Sans", "Yu Mincho", "MS Mincho"]  # macOS, Windows
ARRANGEMENTS = {"vertical": "縦一列", "horizontal": "横一列", "grid": "2 列の格子"}
_MPL_DASH = ["-", (0, (5, 2)), (0, (1, 1.5)), (0, (5, 1.5, 1, 1.5))]


@dataclass
class Curve:
    ds: Dataset
    color: str
    dash: int
    label: str


def hp_mT(ds: Dataset) -> float:
    """Full-penetration field stored in the data header (M-H data only)."""
    return float(ds.meta["derived.Hp_mT"])


def _rc(style: PdfSettings) -> dict:
    serif, math_fs = FONTS.get(style.font, next(iter(FONTS.values())))
    fam = "sans-serif" if "Sans" in style.font else "serif"
    return {
        "font.family": fam,
        f"font.{fam}": serif + _JP_FALLBACK,
        "mathtext.fontset": math_fs,
        "font.size": style.font_pt,
        "axes.labelsize": style.font_pt,
        "legend.fontsize": style.font_pt - 1,
        "xtick.labelsize": style.font_pt - 0.5,
        "ytick.labelsize": style.font_pt - 0.5,
        "axes.linewidth": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "lines.linewidth": 1.0,
        "legend.frameon": False,
        "pdf.fonttype": 42,  # embed TrueType, text stays selectable
        "axes.unicode_minus": True,
    }


def _grid(n: int, arrangement: str) -> tuple[int, int]:
    if n == 1:
        return 1, 1
    if arrangement == "horizontal":
        return 1, n
    if arrangement == "grid":
        return math.ceil(n / 2), 2
    return n, 1


def _draw_mh(ax: Axes, curves: list[Curve], flip: bool, normalize: bool, legend: bool) -> None:
    sgn = -1.0 if flip else 1.0
    for c in curves:
        s = hp_mT(c.ds) if normalize else 1.0
        br = c.ds.columns["branch"]
        H, M = c.ds.columns["H_mT"] / s, c.ds.columns["M_mT"] * sgn / s
        lp, ini = br == "loop", br == "init"
        ax.plot(H[lp], M[lp], color=c.color, ls=_MPL_DASH[c.dash % 4], label=c.label)
        ax.plot(H[ini], M[ini], color=c.color, ls=":", lw=0.8)
    ax.axhline(0, color="0.6", lw=0.4, zorder=0)
    ax.axvline(0, color="0.6", lw=0.4, zorder=0)
    if normalize:
        ax.set_xlabel(r"$H/H_\mathrm{p}$")
        ax.set_ylabel(r"$-M/H_\mathrm{p}$" if flip else r"$M/H_\mathrm{p}$")
    else:
        ax.set_xlabel(r"$\mu_0 H$ (mT)")
        ax.set_ylabel(r"$-\mu_0 M$ (mT)" if flip else r"$\mu_0 M$ (mT)")
    if legend and len(curves) > 1:
        ax.legend()


def _draw_chi(ax: Axes, curves: list[Curve], n: int, show_re: bool, show_im: bool,
              legend: bool) -> None:
    single = len(curves) == 1
    for c in curves:
        t = c.ds.columns["t"]
        if show_re:
            ax.plot(t, c.ds.columns[f"chi{n}_re"], color=c.color, ls="-",
                    label=rf"$\chi_{n}'$" if single else c.label)
        if show_im:
            ax.plot(t, c.ds.columns[f"chi{n}_im"], color=c.color, ls="--",
                    label=rf"$\chi_{n}''$" if single or not show_re else None)
    ax.axhline(0, color="0.6", lw=0.4, zorder=0)
    ax.set_xlabel(r"$t = T/T_\mathrm{c}$")
    parts = ([rf"$\chi_{n}'$"] if show_re else []) + ([rf"$\chi_{n}''$"] if show_im else [])
    ax.set_ylabel(", ".join(parts))
    if legend:
        ax.legend()


def render(kind: str, panels: list[tuple[str, list[Curve]]], path: Path, style: PdfSettings, *,
           flip: bool = False, normalize: bool = False, n: int = 1, show_re: bool = True,
           show_im: bool = True) -> Path:
    """One axes per (panel key, curves); panel keys are printed as (a), (b), ... if >1 panel."""
    if kind == "chi" and not (show_re or show_im):
        raise ValueError("χ′ と χ″ のどちらかを表示してください")
    npan = len(panels)
    rows, cols = _grid(npan, style.arrangement)
    pw = style.width_mm * MM / cols
    with mpl.rc_context(_rc(style)):
        fig = Figure(figsize=(pw * cols, pw * style.aspect * rows), layout="constrained")
        axs = fig.subplots(rows, cols, squeeze=False).ravel()
        for ax, (key, curves) in zip(axs, panels):
            ax.margins(PAD)  # data fills 90 % of the frame, as on screen
            if kind == "mh":
                _draw_mh(ax, curves, flip, normalize, style.legend)
            else:
                _draw_chi(ax, curves, n, show_re, show_im, style.legend)
            if style.panel_labels and npan > 1:
                ax.text(0.96, 0.95, f"({key})", transform=ax.transAxes, ha="right", va="top")
        for ax in axs[npan:]:
            ax.remove()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.01)
    return path


# ---- single-panel shortcuts ---------------------------------------------------------
def _curves(datasets: list[Dataset]) -> list[Curve]:
    return [Curve(d, SERIES[i % 8], i // 8, d.label) for i, d in enumerate(datasets)]


def mh_pdf(datasets: list[Dataset], path: Path, style: PdfSettings, flip: bool = False,
           normalize: bool = False) -> Path:
    return render("mh", [("a", _curves(datasets))], path, style, flip=flip, normalize=normalize)


def chi_pdf(datasets: list[Dataset], path: Path, style: PdfSettings, n: int, show_re: bool,
            show_im: bool) -> Path:
    return render("chi", [("a", _curves(datasets))], path, style, n=n, show_re=show_re,
                  show_im=show_im)


def latex_snippet(pdf: Path, base: Path | None = None) -> str:
    """\\includegraphics at natural size: the trimmed PDF is narrower than the requested
    width, and scaling it to \\columnwidth would also scale the 8 pt labels."""
    rel = pdf.relative_to(base) if base and pdf.is_relative_to(base) else pdf
    return (
        "\\begin{figure}[tb]\n  \\centering\n"
        f"  \\includegraphics{{{rel.as_posix()}}}  % natural size keeps the font size\n"
        "  \\caption{}\n  \\label{fig:" + pdf.stem + "}\n\\end{figure}\n"
    )


__all__ = ["ARRANGEMENTS", "FONTS", "Curve", "FigureStyle", "render", "mh_pdf", "chi_pdf",
           "latex_snippet", "hp_mT", "dash_for"]
