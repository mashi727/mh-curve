"""chi_n(t) temperature scan (CHI999 menu 1, *KIM3)."""

from __future__ import annotations

import numpy as np

from . import analytic, numeric
from .common import Exponents, Params, harmonics, loop_field


def chi_vs_t(
    base: Params, exps: Exponents, ts: np.ndarray, engine: str = "analytic", nmax: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    """Return (chi', chi'') with shape (len(ts), nmax). Points that cannot be evaluated are NaN."""
    ts = np.asarray(ts, dtype=float)
    si0 = base.si
    if engine == "numeric":
        ps = [base.at_t(t, exps).si for t in ts]
        M = numeric.loops_batch(
            [p.K for p in ps], [p.B0 for p in ps], [p.dH for p in ps], [p.Hc1 for p in ps],
            si0.a, si0.Hac, si0.Hdc, si0.T,
        )
        return harmonics(M, si0.Hac, nmax)

    re = np.full((len(ts), nmax), np.nan)
    im = np.full((len(ts), nmax), np.nan)
    for j, t in enumerate(ts):
        p = base.at_t(t, exps).si
        if p.K <= 0:  # t = 1: CHI999 skips this point (line 2680)
            continue
        try:
            M = analytic.compute_loop(p).M_loop
        except analytic.AnalyticError:
            continue
        re[j], im[j] = harmonics(M, p.Hac, nmax)
    return re, im


def t_grid(tmin: float, tmax: float, step: float) -> np.ndarray:
    n = int(np.floor((tmax - tmin) / step + 1e-9)) + 1
    return tmin + step * np.arange(max(n, 1))


__all__ = ["chi_vs_t", "t_grid", "loop_field"]
