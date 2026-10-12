"""General-purpose numerical critical-state solver for the same modified Kim-Anderson model.

Independent of the 113-stage closed form: it only uses the two physical rules of the model,

1. Surface (Dunn-Hlawiczka, Heff = H - sgn(H) Hc1 - sgn(dH/dt) dH): the flux density just inside the surface b_s follows a
   play (backlash) operator of half-width Delta H around the reversible value
   b_eq(H) = sgn(H) max(|H| - Hc1, 0):
       b_s <- clip(b_s, b_eq(H - dH), b_eq(H + dH))
   which reproduces Heff(1..4) and the complete flux trapping over 2 Delta H.

2. Bulk (Kim-Anderson, Jc = k / (B0 + |B|)): with g(b) = sgn(b)(B0|b| + b^2/2) the critical profile is
   linear in g, d g / dx = +/- mu0 k. After each field step the profile is the envelope
   max(b_old, b_fresh) (field increasing) or min(b_old, b_fresh) (decreasing), where b_fresh
   is the critical profile started from the new surface value.

Being history-driven, it accepts any H(t) sequence (minor loops, arbitrary sweeps), and it
is vectorised over a batch of parameter sets -- e.g. all temperatures of a chi_n(t) scan.
"""

from __future__ import annotations

import numpy as np

from .common import LoopResult, SIParams, initial_field, loop_field


def _g(b, B0):
    return np.sign(b) * (B0 * np.abs(b) + 0.5 * b * b)


def _ginv(y, B0):
    return np.sign(y) * (np.sqrt(B0 * B0 + 2.0 * np.abs(y)) - B0)


def _beq(h, Hc1):
    return np.sign(h) * np.maximum(np.abs(h) - Hc1, 0.0)


class CriticalState:
    """Batch of cylinders sharing the applied field; per-sample K, B0, dH, Hc1 (shape (N,))."""

    def __init__(self, K, B0, dH, Hc1, a: float, nr: int = 2001):
        col = lambda v: np.atleast_1d(np.asarray(v, dtype=float))[:, None]  # noqa: E731
        self.K, self.B0, self.dH, self.Hc1 = col(K), col(B0), col(dH), col(Hc1)
        n = max(self.K.shape[0], self.B0.shape[0], self.dH.shape[0], self.Hc1.shape[0])
        self.a = a
        self.r = np.linspace(0.0, a, nr)
        self.depth = a - self.r  # distance from the surface
        self.b = np.zeros((n, nr))
        self.bs = np.zeros((n, 1))
        # trapezoid weights of (2/a^2) * integral r dr
        w = np.full(nr, self.r[1] - self.r[0])
        w[0] = w[-1] = 0.5 * w[0]
        self.w = 2.0 * self.r * w / a**2

    def apply(self, H: float) -> np.ndarray:
        """Move the applied field to H; return mu0 M = <B> - H for every sample."""
        lo = _beq(H - self.dH, self.Hc1)
        hi = _beq(H + self.dH, self.Hc1)
        bs = np.clip(self.bs, lo, hi)
        up = bs > self.bs
        dn = bs < self.bs
        gs = _g(bs, self.B0)
        Kd = self.K * self.depth
        if up.any():
            fresh = _ginv(gs - Kd, self.B0)
            self.b = np.where(up, np.maximum(self.b, fresh), self.b)
        if dn.any():
            fresh = _ginv(gs + Kd, self.B0)
            self.b = np.where(dn, np.minimum(self.b, fresh), self.b)
        self.bs = bs
        return self.b @ self.w - H

    def sweep(self, Hs: np.ndarray) -> np.ndarray:
        """Apply a field sequence; returns M with shape (N, len(Hs))."""
        return np.stack([self.apply(h) for h in Hs], axis=-1)


def _ramp(H_end: float, step: float) -> np.ndarray:
    n = max(int(np.ceil(abs(H_end) / step)), 1)
    return np.linspace(0.0, H_end, n + 1)[1:]


def compute_loop(p: SIParams, nr: int = 2001) -> LoopResult:
    cs = CriticalState(p.K, p.B0, p.dH, p.Hc1, p.a, nr)
    Hi = initial_field(p)
    if p.Hdc > 0:  # virgin ramp 0 -> Hdc (monotonic, so the step size is irrelevant)
        cs.sweep(_ramp(p.Hdc, max(p.Hac, p.Hdc) / p.T))
    Mi = cs.sweep(Hi)[0]
    Hl = loop_field(p)
    Ml = cs.sweep(Hl)[0]
    return LoopResult(Hi, Mi, Hl, Ml, p.regime())


def loops_batch(K, B0, dH, Hc1, a: float, Hac: float, Hdc: float, T: int, nr: int = 1001):
    """Steady loops M(theta) for a batch of parameter sets -> shape (N, 2T+1)."""
    th = np.pi * np.arange(2 * T + 1) / T
    cs = CriticalState(K, B0, dH, Hc1, a, nr)
    if Hdc > 0:
        cs.sweep(_ramp(Hdc, max(Hac, Hdc) / T))
    cs.sweep(Hdc + Hac * np.sin(th[: T // 2 + 1]))
    return cs.sweep(Hdc + Hac * np.cos(th))
