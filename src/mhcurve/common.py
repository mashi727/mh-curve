"""Parameters, applied-field waveforms and harmonic analysis shared by both engines.

Units inside the engines are SI with fields expressed as mu0*H [T]:
K = mu0*k [T^2/m], B0, dH, Hc1, Hac, Hdc [T], a [m]. The UI works in mT / cm.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

MU0 = 4e-7 * np.pi


def reduced_to_kb0(p: float, hp: float, a: float) -> tuple[float, float]:
    """(k [N/m^3], B0 [mT]) from p = (2 mu0 k a)^(1/2) / B0 and Hp [mT] , a [cm]."""
    b0 = hp / (np.sqrt(1.0 + p * p) - 1.0)
    k = (p * b0 * 1e-3) ** 2 / (2 * MU0 * a * 1e-2)
    return float(k), float(b0)


# default: p = 1000 with Hp = 1 mT, so mT axes read directly as H/Hp and M/Hp
_K_DEF, _B0_DEF = reduced_to_kb0(1000.0, 1.0, 0.1)


@dataclass(frozen=True)
class Params:
    """Material and excitation parameters (UI units).

    Defaults: M(H) with Hdc = 0, p = 1000, dH = 0.1 Hp, Hc1 = 0.15 Hp,
    Hac = 4 Hp -- with Hp = 1 mT and a = 0.1 cm (the CHI999 default radius).
    """

    k: float = _K_DEF  # [N/m^3]   Jc(B) = k / (B0 + |B|)
    b0: float = _B0_DEF  # [mT]      B0 (named H0 in CHI999)
    dh: float = 0.1  # [mT]      surface barrier Delta H
    hc1: float = 0.15  # [mT]      lower critical field
    hac: float = 4.0  # [mT]
    hdc: float = 0.0  # [mT]
    a: float = 0.1  # [cm]      cylinder radius
    n_half: int = 180  # points per half cycle (T% in CHI999, even)

    @property
    def si(self) -> "SIParams":
        return SIParams(
            K=MU0 * self.k,
            B0=self.b0 * 1e-3,
            dH=self.dh * 1e-3,
            Hc1=self.hc1 * 1e-3,
            Hac=self.hac * 1e-3,
            Hdc=self.hdc * 1e-3,
            a=self.a * 1e-2,
            T=self.n_half,
        )

    def at_t(self, t: float, exps: "Exponents") -> "Params":
        """Temperature-scaled parameters X(t) = X_max (1-t)^E (CHI999 *KIM3)."""
        f = 1.0 - t
        return replace(
            self,
            k=self.k * f**exps.k,
            b0=self.b0 * f**exps.b0,
            dh=self.dh * f**exps.dh,
            hc1=self.hc1 * f**exps.hc1,
        )


@dataclass(frozen=True)
class Exponents:
    """X(t) = X(0)(1 - t)^E exponents; Hc1 is irrelevant for the default Hc1 = 0."""

    k: float = 3.0
    b0: float = 1.5
    dh: float = 1.5
    hc1: float = 1.0


@dataclass(frozen=True)
class SIParams:
    K: float
    B0: float
    dH: float
    Hc1: float
    Hac: float
    Hdc: float
    a: float
    T: int

    @property
    def HA(self) -> float:
        return self.Hdc + self.Hac

    @property
    def HB(self) -> float:
        return self.Hdc - self.Hac

    @property
    def Hp(self) -> float:
        """Full-penetration field Hp (CHI999 Eq.(12))."""
        return np.sqrt(self.B0**2 + 2 * self.K * self.a) - self.B0

    @property
    def Hstar(self) -> float:
        """Field at which the reverse current reaches the centre, H* (CHI999 Eq.(81))."""
        return np.sqrt(self.B0**2 + 4 * self.K * self.a) - self.B0

    def regime(self) -> str:
        """HA classification (B0 / LOW / MED / HI)."""
        th = self.Hc1 + self.dH
        if self.HA < th:
            return "B0"
        if self.HA < th + self.Hp:
            return "LOW"
        if self.HA < th + self.Hstar:
            return "MED"
        return "HI"


REGIME_LABEL = {
    "B0": "完全反磁性 (HA < Hc1+ΔH)",
    "LOW": "low-HA (未完全侵入)",
    "MED": "medium-HA",
    "HI": "high-HA",
}


def initial_field(p: SIParams) -> np.ndarray:
    """Virgin branch H = Hdc + Hac sin(theta), theta = 0 .. pi/2 (T/2+1 points)."""
    th = np.pi * np.arange(p.T // 2 + 1) / p.T
    return p.Hdc + p.Hac * np.sin(th)


def loop_field(p: SIParams) -> np.ndarray:
    """One period H = Hdc + Hac cos(theta), theta = 0 .. 2 pi (2T+1 points)."""
    th = np.pi * np.arange(2 * p.T + 1) / p.T
    return p.Hdc + p.Hac * np.cos(th)


@dataclass
class LoopResult:
    H_init: np.ndarray  # [T]
    M_init: np.ndarray  # mu0 M [T]
    H_loop: np.ndarray
    M_loop: np.ndarray
    regime: str


def harmonics(M_loop: np.ndarray, Hac: float, nmax: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """chi_n = chi_n' - i chi_n'' with M = Hac * sum(chi_n' cos n wt + chi_n'' sin n wt).

    Trapezoidal rule over the 2T+1 samples, identical to CHI999 *FOURIER/*INTEGRATION.
    Works on the last axis, so a stack of loops (Nt, 2T+1) is accepted.
    """
    npts = M_loop.shape[-1]
    T = (npts - 1) // 2
    th = np.pi * np.arange(npts) / T
    w = np.ones(npts)
    w[0] = w[-1] = 0.5
    n = np.arange(1, nmax + 1)[:, None]
    cos = np.cos(n * th) * w
    sin = np.sin(n * th) * w
    re = M_loop @ cos.T / (T * Hac)
    im = M_loop @ sin.T / (T * Hac)
    return re, im
