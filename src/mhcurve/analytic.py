"""Closed-form M(H) of the modified Kim-Anderson model -- a line-by-line port of CHI999.

Every stage subroutine of the original N88-BASIC program (``*MED1D1``, ``*LOW1I4`` ...) is
kept as a method with the same name and the same B(H) expression, so the two can be compared
side by side. Equation numbers in the comments are those quoted in the BASIC source.
Heff(1..4) = H -/+ Hc1 -/+ dH; p_m = +/-2 mu0 k and q_m set the profiles (p_m x + q_m)^(1/2) - B0.

Deviations from the BASIC, all of which leave the physics untouched:
* M is stored at the index of the H that produced it (the BASIC stores SY(I+1) for H(I)).
* Every WHILE loop is bounded by the end of its half/full cycle, so no subscript overrun.
* sqrt of a negative number within round-off (> -1e-15 T^2) is clamped to zero.

One physics fix: for Hac <= dH (with HA > Hc1 + dH) CHI999 takes a wrong branch; see
``trapped``.
"""

from __future__ import annotations

import math

import numpy as np

from .common import LoopResult, SIParams, initial_field, loop_field


class AnalyticError(ValueError):
    pass


class _Kim:
    def __init__(self, p: SIParams):
        if p.K <= 0:
            raise AnalyticError("k must be positive")
        self.p = p
        self.K, self.A, self.H0 = p.K, p.a, p.B0
        self.HC, self.DH = p.Hc1, p.dH
        self.HA, self.HB = p.HA, p.HB
        self.T = p.T
        self.KA = self.K * self.A
        self.KA2 = 15 * self.KA**2
        self.HHA = p.Hstar  # H*  Eq.(81)
        self.HP = p.Hp  # Hp  Eq.(12)
        K = self.K
        self.P = [2 * K, -2 * K, -2 * K, 2 * K, -2 * K, -2 * K, 2 * K, 2 * K, 2 * K, 2 * K]
        self.Q = [0.0] * 10
        self.X = [0.0] * 10
        self.HEF1A = self.HEF2B = self.HEF3B = 0.0
        self.H = np.zeros(0)
        self.SY = np.zeros(0)
        self.i = 0

    # ---- DEF FN ----------------------------------------------------------
    def hef1(self, h):
        return h - self.HC - self.DH

    def hef2(self, h):
        return h - self.HC + self.DH

    def hef3(self, h):
        return h + self.HC + self.DH

    def hef4(self, h):
        return h + self.HC - self.DH

    def G(self, m: int, x: float) -> float:
        P, Q = self.P[m], self.Q[m]
        v = P * x + Q
        if v < 0:
            if v < -1e-15:
                raise AnalyticError(f"sqrt of negative argument in G{m} ({v:.3e})")
            v = 0.0
        return (3 * P * x - 2 * Q) * math.sqrt(v) ** 3 / self.KA2

    def fnq(self, m: int, h: float) -> float:
        H0, KA = self.H0, self.KA
        if m == 1:
            return (H0 + h) ** 2 + 2 * KA
        if m == 2:
            return 2 * H0**2 - (H0 - h) ** 2 + 2 * KA
        if m == 3:
            return (H0 - h) ** 2 - 2 * KA
        if m == 4:
            return (H0 - h) ** 2 + 2 * KA
        if m == 5:
            return 2 * H0**2 - (H0 + h) ** 2 + 2 * KA
        return (H0 + h) ** 2 - 2 * KA  # m = 0, 6, 7, 8, 9

    def setq(self, m: int, h: float) -> None:
        self.Q[m] = self.fnq(m, h)

    def fnx(self, m: int, h: float) -> float:
        A, K, H0 = self.A, self.K, self.H0
        if m in (0, 6):
            return A - ((H0 + h) ** 2 - H0**2) / 2 / K
        if m == 1:
            return A - ((H0 + self.HEF1A) ** 2 - (H0 + h) ** 2) / 4 / K
        if m == 2:
            return A - ((H0 + self.HEF1A) ** 2 + (H0 - h) ** 2 - 2 * H0**2) / 4 / K
        if m == 3:
            return A - ((H0 - h) ** 2 - H0**2) / 2 / K
        if m == 4:
            return A - ((H0 - self.HEF3B) ** 2 - (H0 - h) ** 2) / 4 / K
        if m in (5, 7):
            return A - ((H0 - self.HEF3B) ** 2 + (H0 + h) ** 2 - 2 * H0**2) / 4 / K
        if m == 8:
            return A + (H0**2 - (H0 + h) ** 2) / 4 / K
        if m == 9:
            return A - ((H0 + h) ** 2 - (H0 + self.HEF2B) ** 2) / 4 / K
        raise ValueError(m)

    def setx(self, m: int, h: float) -> float:
        self.X[m] = self.fnx(m, h)
        return self.X[m]

    @staticmethod
    def _sqrt(v: float) -> float:
        if v < 0:
            if v < -1e-15:
                raise AnalyticError(f"sqrt of negative argument ({v:.3e})")
            return 0.0
        return math.sqrt(v)

    # ---- loop helpers --------------------------------------------------------
    def _h(self) -> float:
        return self.H[self.i]

    def _put(self, B: float) -> None:
        self.SY[self.i] = B - self.H[self.i]
        self.i += 1

    def _dec(self) -> bool:  # still on the descending half (0..T)
        return self.i <= self.T

    def _asc(self) -> bool:  # still inside the period (0..2T)
        return self.i <= 2 * self.T

    # ---- *INI : initial magnetization ---------------------------------------
    def ini(self) -> None:
        G, A, H0 = self.G, self.A, self.H0
        n = self.T // 2
        self.i = 0
        while self.i <= n and self._h() < self.HC + self.DH:
            self._put(0.0)
        while self.i <= n and self._h() < self.HC + self.DH + self.HP:
            hef = self.hef1(self._h())
            x0 = self.setx(0, hef)
            self.setq(0, hef)
            self._put(G(0, A) - G(0, x0) - H0 * (A**2 - x0**2) / A**2)
        while self.i <= n:
            self.setq(0, self.hef1(self._h()))
            self._put(G(0, A) - G(0, 0) - H0)

    # ======================================================================
    # Case selection (line 2880 / 3880)
    # ======================================================================
    def loop(self) -> None:
        r = self.p.regime()
        if r != "B0" and self.HB >= self.HA - 2 * self.DH:
            return self.trapped(r)
        {"HI": self.hi, "MED": self.med, "LOW": self.low, "B0": self.b0}[r]()

    # ---- complete trapping (not in CHI999) ------------------------------------
    def trapped(self, r: str) -> None:
        """Hac <= dH: H never leaves stage 1 (HA - 2 dH <= H <= HA), so the flux stays
        completely trapped over the whole period and B = B(HA), the stage-1 value.

        CHI999 has no branch for this: the descending loop ends inside stage 1 and it jumps
        to 'H increasing 3', evaluating a profile x1(Heff(2)) that never formed.
        """
        G, A, H0 = self.G, self.A, self.H0
        self.HEF1A = self.hef1(self.HA)
        self.setq(0, self.HEF1A)
        if r == "LOW":
            x0 = self.setx(0, self.HEF1A)
            B = G(0, A) - G(0, x0) - H0 * (A**2 - x0**2) / A**2
        else:
            B = G(0, A) - G(0, 0) - H0
        self.i = 0
        while self._asc():
            self._put(B)

    # ---- *HI -----------------------------------------------------------------
    def hi(self) -> None:
        K4 = 4 * self.KA
        H0, HC, DH, HB = self.H0, self.HC, self.DH, self.HB
        self.HEF1A = self.hef1(self.HA)
        self.HHD = self._sqrt((H0 + self.HEF1A) ** 2 - K4) - H0  # Hprh-  Eq.(107)
        self.MED1D1()
        self.HI1D2()
        if self.i > self.T:
            return self._hi_inc6()
        self.HI1D3()
        if self.i > self.T:
            return self._hi_inc5()
        self.HI1D4()
        if self.i > self.T:
            return self._hi_inc4()
        self.MED1D5()
        if self.i > self.T:
            return self._hi_inc3()
        self.MED1D6()
        if HB >= -HC - DH - self.HHA:
            return self._hi_inc2()
        # H increasing 1
        self.HEF3B = self.hef3(HB)
        self.HHI = H0 - self._sqrt((H0 - self.HEF3B) ** 2 - K4)  # Hprh+  Eq.(110)
        self.setq(3, self.HEF3B)
        self.MED1I1()
        self.HI1I2()
        self.HI1I3()
        self.HI1I4()
        self.MED1I5()
        self.MED1I6()

    def _hi_inc2(self) -> None:
        H0 = self.H0
        self.HEF3B = self.hef3(self.HB)
        self.HMI = self._sqrt(4 * self.KA + 2 * H0**2 - (H0 - self.HEF3B) ** 2) - H0  # Eq.(90)
        self.setq(3, self.HEF3B)
        self.MED1I1()
        self.MED1I2()
        self.MED1I3()
        self.MED1I4()
        self.MED1I5()
        self.MED1I6()

    def _hi_inc3(self) -> None:
        H0 = self.H0
        self.HEF3B = self.hef3(self.HB)
        self.HMI = self._sqrt(4 * self.KA + 2 * H0**2 - (H0 - self.HEF3B) ** 2) - H0  # Eq.(90)
        self.HEF2B = self.hef2(self.HB)
        self.setx(3, self.HEF3B)
        self.setq(2, self.HEF3B)
        self.setq(3, self.HEF3B)
        self.MED2I1()
        self.MED2I2()
        self.MED2I3()
        self.MED2I4()
        self.MED2I5()
        self.MED1I6()

    def _hi_inc4(self) -> None:
        self.setq(1, 0.0)
        self.HI4I1()
        self.HI4I2()
        self.MED1I6()

    def _hi_inc5(self) -> None:
        H0 = self.H0
        self.HEF2B = self.hef2(self.HB)
        self.HHI2 = -H0 + self._sqrt((H0 + self.HEF2B) ** 2 + 4 * self.KA)  # Hprh++ Eq.(115)
        self.setq(1, self.HEF2B)
        self.HI5I1()
        self.HI5I2()
        self.MED1I6()

    def _hi_inc6(self) -> None:
        self.HEF2B = self.hef2(self.HB)
        self.setx(1, self.HEF2B)
        self.setq(1, self.HEF2B)
        self.MED5I1()
        self.MED5I2()

    # ---- *MED ----------------------------------------------------------------
    def med(self) -> None:
        H0, KA = self.H0, self.KA
        self.HEF1A = self.hef1(self.HA)
        self.HEF3B = self.hef3(self.HB)
        self.HMD = H0 - self._sqrt(4 * KA + 2 * H0**2 - (H0 + self.HEF1A) ** 2)  # Hprm- Eq.(83)
        self.HMI = self._sqrt(4 * KA + 2 * H0**2 - (H0 - self.HEF3B) ** 2) - H0  # Hprm+ Eq.(90)
        self.MED1D1()
        self.MED1D2()
        if self.i > self.T:
            return self._med_inc5()
        self.MED1D3()
        if self.i > self.T:
            return self._med_inc4()
        self.MED1D4()
        if self.i > self.T:
            return self._med_inc3()
        self.MED1D5()
        if self.i > self.T:
            return self._med_inc2()
        self.MED1D6()
        # H increasing 1
        self.setq(3, self.HEF3B)
        self.MED1I1()
        self.MED1I2()
        self.MED1I3()
        self.MED1I4()
        self.MED1I5()
        self.MED1I6()

    def _med_inc2(self) -> None:
        self.setx(3, self.HEF3B)
        self.setq(2, self.HEF3B)
        self.setq(3, self.HEF3B)
        self.MED2I1()
        self.MED2I2()
        self.MED2I3()
        self.MED2I4()
        self.MED2I5()
        self.MED1I6()

    def _med_inc3(self) -> None:
        self.setx(2, self.HEF3B)
        self.setx(3, self.HEF3B)
        self.setq(2, self.HEF3B)
        self.setq(3, self.HEF3B)
        self.MED3I1()
        self.MED3I2()
        self.MED3I3()
        self.MED3I4()
        self.MED3I5()

    def _med_inc4(self) -> None:
        self.setx(1, 0.0)
        self.setq(1, 0.0)
        self.MED4I1()
        self.MED4I2()

    def _med_inc5(self) -> None:
        self.HEF2B = self.hef2(self.HB)
        self.setx(1, self.HEF2B)
        self.setq(1, self.HEF2B)
        self.MED5I1()
        self.MED5I2()

    # ---- *LOW ----------------------------------------------------------------
    def low(self) -> None:
        self.HEF1A = self.hef1(self.HA)
        self.setx(0, self.HEF1A)
        self.setq(0, self.HEF1A)
        self.LOW1D1()
        self.LOW1D2()
        if self.i > self.T:
            return self._low_inc3()
        self.LOW1D3()
        if self.i > self.T:
            return self._low_inc2()
        self.LOW1D4()
        # H increasing 1
        self.HEF3B = self.hef3(self.HB)
        self.setx(2, self.HEF3B)
        self.setx(3, self.HEF3B)
        self.setq(2, self.HEF3B)
        self.setq(3, self.HEF3B)
        self.LOW1I1()
        self.LOW1I2()
        self.LOW1I3()
        self.LOW1I4()
        self.LOW1I5()

    def _low_inc2(self) -> None:
        self.setx(1, 0.0)
        self.setq(1, 0.0)
        self.LOW2I1()
        self.LOW2I2()

    def _low_inc3(self) -> None:
        self.HEF2B = self.hef2(self.HB)
        self.setx(1, self.HEF2B)
        self.setq(1, self.HEF2B)
        self.LOW3I1()
        self.LOW3I2()

    # ---- *B0 -----------------------------------------------------------------
    def b0(self) -> None:
        self.i = 0
        while self._asc():
            self._put(0.0)

    # ======================================================================
    # High-Ha stages
    # ======================================================================
    def HI1D2(self):  # stage 2 of IV C 1, Eq.(80)
        G, A, X = self.G, self.A, self.X
        while self._dec() and self._h() > self.HHD + self.HC - self.DH:
            hef2 = self.hef2(self._h())
            self.setx(1, hef2)
            self.setq(1, hef2)
            self._put(G(1, A) - G(1, X[1]) + G(0, X[1]) - G(0, 0) - self.H0)

    def HI1D3(self):  # Eq.(108)
        G, A = self.G, self.A
        while self._dec() and self._h() > self.HC - self.DH:
            self.setq(1, self.hef2(self._h()))
            self._put(G(1, A) - G(1, 0) - self.H0)

    def HI1D4(self):  # Eq.(109)
        G, A = self.G, self.A
        self.setq(1, 0.0)
        while self._dec() and self._h() > -self.HC - self.DH:
            self._put(G(1, A) - G(1, 0) - self.H0)

    def HI1I2(self):  # Eq.(88)
        G, A, X = self.G, self.A, self.X
        while self._asc() and self._h() < -self.HC + self.DH + self.HHI:
            hef4 = self.hef4(self._h())
            self.setx(4, hef4)
            self.setq(4, hef4)
            self._put(-G(4, A) + G(4, X[4]) - G(3, X[4]) + G(3, 0) + self.H0)

    def HI1I3(self):  # Eq.(110)
        G, A = self.G, self.A
        while self._asc() and self._h() < -self.HC + self.DH:
            self.setq(4, self.hef4(self._h()))
            self._put(-G(4, A) + G(4, 0) + self.H0)

    def HI1I4(self):  # Eq.(111)
        G, A = self.G, self.A
        self.setq(4, 0.0)
        while self._asc() and self._h() < self.HC + self.DH:
            self._put(-G(4, A) + G(4, 0) + self.H0)

    def HI4I1(self):  # Eq.(109)
        G, A = self.G, self.A
        while self._asc() and self._h() < self.HC + self.DH:
            self._put(G(1, A) - G(1, 0) - self.H0)

    def HI4I2(self):  # Eq.(113)
        G, A, X = self.G, self.A, self.X
        while self._asc() and self._h() < self.HC + self.DH + self.HHA:
            hef1 = self.hef1(self._h())
            self.setx(8, hef1)
            self.setq(8, hef1)
            self._put(G(8, A) - G(8, X[8]) + G(1, X[8]) - G(1, 0) - self.H0)

    def HI5I1(self):  # Eq.(114)
        G, A = self.G, self.A
        while self._asc() and self._h() < self.HB + 2 * self.DH:
            self._put(G(1, A) - G(1, 0) - self.H0)

    def HI5I2(self):  # Eq.(116)
        G, A, X = self.G, self.A, self.X
        while self._asc() and self._h() < self.HC + self.DH + self.HHI2:
            hef1 = self.hef1(self._h())
            self.setx(9, hef1)
            self.setq(9, hef1)
            self._put(G(9, A) - G(9, X[9]) + G(1, X[9]) - G(1, 0) - self.H0)

    # ======================================================================
    # Medium-Ha stages
    # ======================================================================
    def MED1D1(self):  # Eq.(79)
        G, A = self.G, self.A
        self.HEF1A = self.hef1(self.HA)
        self.setq(0, self.HEF1A)
        self.i = 0
        while self._dec() and self._h() > self.HA - 2 * self.DH:
            self._put(G(0, A) - G(0, 0) - self.H0)

    def MED1D2(self):  # Eq.(80)
        G, A, X = self.G, self.A, self.X
        while self._dec() and self._h() > self.HC - self.DH:
            hef2 = self.hef2(self._h())
            self.setx(1, hef2)
            self.setq(1, hef2)
            self._put(G(1, A) - G(1, X[1]) + G(0, X[1]) - G(0, 0) - self.H0)

    def MED1D3(self):  # Eq.(82)
        G, A, X = self.G, self.A, self.X
        self.setx(1, 0.0)
        self.setq(1, 0.0)
        while self._dec() and self._h() > -self.HC - self.DH:
            self._put(G(1, A) - G(1, X[1]) + G(0, X[1]) - G(0, 0) - self.H0)

    def MED1D4(self):  # Eq.(84)
        G, A, X, H0 = self.G, self.A, self.X, self.H0
        while self._dec() and self._h() > -self.HC - self.DH + self.HMD:
            hef3 = self.hef3(self._h())
            self.setx(2, hef3)
            self.setx(3, hef3)
            self.setq(2, hef3)
            self.setq(3, hef3)
            x2, x3 = X[2], X[3]
            self._put(
                -G(3, A) + G(3, x3) + G(2, x3) - G(2, x2) + G(0, x2) - G(0, 0)
                + H0 * (A**2 - 2 * x3**2) / A**2
            )

    def MED1D5(self):  # Eq.(85)
        G, A, X, H0 = self.G, self.A, self.X, self.H0
        while self._dec() and self._h() > -self.HC - self.DH - self.HP:
            hef3 = self.hef3(self._h())
            self.setx(3, hef3)
            self.setq(2, hef3)
            self.setq(3, hef3)
            x3 = X[3]
            self._put(-G(3, A) + G(3, x3) + G(2, x3) - G(2, 0) + H0 * (A**2 - 2 * x3**2) / A**2)

    def MED1D6(self):  # Eq.(86)
        G, A = self.G, self.A
        while self._dec():
            self.setq(3, self.hef3(self._h()))
            self._put(-G(3, A) + G(3, 0) + self.H0)

    def MED1I1(self):  # Eq.(87)
        G, A = self.G, self.A
        while self._asc() and self._h() < self.HB + 2 * self.DH:
            self._put(-G(3, A) + G(3, 0) + self.H0)

    def MED1I2(self):  # Eq.(88)
        G, A, X = self.G, self.A, self.X
        while self._asc() and self._h() < -self.HC + self.DH:
            hef4 = self.hef4(self._h())
            self.setx(4, hef4)
            self.setq(4, hef4)
            self._put(-G(4, A) + G(4, X[4]) - G(3, X[4]) + G(3, 0) + self.H0)

    def MED1I3(self):  # Eq.(89)
        G, A, X = self.G, self.A, self.X
        self.setx(4, 0.0)
        self.setq(4, 0.0)
        while self._asc() and self._h() < self.HC + self.DH:
            self._put(-G(4, A) + G(4, X[4]) - G(3, X[4]) + G(3, 0) + self.H0)

    def MED1I4(self):  # Eq.(91)
        G, A, X, H0 = self.G, self.A, self.X, self.H0
        while self._asc() and self._h() < self.HC + self.DH + self.HMI:
            hef1 = self.hef1(self._h())
            self.setx(5, hef1)
            self.setx(6, hef1)
            self.setq(5, hef1)
            self.setq(6, hef1)
            x5, x6 = X[5], X[6]
            self._put(
                G(6, A) - G(6, x6) - G(5, x6) + G(5, x5) - G(3, x5) + G(3, 0)
                - H0 * (A**2 - 2 * x6**2) / A**2
            )

    def MED1I5(self):  # Eq.(92)
        G, A, X, H0 = self.G, self.A, self.X, self.H0
        while self._asc() and self._h() < self.HC + self.DH + self.HP:
            hef1 = self.hef1(self._h())
            self.setx(6, hef1)
            self.setq(5, hef1)
            self.setq(6, hef1)
            x6 = X[6]
            self._put(G(6, A) - G(6, x6) - G(5, x6) + G(5, 0) - H0 * (A**2 - 2 * x6**2) / A**2)

    def MED1I6(self):  # Eq.(93)
        G, A = self.G, self.A
        while self._asc():
            self.setq(6, self.hef1(self._h()))
            self._put(G(6, A) - G(6, 0) - self.H0)

    def MED2I1(self):  # Eq.(94)
        G, A, X, H0 = self.G, self.A, self.X, self.H0
        x3 = X[3]
        while self._asc() and self._h() < self.HB + 2 * self.DH:
            self._put(-G(3, A) + G(3, x3) + G(2, x3) - G(2, 0) + H0 * (A**2 - 2 * x3**2) / A**2)

    def _med2_tail(self, x4: float) -> float:
        G, A, H0, x3 = self.G, self.A, self.H0, self.X[3]
        return (
            -G(4, A) + G(4, x4) - G(3, x4) + G(3, x3) + G(2, x3) - G(2, 0)
            + H0 * (A**2 - 2 * x3**2) / A**2
        )

    def MED2I2(self):  # Eq.(96)
        while self._asc() and self._h() < -self.HC + self.DH:
            hef4 = self.hef4(self._h())
            self.setx(4, hef4)
            self.setq(4, hef4)
            self._put(self._med2_tail(self.X[4]))

    def MED2I3(self):  # Eq.(96)
        self.setx(4, 0.0)
        self.setq(4, 0.0)
        while self._asc() and self._h() < self.HC + self.DH:
            self._put(self._med2_tail(self.X[4]))

    def MED2I4(self):  # Eq.(97)
        G, A, X, H0 = self.G, self.A, self.X, self.H0
        while self._asc() and self._h() < -self.HB:
            hef1 = self.hef1(self._h())
            self.setx(5, hef1)
            self.setx(6, hef1)
            self.setq(5, hef1)
            self.setq(6, hef1)
            x3, x5, x6 = X[3], X[5], X[6]
            self._put(
                G(6, A) - G(6, x6) - G(5, x6) + G(5, x5) - G(3, x5) + G(3, x3) + G(2, x3) - G(2, 0)
                - H0 * (A**2 + 2 * (x3**2 - x6**2)) / A**2
            )

    def MED2I5(self):  # Eq.(98)
        G, A, X = self.G, self.A, self.X
        while self._asc() and self._h() < self.HC + self.DH + self.HMI:
            hef1 = self.hef1(self._h())
            self.setx(7, hef1)
            self.setq(7, hef1)
            self._put(G(7, A) - G(7, X[7]) + G(2, X[7]) - G(2, 0) - self.H0)

    def _med3_base(self) -> float:
        G, A, H0, x2, x3 = self.G, self.A, self.H0, self.X[2], self.X[3]
        return (
            G(3, x3) + G(2, x3) - G(2, x2) + G(0, x2) - G(0, 0)
            + H0 * (A**2 - 2 * x3**2) / A**2
        )

    def MED3I1(self):  # Eq.(99)
        G, A = self.G, self.A
        while self._asc() and self._h() < self.HB + 2 * self.DH:
            self._put(-G(3, A) + self._med3_base())

    def MED3I2(self):  # Eq.(100)
        G, A, X = self.G, self.A, self.X
        while self._asc() and self._h() < -self.HC + self.DH:
            hef4 = self.hef4(self._h())
            self.setx(4, hef4)
            self.setq(4, hef4)
            self._put(-G(4, A) + G(4, X[4]) - G(3, X[4]) + self._med3_base())

    def MED3I3(self):  # Eq.(101)
        G, A, X = self.G, self.A, self.X
        self.setx(4, 0.0)
        self.setq(4, 0.0)
        while self._asc() and self._h() < self.HC + self.DH:
            self._put(-G(4, A) + G(4, X[4]) - G(3, X[4]) + self._med3_base())

    def MED3I4(self):  # Eq.(102)
        G, A, X, H0 = self.G, self.A, self.X, self.H0
        while self._asc() and self._h() < -self.HB:
            hef1 = self.hef1(self._h())
            self.setx(5, hef1)
            self.setx(6, hef1)
            self.setq(5, hef1)
            self.setq(6, hef1)
            x2, x3, x5, x6 = X[2], X[3], X[5], X[6]
            self._put(
                G(6, A) - G(6, x6) - G(5, x6) + G(5, x5) - G(3, x5) + G(3, x3) + G(2, x3)
                - G(2, x2) + G(0, x2) - G(0, 0)
                - H0 * (A**2 + 2 * x3**2 - 2 * x6**2) / A**2
            )

    def MED3I5(self):  # Eq.(103)
        G, A, X = self.G, self.A, self.X
        while self._asc():
            hef1 = self.hef1(self._h())
            self.setx(7, hef1)
            self.setq(7, hef1)
            x2, x7 = X[2], X[7]
            self._put(G(7, A) - G(7, x7) + G(2, x7) - G(2, x2) + G(0, x2) - G(0, 0) - self.H0)

    def MED4I1(self):  # Eq.(82)
        G, A, X = self.G, self.A, self.X
        while self._asc() and self._h() < self.HC + self.DH:
            self._put(G(1, A) - G(1, X[1]) + G(0, X[1]) - G(0, 0) - self.H0)

    def MED4I2(self):  # Eq.(104)
        G, A, X = self.G, self.A, self.X
        while self._asc():
            hef1 = self.hef1(self._h())
            self.setx(8, hef1)
            self.setq(8, hef1)
            x1, x8 = X[1], X[8]
            self._put(G(8, A) - G(8, x8) + G(1, x8) - G(1, x1) + G(0, x1) - G(0, 0) - self.H0)

    def MED5I1(self):  # Eq.(105)
        G, A, X = self.G, self.A, self.X
        while self._asc() and self._h() < self.HB + 2 * self.DH:
            self._put(G(1, A) - G(1, X[1]) + G(0, X[1]) - G(0, 0) - self.H0)

    def MED5I2(self):  # Eq.(106)
        G, A, X = self.G, self.A, self.X
        while self._asc():
            hef1 = self.hef1(self._h())
            self.setx(9, hef1)
            self.setq(9, hef1)
            x1, x9 = X[1], X[9]
            self._put(G(9, A) - G(9, x9) + G(1, x9) - G(1, x1) + G(0, x1) - G(0, 0) - self.H0)

    # ======================================================================
    # Low-Ha stages (the virgin front x0 at HA stays inside the sample)
    # ======================================================================
    def _w0(self) -> float:
        """-B0 (a^2 - x0^2)/a^2 -- the untouched virgin part."""
        A, x0 = self.A, self.X[0]
        return -self.H0 * (A**2 - x0**2) / A**2

    def LOW1D1(self):  # Eq.(25)
        G, A, x0 = self.G, self.A, self.X[0]
        self.i = 0
        while self._dec() and self._h() > self.HA - 2 * self.DH:
            self._put(G(0, A) - G(0, x0) + self._w0())

    def LOW1D2(self):  # Eq.(29)
        G, A, X = self.G, self.A, self.X
        while self._dec() and self._h() > self.HC - self.DH:
            hef2 = self.hef2(self._h())
            self.setx(1, hef2)
            self.setq(1, hef2)
            self._put(G(1, A) - G(1, X[1]) + G(0, X[1]) - G(0, X[0]) + self._w0())

    def LOW1D3(self):
        G, A, X = self.G, self.A, self.X
        self.setx(1, 0.0)
        self.setq(1, 0.0)
        while self._dec() and self._h() > -self.HC - self.DH:
            self._put(G(1, A) - G(1, X[1]) + G(0, X[1]) - G(0, X[0]) + self._w0())

    def _low1_base(self) -> float:
        G, A, H0, X = self.G, self.A, self.H0, self.X
        x0, x2, x3 = X[0], X[2], X[3]
        return (
            G(3, x3) + G(2, x3) - G(2, x2) + G(0, x2) - G(0, x0)
            + H0 * (A**2 + x0**2 - 2 * x3**2) / A**2
        )

    def LOW1D4(self):
        G, A = self.G, self.A
        while self._dec():
            hef3 = self.hef3(self._h())
            self.setx(2, hef3)
            self.setx(3, hef3)
            self.setq(2, hef3)
            self.setq(3, hef3)
            self._put(-G(3, A) + self._low1_base())

    def LOW1I1(self):  # Eq.(47)
        G, A = self.G, self.A
        while self._asc() and self._h() < self.HB + 2 * self.DH:
            self._put(-G(3, A) + self._low1_base())

    def LOW1I2(self):  # Eq.(51)
        G, A, X = self.G, self.A, self.X
        while self._asc() and self._h() < -self.HC + self.DH:
            hef4 = self.hef4(self._h())
            self.setx(4, hef4)
            self.setq(4, hef4)
            self._put(-G(4, A) + G(4, X[4]) - G(3, X[4]) + self._low1_base())

    def LOW1I3(self):  # Eq.(55)
        G, A, X = self.G, self.A, self.X
        self.setx(4, 0.0)
        self.setq(4, 0.0)
        while self._asc() and self._h() < self.HC + self.DH:
            self._put(-G(4, A) + G(4, X[4]) - G(3, X[4]) + self._low1_base())

    def LOW1I4(self):  # Eq.(62)
        G, A, X, H0 = self.G, self.A, self.X, self.H0
        while self._asc() and self._h() < -self.HB:
            hef1 = self.hef1(self._h())
            self.setx(5, hef1)
            self.setx(6, hef1)
            self.setq(5, hef1)
            self.setq(6, hef1)
            x0, x2, x3, x5, x6 = X[0], X[2], X[3], X[5], X[6]
            self._put(
                G(6, A) - G(6, x6) - G(5, x6) + G(5, x5) - G(3, x5) + G(3, x3) + G(2, x3)
                - G(2, x2) + G(0, x2) - G(0, x0)
                - H0 * (A**2 - x0**2 + 2 * (x3**2 - x6**2)) / A**2
            )

    def LOW1I5(self):  # Eq.(66)
        G, A, X = self.G, self.A, self.X
        while self._asc():
            hef1 = self.hef1(self._h())
            self.setx(7, hef1)
            self.setq(7, hef1)
            x0, x2, x7 = X[0], X[2], X[7]
            self._put(G(7, A) - G(7, x7) + G(2, x7) - G(2, x2) + G(0, x2) - G(0, x0) + self._w0())

    def LOW2I1(self):  # Eq.(33)
        G, A, X = self.G, self.A, self.X
        while self._asc() and self._h() < self.HC + self.DH:
            self._put(G(1, A) - G(1, X[1]) + G(0, X[1]) - G(0, X[0]) + self._w0())

    def LOW2I2(self):  # Eq.(70)
        G, A, X = self.G, self.A, self.X
        while self._asc():
            hef1 = self.hef1(self._h())
            self.setx(8, hef1)
            self.setq(8, hef1)
            x0, x1, x8 = X[0], X[1], X[8]
            self._put(G(8, A) - G(8, x8) + G(1, x8) - G(1, x1) + G(0, x1) - G(0, x0) + self._w0())

    def LOW3I1(self):  # Eq.(74)
        G, A, X = self.G, self.A, self.X
        while self._asc() and self._h() < self.HB + 2 * self.DH:
            self._put(G(1, A) - G(1, X[1]) + G(0, X[1]) - G(0, X[0]) + self._w0())

    def LOW3I2(self):  # Eq.(78)
        G, A, X = self.G, self.A, self.X
        while self._asc():
            hef1 = self.hef1(self._h())
            self.setx(9, hef1)
            self.setq(9, hef1)
            x0, x1, x9 = X[0], X[1], X[9]
            self._put(G(9, A) - G(9, x9) + G(1, x9) - G(1, x1) + G(0, x1) - G(0, x0) + self._w0())


def compute_loop(p: SIParams) -> LoopResult:
    """Initial magnetization (*INI) and one full hysteresis period (*HI/*MED/*LOW/*B0)."""
    m = _Kim(p)
    m.H = initial_field(p)
    m.SY = np.zeros_like(m.H)
    m.ini()
    H_init, M_init = m.H, m.SY

    m = _Kim(p)
    m.H = loop_field(p)
    m.SY = np.full_like(m.H, np.nan)
    m.loop()
    return LoopResult(H_init, M_init, m.H, m.SY, p.regime())
