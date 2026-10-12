"""The closed-form port and the independent numerical solver must give the same M(H)."""

from dataclasses import replace

import numpy as np
import pytest

from mhcurve import analytic, numeric
from mhcurve.common import Params, harmonics

BASE = Params(k=1.0e6, b0=10.0, dh=1.0, hc1=5.0, a=0.1)
# Hp ~ 41.2 mT, H* ~ 61.7 mT, Hc1 + dH = 6 mT


def _cases():
    for hac in (3.0, 20.0, 45.0, 60.0, 100.0, 200.0):
        for hdc in (0.0, 4.0, 15.0, 40.0, 120.0):
            yield replace(BASE, hac=hac, hdc=hdc)
    for dh, hc1 in ((0.0, 0.0), (3.0, 0.0), (0.0, 8.0)):
        for hac in (20.0, 60.0, 100.0):
            yield replace(BASE, hac=hac, dh=dh, hc1=hc1, hdc=10.0)
    # Hac <= dH: complete trapping over the period (CHI999 took a wrong branch here)
    for hdc in (2.0, 10.0, 60.0, 100.0):  # LOW, LOW, MED, HI
        for hac in (4.0, 5.0):
            yield replace(BASE, hac=hac, hdc=hdc, dh=5.0, hc1=0.0)


def _id(p):
    return f"Hac{p.hac:g}-Hdc{p.hdc:g}-dH{p.dh:g}-Hc1{p.hc1:g}"


@pytest.mark.parametrize("p", list(_cases()), ids=_id)
def test_loop_agrees(p):
    si = p.si
    ra = analytic.compute_loop(si)
    rn = numeric.compute_loop(si, nr=4001)
    scale = si.Hac
    np.testing.assert_allclose(ra.M_init, rn.M_init, atol=2e-4 * scale)
    np.testing.assert_allclose(ra.M_loop, rn.M_loop, atol=2e-4 * scale)


def test_regimes_covered():
    assert {p.si.regime() for p in _cases()} == {"B0", "LOW", "MED", "HI"}


def test_full_shielding_chi1_is_minus_one():
    p = replace(BASE, hac=3.0).si  # below Hc1 + dH
    re, im = harmonics(analytic.compute_loop(p).M_loop, p.Hac)
    assert re[0] == pytest.approx(-1.0, abs=1e-9)
    assert abs(im[0]) < 1e-9
