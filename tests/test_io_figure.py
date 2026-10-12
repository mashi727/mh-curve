from dataclasses import replace

import numpy as np

import pytest

from mhcurve import io, presets
from mhcurve.common import Params


def test_default_is_p1000_d():
    name = presets.DEFAULT_PRESET
    assert name == presets.mh_key(1000.0, "d")
    d = presets.get(name)
    assert io.RunSpec().params == d.params
    si = d.params.si
    assert (2 * si.K * si.a) ** 0.5 / si.B0 == pytest.approx(1000.0, rel=1e-12)
    assert si.Hp * 1e3 == pytest.approx(1.0, rel=1e-12)
    assert (si.dH, si.Hc1, si.Hdc, si.Hac) == (pytest.approx(1e-4), pytest.approx(1.5e-4), 0.0,
                                               pytest.approx(4e-3))


def test_p1000_d_values():
    """p = 1000, (d), Hac = 4Hp: peak M/Hp ~ 0.93, M/Hp at H = HA ~ -0.30 (published curve)."""
    ds = io.compute(presets.get(presets.DEFAULT_PRESET))
    loop = ds.columns["branch"] == "loop"
    M = ds.columns["M_mT"][loop]
    assert M.max() == pytest.approx(0.93, abs=0.03)
    assert M[0] == pytest.approx(-0.30, abs=0.03)


@pytest.mark.parametrize("panel, peak", [("a", 0.60), ("b", 0.70), ("c", 0.75), ("d", 0.85)])
def test_p10_peak(panel, peak):
    """Maximum M/Hp of the Hac = 4Hp loop, p = 10 (read off the published curves)."""
    ds = io.compute(presets.get(presets.mh_key(10.0, panel)))
    loop = ds.columns["branch"] == "loop"
    assert ds.columns["M_mT"][loop].max() == pytest.approx(peak, abs=0.03)



@pytest.mark.parametrize("panel, m_at_ha", [("a", -0.33), ("b", -0.43), ("c", -0.48), ("d", -0.58)])
def test_p01_saturation(panel, m_at_ha):
    """M/Hp at H = HA = 4Hp, p = 0.1 (read off the published curves; Hp = 1 mT)."""
    name = presets.mh_key(0.1, panel)
    ds = io.compute(presets.get(name))
    loop = ds.columns["branch"] == "loop"
    assert ds.columns["M_mT"][loop][0] == pytest.approx(m_at_ha, abs=0.02)


def test_spec_and_dataset_roundtrip(tmp_path):
    spec = io.RunSpec(kind="mh", engine="numeric", name="s1",
                      params=replace(Params(), hac=45.0, hdc=12.5, n_half=120), t=0.3)
    io.save_spec(spec, tmp_path / "s1.toml")
    back = io.load_spec(tmp_path / "s1.toml")
    assert back == spec

    ds = io.compute(spec)
    io.save_dataset(ds, tmp_path / "s1.csv")
    ds2 = io.load_dataset(tmp_path / "s1.csv")
    assert ds2.kind == "mh"
    assert ds2.spec == spec
    np.testing.assert_allclose(ds2.columns["M_mT"], ds.columns["M_mT"], rtol=1e-9)
    assert list(ds2.columns["branch"][:2]) == ["init", "init"]


def test_chi_scan_engines_agree():
    spec = io.RunSpec(kind="chi", params=replace(Params(), hac=5.0, n_half=120),
                      tmin=0.0, tmax=0.95, tstep=0.05)
    a = io.compute(spec)
    n = io.compute(replace(spec, engine="numeric"))
    for c in ("chi1_re", "chi1_im", "chi3_re", "chi3_im"):
        np.testing.assert_allclose(a.columns[c], n.columns[c], atol=2e-3)


def test_chi_example_engines_agree():
    """chi_n(t) film example; 0.59 < t < 0.66 is the Hac < dH(t) trapping window."""
    base = presets.get(presets.CHI_EXAMPLE)
    spec = replace(base, tmin=0.5, tmax=0.99, tstep=0.005)
    a = io.compute(spec)
    n = io.compute(replace(spec, engine="numeric"))
    for c in a.columns:
        if c != "t":
            np.testing.assert_allclose(a.columns[c], n.columns[c], atol=5e-4, err_msg=c)


def test_pdf_export(tmp_path):
    from mhcurve import figure

    ds = io.compute(io.RunSpec(kind="mh", params=replace(Params(), hac=60.0, k=1e6)))
    out = figure.mh_pdf([ds, ds], tmp_path / "f.pdf", figure.FigureStyle(), flip=True)
    assert out.read_bytes()[:5] == b"%PDF-"
    chi = io.compute(io.RunSpec(kind="chi", tstep=0.1))
    out = figure.chi_pdf([chi], tmp_path / "c.pdf", figure.FigureStyle(), 1, True, True)
    assert out.stat().st_size > 1000
