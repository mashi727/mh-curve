from dataclasses import replace
from pathlib import Path

import pytest

from mhcurve import figure, io, presets
from mhcurve.palette import SERIES
from mhcurve.view import Layer, PdfSettings, ViewState, load_view, next_style, save_view, tex_to_html


def test_view_roundtrip(tmp_path):
    (tmp_path / "sub").mkdir()
    st = ViewState(
        layers=[
            Layer(tmp_path / "a.csv", r"$H_\mathrm{ac} = H_\mathrm{p}$", "#2a78d6", "a", True, 0),
            Layer(tmp_path / "sub" / "b.csv", "it's \"quoted\"", "#eb6834", "c", False, 1),
        ],
        normalize=True, flip=True, harmonic=3, show_re=False, show_im=True,
        pdf=PdfSettings(width_mm=178.0, arrangement="grid", panel_labels=False),
    )
    p = save_view(st, tmp_path / "fig.view.toml")
    assert "file = 'sub/b.csv'" in p.read_text()  # stored relative to the view file
    back = load_view(p)
    assert back.layers[0].path == (tmp_path / "a.csv").resolve()
    assert [l.label for l in back.layers] == [l.label for l in st.layers]
    assert (back.normalize, back.flip, back.harmonic, back.show_re) == (True, True, 3, False)
    assert back.pdf == st.pdf
    assert [(l.panel, l.visible, l.dash, l.color) for l in back.layers] == \
        [(l.panel, l.visible, l.dash, l.color) for l in st.layers]


def test_next_style_fixed_order_then_dash():
    lays = []
    for i in range(10):
        c, d = next_style(lays)
        lays.append(Layer(Path(f"{i}.csv"), "", c, dash=d))
    assert [l.color for l in lays[:8]] == SERIES
    assert (lays[8].color, lays[8].dash) == (SERIES[0], 1)
    # removing a curve frees its slot; survivors keep their colours
    del lays[2]
    assert next_style(lays) == (SERIES[2], 0)


@pytest.mark.parametrize("src, html", [
    (r"$H_\mathrm{ac} = 4H_\mathrm{p}$", "H<sub>ac</sub> = 4H<sub>p</sub>"),
    (r"$\Delta H = 0.1H_\mathrm{p}$", "ΔH = 0.1H<sub>p</sub>"),
    (r"$\chi_3''$ at $\mu_0 H$", "χ<sub>3</sub>″ at μ<sub>0</sub> H"),
    ("a<b & plain_text", "a&lt;b &amp; plain_text"),
])
def test_tex_to_html(src, html):
    assert tex_to_html(src) == html


def test_four_panel_pdf(tmp_path):
    panels = []
    for i, key in enumerate("abcd"):
        base = presets.get(presets.mh_key(10.0, key))
        curves = [figure.Curve(io.compute(replace(base, params=replace(base.params, hac=h))),
                               SERIES[j], 0, rf"$H_\mathrm{{ac}} = {h:g}H_\mathrm{{p}}$")
                  for j, h in enumerate((1.0, 1.5, 4.0))]
        panels.append((key, curves))
    for arr, width in (("vertical", 86.0), ("horizontal", 178.0), ("grid", 178.0)):
        out = figure.render("mh", panels, tmp_path / f"{arr}.pdf",
                            PdfSettings(width_mm=width, arrangement=arr), normalize=True)
        assert out.read_bytes()[:5] == b"%PDF-"
