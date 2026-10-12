"""Display list ("layers") and its file format (*.view.toml). No Qt here.

    format = "mh-curve-view/1"
    normalize = false          # H/Hp, M/Hp
    flip = false               # -M upward
    harmonic = 1
    show_re = true
    show_im = true

    [pdf]                      # publication-figure settings (figure.FigureStyle)
    width_mm = 86.0
    ...

    [[layer]]                  # one curve, in drawing order
    file = 'loopA.csv'         # relative to the view file
    label = '$H_\\mathrm{ac} = H_\\mathrm{p}$'
    color = '#2a78d6'
    panel = 'a'
    visible = true
    dash = 0
"""

from __future__ import annotations

import json
import os
import re
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .palette import SERIES

PANELS = "abcdefgh"
VIEW_SUFFIX = ".view.toml"


@dataclass
class Layer:
    path: Path
    label: str
    color: str
    panel: str = "a"
    visible: bool = True
    dash: int = 0  # 0 solid, 1 dashed, 2 dotted, 3 dash-dot


@dataclass
class PdfSettings:
    width_mm: float = 86.0
    aspect: float = 0.75  # panel height / panel width
    font: str = "STIX Two (Times 系)"
    font_pt: float = 8.0
    legend: bool = True
    arrangement: str = "vertical"  # vertical | horizontal | grid
    panel_labels: bool = True


@dataclass
class ViewState:
    layers: list[Layer] = field(default_factory=list)
    normalize: bool = False
    flip: bool = False
    harmonic: int = 1
    show_re: bool = True
    show_im: bool = True
    pdf: PdfSettings = field(default_factory=PdfSettings)


def next_style(layers: list[Layer]) -> tuple[str, int]:
    """First palette slot not in use (fixed order); past eight, repeat hues with a new dash."""
    used = {(l.color, l.dash) for l in layers}
    for dash in range(4):
        for c in SERIES:
            if (c, dash) not in used:
                return c, dash
    return SERIES[len(layers) % len(SERIES)], 0


def is_view_file(p: Path) -> bool:
    return p.name.endswith(VIEW_SUFFIX)


# ---------------------------------------------------------------------------
# file I/O
# ---------------------------------------------------------------------------
def _toml(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    s = str(v)
    if "'" not in s and "\n" not in s:
        return f"'{s}'"  # literal string: backslashes of TeX stay as typed
    return json.dumps(s, ensure_ascii=False)


def save_view(state: ViewState, path: str | Path) -> Path:
    path = Path(path)
    base = path.parent
    out = ['# mh-curve view (display list)', 'format = "mh-curve-view/1"']
    for k in ("normalize", "flip", "harmonic", "show_re", "show_im"):
        out.append(f"{k} = {_toml(getattr(state, k))}")
    out += ["", "[pdf]"]
    out += [f"{k} = {_toml(v)}" for k, v in asdict(state.pdf).items()]
    for l in state.layers:
        rel = Path(os.path.relpath(l.path, base)).as_posix()
        out += ["", "[[layer]]", f"file = {_toml(rel)}", f"label = {_toml(l.label)}",
                f"color = {_toml(l.color)}", f"panel = {_toml(l.panel)}",
                f"visible = {_toml(l.visible)}", f"dash = {l.dash}"]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


def load_view(path: str | Path) -> ViewState:
    path = Path(path)
    with open(path, "rb") as f:
        doc = tomllib.load(f)
    if not str(doc.get("format", "")).startswith("mh-curve-view"):
        raise ValueError(f"{path.name}: 表示リストのファイルではありません")
    pdf_keys = {f.name for f in fields(PdfSettings)}
    st = ViewState(pdf=PdfSettings(**{k: v for k, v in doc.get("pdf", {}).items() if k in pdf_keys}))
    for k in ("normalize", "flip", "harmonic", "show_re", "show_im"):
        if k in doc:
            setattr(st, k, doc[k])
    for d in doc.get("layer", []):
        p = Path(d["file"])
        st.layers.append(Layer(
            path=(path.parent / p).resolve() if not p.is_absolute() else p,
            label=d.get("label", p.stem), color=d.get("color", SERIES[0]),
            panel=str(d.get("panel", "a"))[:1] or "a", visible=bool(d.get("visible", True)),
            dash=int(d.get("dash", 0))))
    return st


# ---------------------------------------------------------------------------
# legend text: TeX (matplotlib mathtext) -> rich text for the on-screen legend
# ---------------------------------------------------------------------------
_SYMBOLS = {
    "Delta": "Δ", "delta": "δ", "mu": "μ", "chi": "χ", "alpha": "α", "beta": "β",
    "gamma": "γ", "epsilon": "ε", "theta": "θ", "lambda": "λ", "pi": "π", "sigma": "σ",
    "omega": "ω", "Omega": "Ω", "phi": "φ", "Phi": "Φ", "xi": "ξ", "kappa": "κ",
    "prime": "′", "times": "×", "cdot": "·", "pm": "±", "leq": "≤", "geq": "≥", "le": "≤",
    "ge": "≥", "approx": "≈", "infty": "∞", "to": "→", "propto": "∝", "sim": "∼",
}


def _math_to_html(m: str) -> str:
    m = re.sub(r"\\math(?:rm|it|bf|sf|tt)\{([^{}]*)\}", r"{\1}", m)  # keep the group for _{..}
    m = re.sub(r"\\(?:,|;|:|!| )", " ", m)
    # like TeX, a control word swallows the following space
    m = re.sub(r"\\([A-Za-z]+) ?", lambda g: _SYMBOLS.get(g.group(1), g.group(1)), m)
    m = m.replace("''", "″").replace("'", "′")
    for _ in range(3):  # nested scripts
        m = re.sub(r"_\{([^{}]*)\}", r"<sub>\1</sub>", m)
        m = re.sub(r"\^\{([^{}]*)\}", r"<sup>\1</sup>", m)
    m = re.sub(r"_([^\s<{}])", r"<sub>\1</sub>", m)
    m = re.sub(r"\^([^\s<{}])", r"<sup>\1</sup>", m)
    return m.replace("{", "").replace("}", "")


def tex_to_html(s: str) -> str:
    parts = s.split("$")
    out = []
    for i, p in enumerate(parts):
        p_esc = p.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        out.append(_math_to_html(p_esc) if i % 2 else p_esc)
    return "".join(out)
