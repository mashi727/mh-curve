"""Named parameter sets; ``DEFAULT_PRESET`` equals the application default (``RunSpec()``)."""

from __future__ import annotations

from dataclasses import replace

from .common import Exponents, Params, reduced_to_kb0
from .io import RunSpec


def from_normalized(p: float, hp: float = 1.0, a: float = 0.1, dh: float = 0.0, hc1: float = 0.0,
                    hac: float = 1.0, hdc: float = 0.0) -> Params:
    """Params from the reduced parameter p and fields given in units of Hp (hp in mT, a in cm)."""
    k, b0 = reduced_to_kb0(p, hp, a)
    return Params(k=k, b0=b0, dh=dh * hp, hc1=hc1 * hp, hac=hac * hp, hdc=hdc * hp, a=a)


# M(H) sets: (dH, Hc1) in units of Hp
_PANELS = {"a": (0.0, 0.0), "b": (0.1, 0.0), "c": (0.0, 0.15), "d": (0.1, 0.15)}


def _cond(panel: str) -> str:
    dh, hc1 = _PANELS[panel]
    f = lambda v: "0" if v == 0 else f"{v:g}Hp"  # noqa: E731
    return f"({panel}) ΔH = {f(dh)}, Hc1 = {f(hc1)}"


def _mh(p: float, panel: str) -> tuple[str, RunSpec]:
    # Hdc = 0, Hac = 4 Hp (largest loop); Hp = 1 mT so the mT axes read as H/Hp, M/Hp
    dh, hc1 = _PANELS[panel]
    spec = RunSpec(kind="mh", name=f"p{p:g}_{panel}".replace(".", "_"),
                   params=from_normalized(p, 1.0, dh=dh, hc1=hc1, hac=4.0))
    return f"M(H) p = {p:g}  {_cond(panel)}", spec


_P_VALUES = (1000.0, 10.0, 0.1)
CHI_EXAMPLE = "χₙ(t) 薄膜の例  Hac = 0.1 mT, Hdc = 0.03 mT"

PRESETS: dict[str, RunSpec] = dict(
    [_mh(p, c) for p in _P_VALUES for c in "abcd"]
    + [(CHI_EXAMPLE, RunSpec(
        kind="chi", name="chi_film",
        exps=Exponents(k=3.0, b0=1.5, dh=1.5, hc1=1.0),
        params=Params(k=250.0, b0=1.0, dh=0.5, hc1=0.0, hac=0.1, hdc=0.03, a=0.1)))]
)

DEFAULT_PRESET = _mh(1000.0, "d")[0]
assert DEFAULT_PRESET in PRESETS

# two-level view for the UI: group -> condition -> key of PRESETS
GROUPS: dict[str, dict[str, str]] = {
    **{f"M(H)  p = {p:g}": {_cond(c): _mh(p, c)[0] for c in "abcd"} for p in _P_VALUES},
    "χₙ(t)  薄膜の例": {"Hac = 0.1 mT, Hdc = 0.03 mT": CHI_EXAMPLE},
}
assert {k for g in GROUPS.values() for k in g.values()} == set(PRESETS)


def locate(key: str) -> tuple[str, str]:
    """(group, condition) of a PRESETS key."""
    for grp, conds in GROUPS.items():
        for cond, k in conds.items():
            if k == key:
                return grp, cond
    raise KeyError(key)


def mh_key(p: float, panel: str) -> str:
    return _mh(p, panel)[0]


def get(name: str) -> RunSpec:
    return replace(PRESETS[name])
