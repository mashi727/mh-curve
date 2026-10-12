"""Run specification (.toml) and result data (.csv) -- the formats shared by the GUI and a
future batch runner.

Parameter file::

    kind = "mh"            # "mh" (M-H loop) | "chi" (chi_n vs t)
    engine = "analytic"    # "analytic" (CHI999 closed form) | "numeric"
    name = "Y123_a"        # output basename

    [material]             # values at t = 0, X(t) = X (1-t)^E
    k = 1.0e6              # N/m^3
    b0 = 10.0              # mT
    dh = 1.0               # mT
    hc1 = 5.0              # mT
    [exponents]
    k = 2.0
    ...
    [excitation]
    hac = 60.0             # mT
    hdc = 0.0              # mT
    [sample]
    a = 0.1                # cm
    [temperature]
    t = 0.0                # M-H loop is evaluated at this t
    tmin = 0.0             # chi_n(t) scan
    tmax = 0.99
    tstep = 0.01
    [numerics]
    n_half = 180

Result file: ``# key = value`` header lines carrying the full specification and derived
quantities, then a plain CSV table (readable by numpy, pandas, pgfplots).
"""

from __future__ import annotations

import csv
import datetime as _dt
import tomllib
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import numpy as np

from . import analytic, numeric
from .common import REGIME_LABEL, Exponents, Params, harmonics
from .sweep import chi_vs_t, t_grid

NMAX = 7
_SECTIONS = {
    "material": ("k", "b0", "dh", "hc1"),
    "excitation": ("hac", "hdc"),
    "sample": ("a",),
    "numerics": ("n_half",),
}
_UNITS = {"k": "N/m^3", "b0": "mT", "dh": "mT", "hc1": "mT", "hac": "mT", "hdc": "mT", "a": "cm"}


@dataclass
class RunSpec:
    kind: str = "mh"
    engine: str = "analytic"
    name: str = "run"
    params: Params = field(default_factory=Params)
    exps: Exponents = field(default_factory=Exponents)
    t: float = 0.0
    tmin: float = 0.0
    tmax: float = 0.99
    tstep: float = 0.01

    def params_at_t(self) -> Params:
        return self.params.at_t(self.t, self.exps)

    # ---- flat key/value view (CSV header) --------------------------------
    def to_flat(self) -> dict[str, str]:
        d = {"kind": self.kind, "engine": self.engine, "name": self.name}
        for k, v in asdict(self.params).items():
            d[f"param.{k}"] = repr(v)
        for k, v in asdict(self.exps).items():
            d[f"exp.{k}"] = repr(v)
        for k in ("t", "tmin", "tmax", "tstep"):
            d[f"temp.{k}"] = repr(getattr(self, k))
        return d

    @classmethod
    def from_flat(cls, d: dict[str, str]) -> "RunSpec":
        def pick(prefix, typ):
            vals = {}
            for f in fields(typ):
                key = f"{prefix}.{f.name}"
                if key in d:
                    vals[f.name] = int(float(d[key])) if f.type in (int, "int") else float(d[key])
            return typ(**vals)

        spec = cls(
            kind=d.get("kind", "mh"),
            engine=d.get("engine", "analytic"),
            name=d.get("name", "run"),
            params=pick("param", Params),
            exps=pick("exp", Exponents),
        )
        for k in ("t", "tmin", "tmax", "tstep"):
            if f"temp.{k}" in d:
                setattr(spec, k, float(d[f"temp.{k}"]))
        return spec


# ---------------------------------------------------------------------------
# .toml
# ---------------------------------------------------------------------------
def load_spec(path: str | Path) -> RunSpec:
    with open(path, "rb") as f:
        doc = tomllib.load(f)
    spec = RunSpec(
        kind=doc.get("kind", "mh"),
        engine=doc.get("engine", "analytic"),
        name=doc.get("name", Path(path).stem),
    )
    pvals = {}
    for sec, keys in _SECTIONS.items():
        for k in keys:
            if k in doc.get(sec, {}):
                pvals[k] = doc[sec][k]
    if "n_half" in pvals:
        pvals["n_half"] = int(pvals["n_half"])
    spec.params = replace(spec.params, **pvals)
    spec.exps = replace(spec.exps, **{k: float(v) for k, v in doc.get("exponents", {}).items()})
    for k, v in doc.get("temperature", {}).items():
        if k in ("t", "tmin", "tmax", "tstep"):
            setattr(spec, k, float(v))
    return spec


def save_spec(spec: RunSpec, path: str | Path) -> None:
    p, e = spec.params, spec.exps
    lines = [
        "# mh-curve parameter file (modified Kim-Anderson model)",
        f'kind = "{spec.kind}"          # "mh" | "chi"',
        f'engine = "{spec.engine}"  # "analytic" | "numeric"',
        f'name = "{spec.name}"',
        "",
        "[material]  # values at t = 0;  X(t) = X * (1 - t)^E",
    ]
    for k in _SECTIONS["material"]:
        lines.append(f"{k} = {getattr(p, k)!r}  # {_UNITS[k]}")
    lines += ["", "[exponents]"]
    for k in ("k", "b0", "dh", "hc1"):
        lines.append(f"{k} = {getattr(e, k)!r}")
    lines += ["", "[excitation]"]
    for k in _SECTIONS["excitation"]:
        lines.append(f"{k} = {getattr(p, k)!r}  # {_UNITS[k]}")
    lines += ["", "[sample]", f"a = {p.a!r}  # cm", "", "[temperature]"]
    for k in ("t", "tmin", "tmax", "tstep"):
        lines.append(f"{k} = {getattr(spec, k)!r}")
    lines += ["", "[numerics]", f"n_half = {p.n_half}  # points per half cycle", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# computation -> Dataset
# ---------------------------------------------------------------------------
@dataclass
class Dataset:
    kind: str
    meta: dict[str, str]
    columns: dict[str, np.ndarray]
    path: Path | None = None

    @property
    def label(self) -> str:
        return self.path.stem if self.path else self.meta.get("name", "")

    @property
    def spec(self) -> RunSpec:
        return RunSpec.from_flat(self.meta)


def compute(spec: RunSpec) -> Dataset:
    meta = spec.to_flat()
    if spec.kind == "mh":
        si = spec.params_at_t().si
        eng = analytic if spec.engine == "analytic" else numeric
        r = eng.compute_loop(si)
        T = si.T
        th = np.concatenate([np.arange(len(r.H_init)), np.arange(len(r.H_loop))]) * 180.0 / T
        cols = {
            "branch": np.array(["init"] * len(r.H_init) + ["loop"] * len(r.H_loop)),
            "theta_deg": th,
            "H_mT": np.concatenate([r.H_init, r.H_loop]) * 1e3,
            "M_mT": np.concatenate([r.M_init, r.M_loop]) * 1e3,
        }
        re, im = harmonics(r.M_loop, si.Hac, NMAX)
        meta["derived.regime"] = r.regime
        meta["derived.Hp_mT"] = f"{si.Hp * 1e3:.6g}"
        meta["derived.Hstar_mT"] = f"{si.Hstar * 1e3:.6g}"
        for n in range(NMAX):
            meta[f"derived.chi{n + 1}"] = f"{re[n]:.6g} -i {im[n]:.6g}"
    else:
        ts = t_grid(spec.tmin, spec.tmax, spec.tstep)
        re, im = chi_vs_t(spec.params, spec.exps, ts, spec.engine, NMAX)
        cols = {"t": ts}
        for n in range(NMAX):
            cols[f"chi{n + 1}_re"] = re[:, n]
            cols[f"chi{n + 1}_im"] = im[:, n]
    return Dataset(spec.kind, meta, cols)


def describe(spec: RunSpec) -> str:
    si = spec.params_at_t().si
    return (
        f"t={spec.t:.3g}:  Hp = {si.Hp * 1e3:.4g} mT,  H* = {si.Hstar * 1e3:.4g} mT,  "
        f"HA = {si.HA * 1e3:.4g} mT  →  {REGIME_LABEL[si.regime()]}"
    )


# ---------------------------------------------------------------------------
# .csv
# ---------------------------------------------------------------------------
def save_dataset(ds: Dataset, path: str | Path) -> Path:
    path = Path(path)
    meta = {"format": "mh-curve/1", "created": _dt.datetime.now().isoformat(timespec="seconds")}
    meta.update(ds.meta)
    names = list(ds.columns)
    n = len(next(iter(ds.columns.values())))
    with open(path, "w", newline="", encoding="utf-8") as f:
        for k, v in meta.items():
            f.write(f"# {k} = {v}\n")
        w = csv.writer(f)
        w.writerow(names)
        for i in range(n):
            row = []
            for c in names:
                v = ds.columns[c][i]
                row.append(v if isinstance(v, str) else f"{float(v):.10g}")
            w.writerow(row)
    ds.path = path
    return path


def load_dataset(path: str | Path) -> Dataset:
    path = Path(path)
    meta: dict[str, str] = {}
    body: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                k, sep, v = line[1:].partition("=")
                if sep:
                    meta[k.strip()] = v.strip()
            elif line.strip():
                body.append(line)
    if meta.get("format", "").split("/")[0] != "mh-curve":
        raise ValueError(f"{path.name}: mh-curve のデータファイルではありません")
    rows = list(csv.reader(body))
    names, data = rows[0], rows[1:]
    cols: dict[str, np.ndarray] = {}
    for j, c in enumerate(names):
        vals = [r[j] for r in data]
        cols[c] = np.array(vals) if c == "branch" else np.array(vals, dtype=float)
    return Dataset(meta.get("kind", "mh"), meta, cols, path)
