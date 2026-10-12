"""Colours (validated reference palette) and on-screen plot metrics.

Series hues are assigned in fixed order. Past eight series the hue order repeats with a
different dash pattern, so identity is carried by colour x dash rather than by a generated
ninth hue. Stored colours are the light steps; the dark theme maps them to the dark steps of
the same hue for display only (PDF output always uses the stored light colours).
"""

from __future__ import annotations

from dataclasses import dataclass

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9",
               "#e66767"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
GRID_STRONG = "#c3c2b7"  # live preview: a recessive wide band behind the saved curves


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str  # plot background
    ink: str
    ink2: str  # axis text, legend, labels
    muted: str  # axis lines
    grid: str  # zero lines
    preview: str  # live-preview band
    series: tuple[str, ...]

    @property
    def dark(self) -> bool:
        return self.name == "dark"

    def display(self, stored: str) -> str:
        """Colour to draw a stored (light-step) series colour with in this theme."""
        if stored in SERIES:
            return self.series[SERIES.index(stored)]
        return stored


LIGHT = Theme("light", SURFACE, INK, INK_2, MUTED, GRID, GRID_STRONG, tuple(SERIES))
DARK = Theme("dark", "#1a1a19", "#ffffff", "#c3c2b7", "#898781", "#2c2c2a", "#4a4945",
             tuple(SERIES_DARK))
THEME_MODES = {"light": "ライト", "dark": "ダーク", "system": "システムに合わせる"}

# on-screen plot text sizes [pt] (macOS default was 13 / ~14 / 9 / 10)
PT_TICK, PT_AXIS, PT_LEGEND, PT_TITLE = 16, 18, 14, 16

# data span / axis span on autoscale; the padding is added on each side
DATA_FILL = 0.9
PAD = (1 / DATA_FILL - 1) / 2  # = 0.0556 (pyqtgraph default padding is 0.02)

_DASHES = ["-", (0, (5, 2)), (0, (1, 1.5)), (0, (5, 1.5, 1, 1.5))]


def dash_for(i: int):
    """matplotlib linestyle for the i-th series (changes every 8 series)."""
    return _DASHES[(i // len(SERIES)) % len(_DASHES)]


def qt_dash_index(i: int) -> int:
    return (i // len(SERIES)) % len(_DASHES)
