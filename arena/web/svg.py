"""Server-side SVG line charts. Pure functions, deterministic output, no JS.

Axes and text use ``currentColor`` so the chart follows the page's light or
dark theme; series colours are a fixed colour-blind-safe palette (Okabe-Ito).
"""

from __future__ import annotations

from math import floor, log10
from xml.sax.saxutils import escape

import pandas as pd

PALETTE = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#999999")
MARGIN = {"top": 16, "right": 16, "bottom": 40, "left": 56}
Y_TICKS = 5
X_TICKS = 6
LEGEND_ROW = 18
MAX_POINTS = 400


def _fmt(v: float) -> str:
    """Compact tick label: 3 significant digits, no trailing zeros."""
    if v == 0:
        return "0"
    digits = max(0, 2 - int(floor(log10(abs(v)))))
    return f"{v:.{digits}f}".rstrip("0").rstrip(".") if digits else f"{v:.0f}"


def _ticks(lo: float, hi: float, n: int) -> list[float]:
    if hi <= lo:
        return [lo]
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def _thin(s: pd.Series, max_points: int = MAX_POINTS) -> pd.Series:
    """Keep every k-th point plus the last one so a path never exceeds ~``max_points`` nodes."""
    if s.size <= max_points:
        return s
    step = -(-s.size // max_points)
    return pd.concat([s.iloc[::step], s.iloc[[-1]]])


def line_chart(
    series: dict[str, pd.Series],
    width: int = 800,
    height: int = 320,
    title: str = "",
    y_label: str = "",
) -> str:
    """Multi-series line chart of ``pd.Series`` indexed by UTC timestamps.

    Each series is drawn as one ``<path>`` with a ``data-series`` attribute;
    the legend lists series in insertion order. Empty input renders an
    accessible placeholder instead of raising.
    """
    clean = {k: _thin(s.dropna().astype(float)) for k, s in series.items() if s is not None and s.dropna().size > 0}
    legend_h = LEGEND_ROW * ((len(clean) + 2) // 3) if clean else 0
    plot_w = width - MARGIN["left"] - MARGIN["right"]
    plot_h = height - MARGIN["top"] - MARGIN["bottom"] - legend_h
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" '
        f'role="img" aria-label="{escape(title or "line chart")}" class="chart" font-size="11">',
        f"<title>{escape(title or 'line chart')}</title>",
    ]
    if not clean:
        parts.append(
            f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" fill="currentColor">no data</text></svg>'
        )
        return "".join(parts)

    x_min = min(int(s.index.min().timestamp()) for s in clean.values())
    x_max = max(int(s.index.max().timestamp()) for s in clean.values())
    y_min = min(float(s.min()) for s in clean.values())
    y_max = max(float(s.max()) for s in clean.values())
    if y_max == y_min:
        y_min, y_max = y_min - 0.5, y_max + 0.5
    pad = (y_max - y_min) * 0.05
    y_min, y_max = y_min - pad, y_max + pad
    x_span = max(1, x_max - x_min)

    def sx(t: int) -> float:
        return MARGIN["left"] + (t - x_min) / x_span * plot_w

    def sy(v: float) -> float:
        return MARGIN["top"] + (y_max - v) / (y_max - y_min) * plot_h

    x0, y0 = MARGIN["left"], MARGIN["top"] + plot_h
    parts.append('<g class="grid" stroke="currentColor" stroke-opacity="0.15" stroke-width="1">')
    for v in _ticks(y_min, y_max, Y_TICKS):
        y = sy(v)
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + plot_w}" y2="{y:.1f}"/>')
    parts.append("</g>")
    parts.append('<g class="axis" fill="currentColor">')
    for v in _ticks(y_min, y_max, Y_TICKS):
        parts.append(f'<text x="{x0 - 6}" y="{sy(v) + 4:.1f}" text-anchor="end">{_fmt(v)}</text>')
    for t in _ticks(x_min, x_max, X_TICKS):
        label = pd.Timestamp(int(t), unit="s", tz="UTC").strftime("%m-%d")
        parts.append(f'<text x="{sx(int(t)):.1f}" y="{y0 + 16}" text-anchor="middle">{label}</text>')
    if y_label:
        parts.append(f'<text x="{x0}" y="{MARGIN["top"] - 4}" font-size="10">{escape(y_label)}</text>')
    parts.append("</g>")
    parts.append(
        f'<path d="M{x0} {MARGIN["top"]} V{y0} H{x0 + plot_w}" fill="none" stroke="currentColor" stroke-opacity="0.5"/>'
    )
    for i, (name, s) in enumerate(clean.items()):
        colour = PALETTE[i % len(PALETTE)]
        pts = " ".join(f"{sx(int(ts.timestamp())):.1f},{sy(float(v)):.1f}" for ts, v in s.items())
        parts.append(
            f'<path data-series="{escape(name)}" d="M{pts.replace(" ", " L")}" fill="none" '
            f'stroke="{colour}" stroke-width="1.5" stroke-linejoin="round"><title>{escape(name)}</title></path>'
        )
    ly = y0 + 32
    parts.append('<g class="legend" fill="currentColor">')
    for i, name in enumerate(clean):
        col, row = i % 3, i // 3
        lx = x0 + col * (plot_w / 3)
        y = ly + row * LEGEND_ROW
        parts.append(f'<rect x="{lx:.1f}" y="{y - 9}" width="12" height="3" fill="{PALETTE[i % len(PALETTE)]}"/>')
        parts.append(f'<text x="{lx + 16:.1f}" y="{y - 5}">{escape(name)}</text>')
    parts.append("</g></svg>")
    return "".join(parts)
