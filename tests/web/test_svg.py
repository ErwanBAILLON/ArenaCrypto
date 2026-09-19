import pytest

pytest.importorskip("fastapi", reason="web extra not installed")
"""svg.line_chart: valid XML, one path per series, deterministic."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pandas as pd

from arena.web import svg

NS = "{http://www.w3.org/2000/svg}"


def _series(n: int = 48, scale: float = 1.0) -> pd.Series:
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    return pd.Series([scale * (1 + 0.001 * i) for i in range(n)], index=idx)


def test_line_chart_one_path_per_series() -> None:
    out = svg.line_chart({"alpha": _series(), "beta & co": _series(scale=0.9)}, title="t")
    root = ET.fromstring(out)
    assert root.tag == f"{NS}svg"
    paths = [p for p in root.iter(f"{NS}path") if p.get("data-series")]
    assert [p.get("data-series") for p in paths] == ["alpha", "beta & co"]
    assert root.find(f"{NS}title").text == "t"
    assert len(root.findall(f".//{NS}text")) >= svg.Y_TICKS + svg.X_TICKS


def test_line_chart_deterministic_and_empty_safe() -> None:
    series = {"a": _series()}
    assert svg.line_chart(series) == svg.line_chart(series)
    empty = svg.line_chart({})
    assert "no data" in empty
    ET.fromstring(empty)
    only_nan = svg.line_chart({"x": pd.Series([float("nan")], index=pd.DatetimeIndex(["2026-01-01"], tz="UTC"))})
    assert "no data" in only_nan
