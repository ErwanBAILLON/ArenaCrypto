"""FastAPI application factory for the read-only dashboard.

One Postgres connection per request (opened and closed by a dependency),
server-rendered Jinja2 templates, inline SVG charts. No forms, no writes,
no authentication: access control lives at the ingress.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from arena.settings import Settings
from arena.store.db import connect
from arena.web import queries, svg

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def _now() -> datetime:
    return datetime.now(UTC)


def _pct(v: Any) -> str:
    return "n/a" if v is None else f"{float(v) * 100:+.1f}%"


def _num(v: Any, digits: int = 2) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, (list, tuple)):
        return ", ".join(_num(x, digits) for x in v)
    return f"{float(v):.{digits}f}" if isinstance(v, (int, float)) else str(v)


def _ago(ts: datetime | None, now: datetime | None = None) -> str:
    if ts is None:
        return "never"
    delta: timedelta = (now or _now()) - ts
    secs = int(delta.total_seconds())
    if secs < 90:
        return f"{secs}s ago"
    if secs < 5400:
        return f"{secs // 60}m ago"
    if secs < 172800:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


def _iso(ts: datetime | None) -> str:
    return "" if ts is None else ts.strftime("%Y-%m-%d %H:%M")


def _pretty(d: Any) -> str:
    return json.dumps(d, indent=2, sort_keys=True, default=str)


def _kv(d: dict[str, Any] | None) -> str:
    return "" if not d else ", ".join(f"{k}={_num(v) if isinstance(v, float) else v}" for k, v in d.items())


def _templates() -> Jinja2Templates:
    t = Jinja2Templates(directory=str(TEMPLATES_DIR))
    t.env.filters.update({"pct": _pct, "num": _num, "ago": _ago, "iso": _iso, "pretty": _pretty, "kv": _kv})
    return t


def create_app(settings: Settings) -> FastAPI:
    """Build the dashboard app bound to ``settings.database_url``."""
    if not settings.database_url:
        raise ValueError("DATABASE_URL is required for the web dashboard")
    app = FastAPI(title="Arena", docs_url=None, redoc_url=None, openapi_url=None)
    templates = _templates()

    def get_conn() -> Iterator[psycopg.Connection]:
        conn = connect(settings.database_url)
        try:
            yield conn
        finally:
            conn.close()

    def render(request: Request, name: str, **ctx: Any) -> HTMLResponse:
        return templates.TemplateResponse(request, name, {"now": _now(), **ctx})

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, conn: psycopg.Connection = Depends(get_conn)) -> HTMLResponse:
        now = _now()
        board = queries.leaderboard(conn, now)
        series = queries.nav_series(conn, board.chart_ids, now - timedelta(days=queries.CHART_DAYS), now)
        chart = svg.line_chart(series, title=f"NAV, normalised, last {queries.CHART_DAYS} days", y_label="NAV / NAV[0]")
        return render(request, "index.html", board=board, chart=chart, active="home")

    @app.get("/competitors/{competitor_id}", response_class=HTMLResponse)
    def competitor(request: Request, competitor_id: int, conn: psycopg.Connection = Depends(get_conn)) -> HTMLResponse:
        detail = queries.competitor_detail(conn, competitor_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="unknown competitor")
        now = _now()
        start = detail.first_ts or now
        series = queries.nav_series(conn, [competitor_id], start, now)
        chart = svg.line_chart(series, title=f"{detail.spec.name} NAV since first book row", y_label="NAV / NAV[0]")
        return render(request, "competitor.html", d=detail, chart=chart, active="home")

    @app.get("/alerts", response_class=HTMLResponse)
    def alerts(request: Request, conn: psycopg.Connection = Depends(get_conn)) -> HTMLResponse:
        return render(request, "alerts.html", alerts=queries.alerts(conn, 200), active="alerts")

    @app.get("/trials", response_class=HTMLResponse)
    def trials(request: Request, conn: psycopg.Connection = Depends(get_conn)) -> HTMLResponse:
        return render(request, "trials.html", trials=queries.trials(conn, 200), active="trials")

    @app.get("/healthz")
    def healthz(conn: psycopg.Connection = Depends(get_conn)) -> JSONResponse:
        last = queries.last_bar(conn)
        stale = queries.is_stale(last, _now())
        body = {"ok": not stale, "last_bar": last.isoformat() if last else None}
        return JSONResponse(body, status_code=503 if stale else 200)

    @app.get("/api/leaderboard.json")
    def leaderboard_json(conn: psycopg.Connection = Depends(get_conn)) -> JSONResponse:
        return JSONResponse(queries.leaderboard(conn, _now()).to_json())

    return app
