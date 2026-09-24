"""FastAPI application factory for the read-only dashboard.

One Postgres connection per request (opened and closed by a dependency),
server-rendered Jinja2 templates in French, inline SVG charts. No forms, no
writes, no authentication: access control lives at the ingress.
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
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from arena.explain import families
from arena.settings import Settings
from arena.store.db import connect
from arena.web import attribution, fr, live, queries, svg

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
    return fr.ago_fr(ts, now or _now())


def _iso(ts: datetime | None) -> str:
    return "" if ts is None else ts.strftime("%Y-%m-%d %H:%M")


def _pretty(d: Any) -> str:
    return json.dumps(d, indent=2, sort_keys=True, default=str)


def _kv(d: dict[str, Any] | None) -> str:
    return "" if not d else ", ".join(f"{k}={_num(v) if isinstance(v, float) else v}" for k, v in d.items())


def _family_title(family: str) -> str:
    return families.card(family).title


def _templates() -> Jinja2Templates:
    t = Jinja2Templates(directory=str(TEMPLATES_DIR))
    t.env.filters.update(
        {
            "pct": _pct,
            "num": _num,
            "ago": _ago,
            "iso": _iso,
            "pretty": _pretty,
            "kv": _kv,
            "heure": fr.heure,
            "universe_fr": fr.universe_fr,
            "date_heure": fr.date_heure,
            "date_fr": fr.date_fr,
            "euros": fr.euros,
            "pct_fr": fr.pct_fr,
            "pct_plain": fr.pct_plain,
            "num_fr": fr.num_fr,
            "sign_css": fr.sign_css,
            "duree": fr.duration_fr,
            "kind_fr": fr.kind_fr,
            "trial_kind_fr": fr.trial_kind_fr,
            "verdict_fr": fr.verdict_fr,
            "family_title": _family_title,
        }
    )
    t.env.globals.update(
        {
            "status_fr": fr.status_fr,
            "status_css": fr.status_css,
            "REGIME_FR": fr.REGIME_FR_SHORT,
            "sharpe_fr": fr.sharpe_fr,
            "psr_fr": fr.psr_fr,
            "evidence_fr": fr.evidence_fr,
        }
    )
    return t


def create_app(settings: Settings) -> FastAPI:
    """Build the dashboard app bound to ``settings.database_url``."""
    if not settings.database_url:
        raise ValueError("DATABASE_URL is required for the web dashboard")
    app = FastAPI(title="Arena", docs_url=None, redoc_url=None, openapi_url=None)
    templates = _templates()
    app.mount("/static", StaticFiles(directory=str(TEMPLATES_DIR.parent / "static")), name="static")

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
        return render(request, "index.html", active="home")

    @app.get("/systeme", response_class=HTMLResponse)
    def systeme(request: Request, conn: psycopg.Connection = Depends(get_conn)) -> HTMLResponse:
        now = _now()
        live = queries.live_page(conn, now)
        board = queries.leaderboard(conn, now)
        series = queries.nav_series(conn, board.chart_ids, now - timedelta(days=queries.CHART_DAYS), now)
        series = {k: v * queries.NAV0 for k, v in series.items()}
        chart = svg.line_chart(
            series, title=f"10 000 € virtuels : évolution sur {queries.CHART_DAYS} jours", y_label="valeur en €"
        )
        return render(
            request,
            "systeme.html",
            live=live,
            board=board,
            chart=chart,
            maturity=queries.maturity(conn, board, now),
            active="systeme",
        )

    @app.get("/competitors/{competitor_id}", response_class=HTMLResponse)
    def competitor(request: Request, competitor_id: int, conn: psycopg.Connection = Depends(get_conn)) -> HTMLResponse:
        now = _now()
        page = queries.competitor_page(conn, competitor_id, now)
        if page is None:
            raise HTTPException(status_code=404, detail="unknown competitor")
        start = page.first_ts or now
        series = queries.nav_series(conn, [competitor_id], start, now)
        series = {k: v * queries.NAV0 for k, v in series.items()}
        chart = svg.line_chart(series, title=f"{page.card.title} : 10 000 € virtuels depuis le début", y_label="€")
        days = max(1, int((now - start).days) + 1) if page.first_ts else 90
        return render(request, "competitor.html", p=page, chart=chart, chart_days=min(days, 3650), active="home")

    @app.get("/competitors/{competitor_id}/attribution", response_class=HTMLResponse)
    def attribution_page(
        request: Request, competitor_id: int, conn: psycopg.Connection = Depends(get_conn)
    ) -> HTMLResponse:
        spec = queries.competitor_page(conn, competitor_id, _now())
        if spec is None:
            raise HTTPException(status_code=404, detail="unknown competitor")
        eps = attribution.episodes(conn, competitor_id)
        return render(
            request,
            "attribution.html",
            p=spec,
            summary=attribution.summary(eps),
            by_symbol=attribution.decompose(eps, "symbol")[:20],
            by_side=attribution.decompose(eps, "side"),
            by_conviction=attribution.decompose(eps, "conviction"),
            by_duration=attribution.decompose(eps, "bars_held"),
            reliability=attribution.reliability(eps),
            worst=attribution.worst(eps, 12),
            journal=list(reversed(eps))[:60],
            active="home",
        )

    @app.get("/modeles", response_class=HTMLResponse)
    def modeles(request: Request, conn: psycopg.Connection = Depends(get_conn)) -> HTMLResponse:
        return render(request, "families.html", families=queries.families_overview(conn), active="modeles")

    @app.get("/alerts", response_class=HTMLResponse)
    def alerts(request: Request, conn: psycopg.Connection = Depends(get_conn)) -> HTMLResponse:
        return render(request, "alerts.html", alerts=queries.recent_alerts_fr(conn, 200), active="alerts")

    @app.get("/trials", response_class=HTMLResponse)
    def trials(request: Request, tous: int = 0, conn: psycopg.Connection = Depends(get_conn)) -> HTMLResponse:
        return render(
            request,
            "trials.html",
            trials=queries.trials(conn, 200, include_search=bool(tous)),
            hidden=0 if tous else queries.search_trial_count(conn),
            include_search=bool(tous),
            active="trials",
        )

    @app.get("/healthz")
    def healthz(conn: psycopg.Connection = Depends(get_conn)) -> JSONResponse:
        last = queries.last_bar(conn)
        stale = queries.is_stale(last, _now())
        body = {"ok": not stale, "last_bar": last.isoformat() if last else None}
        return JSONResponse(body, status_code=503 if stale else 200)

    @app.get("/api/live")
    def api_live(conn: psycopg.Connection = Depends(get_conn)) -> JSONResponse:
        """The two facts a page may animate every second, and whether anything else changed."""
        return JSONResponse(live.live_state(conn, _now()))

    @app.get("/api/book")
    def api_book(conn: psycopg.Connection = Depends(get_conn)) -> JSONResponse:
        """Every agent's stored book with the reference prices the browser marks to market."""
        return JSONResponse(live.book_state(conn, _now()))

    @app.get("/api/series/{competitor_id}")
    def api_series(competitor_id: int, days: int = 90, conn: psycopg.Connection = Depends(get_conn)) -> JSONResponse:
        """One competitor's NAV with every trade that changed its book, for the chart."""
        body = live.competitor_series(conn, competitor_id, max(1, min(days, 3650)), _now())
        if not body:
            raise HTTPException(status_code=404, detail="unknown competitor")
        return JSONResponse(body)

    @app.get("/api/board")
    def api_board(days: int = 90, conn: psycopg.Connection = Depends(get_conn)) -> JSONResponse:
        """Champions and benchmarks, normalised to 100, on one time axis."""
        board = queries.leaderboard(conn, _now())
        ids = [r["id"] for r in board.rows if r["role"] != "null"][:8]
        return JSONResponse(live.board_series(conn, ids, max(1, min(days, 3650)), _now()))

    @app.get("/api/leaderboard.json")
    def leaderboard_json(conn: psycopg.Connection = Depends(get_conn)) -> JSONResponse:
        return JSONResponse(queries.leaderboard(conn, _now()).to_json())

    return app
