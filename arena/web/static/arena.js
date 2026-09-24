/* Arena dashboard: interactive charts and the live layer.
 *
 * Two clocks, deliberately distinct. Every second the page recomputes what is
 * true every second: how old the last tick is, how long until the next. Every
 * POLL_MS it asks /api/live whether a tick has written (the `version` field is
 * the last booked bar); only then do positions, events and charts refetch. The
 * arena decides hourly, and the page never pretends otherwise.
 *
 * Charts: uPlot (vendored). Categorical hues are the validated palette from the
 * dataviz reference, assigned in fixed slot order per series, never cycled.
 * Trade markers are drawn on the NAV line: ▲ entrée / ▼ sortie / ◆ retournement /
 * ● taille, coloured by side, with a 2px surface ring so they stay legible on
 * the line. Every marker also lives in the events table beneath the chart, so
 * nothing is hover-only.
 */
(function () {
  "use strict";

  const POLL_MS = 10000;
  const LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"];
  const DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];
  const isDark = () => document.documentElement.dataset.theme === "dark" ||
    (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  const palette = () => (isDark() ? DARK : LIGHT);
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const fmtEur = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 0 });
  const fmtPct = (v) => (v >= 0 ? "+" : "") + (v * 100).toFixed(1).replace(".", ",") + " %";
  const fmtTs = (t) => new Date(t * 1000).toLocaleString("fr-FR", { timeZone: "UTC", day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" }) + " UTC";
  const SIDE_COLOR = { long: "#1baf7a", short: "#e34948" };
  const KIND_GLYPH = { entrée: "▲", sortie: "▼", retournement: "◆", renfort: "●", allège: "○" };

  function ago(iso) {
    if (!iso) return "jamais";
    const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
    if (s < 60) return `il y a ${Math.floor(s)} s`;
    if (s < 3600) return `il y a ${Math.floor(s / 60)} min ${Math.floor(s % 60)} s`;
    return `il y a ${Math.floor(s / 3600)} h ${Math.floor((s % 3600) / 60)} min`;
  }
  function countdown(iso) {
    if (!iso) return "—";
    const s = Math.max(0, (new Date(iso).getTime() - Date.now()) / 1000);
    return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  }

  /* ------------------------------------------------------------------ live layer */
  let liveState = null;
  let liveVersion = null;
  const subscribers = [];
  function onTick(fn) { subscribers.push(fn); }

  async function pollLive() {
    try {
      const r = await fetch("/api/live", { cache: "no-store" });
      if (!r.ok) throw new Error(r.status);
      const state = await r.json();
      liveState = state;
      renderLive(state);
      if (state.version !== liveVersion) {
        const first = liveVersion === null;
        liveVersion = state.version;
        if (!first) subscribers.forEach((fn) => fn(state));
      }
      document.body.classList.remove("offline");
    } catch (e) {
      document.body.classList.add("offline");
    }
  }
  function renderClocks() {
    if (!liveState) return;
    const a = document.getElementById("live-age");
    const c = document.getElementById("live-countdown");
    if (a) a.textContent = ago(liveState.last_tick);
    if (c) c.textContent = countdown(liveState.next_tick);
  }
  function renderLive(state) {
    const dot = document.getElementById("live-dot");
    if (dot) dot.className = "live-dot " + (state.tick_ok === false ? "bad" : state.tick_ok ? "ok" : "warn");
    const box = document.getElementById("live-events");
    if (box) {
      const seen = Number(box.dataset.maxT || 0);
      box.textContent = "";
      for (const e of state.recent_events) {
        const li = document.createElement("li");
        if (seen && e.t > seen) li.className = "new";
        const g = document.createElement("span");
        g.className = "glyph " + e.side; g.textContent = KIND_GLYPH[e.kind] || "●";
        const t = document.createElement("span"); t.className = "time"; t.textContent = fmtTs(e.t);
        const who = document.createElement("a"); who.href = `/competitors/${e.competitor_id}`; who.textContent = e.name;
        const what = document.createElement("span");
        what.textContent = ` ${e.kind} ${e.symbol} ` + (e.kind === "sortie" ? (e.reason.live_exit ? `· ${e.reason.live_exit === "stop" ? "stop" : "objectif atteint"} en direct à ${e.reason.price}` : "") : `(${e.side === "long" ? "achat" : "vente"} ${(Math.abs(e.after) * 100).toFixed(1)} % du capital)`);
        if (e.reason.live_exit) li.classList.add("live-exit");
        li.append(g, t, who, what);
        box.appendChild(li);
      }
      if (!state.recent_events.length) { const li = document.createElement("li"); li.className = "muted"; li.textContent = "aucun mouvement sur 48 h"; box.appendChild(li); }
      box.dataset.maxT = String(Math.max(seen, ...state.recent_events.map((e) => e.t)));
    }
    const pos = document.getElementById("live-positions");
    if (pos) {
      pos.textContent = "";
      const byComp = new Map();
      for (const p of state.positions) { if (!byComp.has(p.name)) byComp.set(p.name, { id: p.competitor_id, family: p.family, status: p.status, rows: [] }); byComp.get(p.name).rows.push(p); }
      for (const [name, c] of byComp) {
        const card = document.createElement("div"); card.className = "poscard";
        const h = document.createElement("div"); h.className = "poshead";
        const a = document.createElement("a"); a.href = `/competitors/${c.id}`; a.textContent = name;
        const pill = document.createElement("span"); pill.className = "pill " + c.status; pill.textContent = c.status === "champion" ? "champion" : "prétendant";
        h.append(a, pill); card.appendChild(h);
        const bars = document.createElement("div"); bars.className = "posbars";
        const maxW = Math.max(...c.rows.map((r) => Math.abs(r.weight)), 0.01);
        for (const r of c.rows) {
          const row = document.createElement("div"); row.className = "posrow";
          const lab = document.createElement("span"); lab.className = "poslab"; lab.textContent = r.symbol;
          const bar = document.createElement("span"); bar.className = "posbar " + (r.weight > 0 ? "long" : "short");
          bar.style.width = `${Math.round((Math.abs(r.weight) / maxW) * 100)}%`;
          const val = document.createElement("span"); val.className = "posval"; val.textContent = `${r.weight > 0 ? "+" : "−"}${(Math.abs(r.weight) * 100).toFixed(1)} %` + (r.kind === "carry" ? " carry" : "");
          row.append(lab, bar, val); bars.appendChild(row);
        }
        card.appendChild(bars); pos.appendChild(card);
      }
      if (!byComp.size) { const d = document.createElement("div"); d.className = "muted"; d.textContent = "Tout le monde est en cash."; pos.appendChild(d); }
    }
    renderClocks();
  }

  /* ------------------------------------------------------------------ charts */
  const charts = [];
  function baseOpts(el, ySuffix, seriesNames, colors) {
    const dark = isDark();
    const fg = css("--fg") || (dark ? "#e6e7ea" : "#1d1f21");
    const grid = css("--line") || (dark ? "#2b2f36" : "#e2e4e8");
    return {
      width: el.clientWidth || 800,
      height: Math.max(220, Math.min(420, Math.round((el.clientWidth || 800) * 0.42))),
      cursor: { drag: { x: true, y: false }, points: { size: 9, width: 2, stroke: (u, i) => colors[i - 1], fill: css("--card") || "#fff" } },
      legend: { show: false },
      scales: { x: { time: true } },
      axes: [
        { stroke: fg, grid: { stroke: grid, width: 1 }, ticks: { stroke: grid, width: 1 }, font: "12px system-ui", values: (u, vals) => vals.map((v) => new Date(v * 1000).toLocaleDateString("fr-FR", { timeZone: "UTC", day: "2-digit", month: "short" })) },
        { stroke: fg, grid: { stroke: grid, width: 1 }, ticks: { stroke: grid, width: 1 }, font: "12px system-ui", size: 76, values: (u, vals) => vals.map((v) => (ySuffix === "€" ? fmtEur.format(v) + " €" : v.toFixed(1))) },
      ],
      series: [{ label: "date" }, ...seriesNames.map((n, i) => ({ label: n, stroke: colors[i], width: 2, points: { show: false } }))],
    };
  }

  function tooltipPlugin(u, container, seriesNames, colors, fmt, eventsAt) {
    const tip = document.createElement("div"); tip.className = "u-tip"; tip.style.display = "none"; container.appendChild(tip);
    u.over.addEventListener("mouseleave", () => (tip.style.display = "none"));
    return (idx) => {
      if (idx == null) { tip.style.display = "none"; return; }
      tip.textContent = "";
      const head = document.createElement("div"); head.className = "u-tip-h"; head.textContent = fmtTs(u.data[0][idx]); tip.appendChild(head);
      for (let s = 1; s < u.data.length; s++) {
        if (!u.series[s].show) continue;
        const v = u.data[s][idx]; if (v == null) continue;
        const row = document.createElement("div"); row.className = "u-tip-r";
        const key = document.createElement("span"); key.className = "u-key"; key.style.background = colors[s - 1];
        const val = document.createElement("strong"); val.textContent = fmt(v);
        const lab = document.createElement("span"); lab.className = "muted"; lab.textContent = seriesNames[s - 1];
        row.append(key, val, lab); tip.appendChild(row);
      }
      const evs = eventsAt ? eventsAt(u.data[0][idx]) : [];
      for (const e of evs.slice(0, 6)) {
        const row = document.createElement("div"); row.className = "u-tip-e " + e.side;
        row.textContent = `${KIND_GLYPH[e.kind] || "●"} ${e.kind} ${e.symbol} → ${(e.after * 100).toFixed(1).replace(".", ",")} %`;
        tip.appendChild(row);
      }
      const left = u.valToPos(u.data[0][idx], "x");
      tip.style.display = "block";
      tip.style.left = `${Math.min(left + 14, u.over.clientWidth - tip.offsetWidth - 4)}px`;
      tip.style.top = "8px";
    };
  }

  function markersPlugin(getEvents, timeIndex) {
    /* draws ▲/▼/◆/● at the NAV of the bar each event happened on */
    return {
      hooks: {
        draw: (u) => {
          const events = getEvents(); if (!events || !events.length) return;
          const ctx = u.ctx; const ring = css("--card") || "#fff";
          ctx.save();
          for (const e of events) {
            const i = timeIndex(e.t); if (i < 0) continue;
            const x = u.valToPos(u.data[0][i], "x", true), y = u.valToPos(u.data[1][i], "y", true);
            if (!isFinite(x) || !isFinite(y)) continue;
            const r = 6 * devicePixelRatio, color = SIDE_COLOR[e.side] || "#888";
            ctx.beginPath(); ctx.fillStyle = ring; ctx.strokeStyle = ring; ctx.lineWidth = 2 * devicePixelRatio;
            drawGlyph(ctx, e.kind, x, y, r + 1.5 * devicePixelRatio); ctx.fill();
            ctx.beginPath(); ctx.fillStyle = color; drawGlyph(ctx, e.kind, x, y, r); ctx.fill();
          }
          ctx.restore();
        },
      },
    };
  }
  function drawGlyph(ctx, kind, x, y, r) {
    if (kind === "entrée") { ctx.moveTo(x, y - r); ctx.lineTo(x + r, y + r); ctx.lineTo(x - r, y + r); ctx.closePath(); }
    else if (kind === "sortie") { ctx.moveTo(x, y + r); ctx.lineTo(x + r, y - r); ctx.lineTo(x - r, y - r); ctx.closePath(); }
    else if (kind === "retournement") { ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y); ctx.closePath(); }
    else { ctx.arc(x, y, r * 0.8, 0, Math.PI * 2); }
  }

  function legend(container, names, colors, u) {
    const box = document.createElement("div"); box.className = "u-legend";
    names.forEach((n, i) => {
      const b = document.createElement("button"); b.type = "button"; b.className = "u-lg"; b.setAttribute("aria-pressed", "true");
      const key = document.createElement("span"); key.className = "u-key"; key.style.background = colors[i];
      const lab = document.createElement("span"); lab.textContent = n;
      b.append(key, lab);
      b.addEventListener("click", () => { const s = u.series[i + 1]; u.setSeries(i + 1, { show: !s.show }); b.setAttribute("aria-pressed", String(s.show)); b.classList.toggle("off", !s.show); });
      box.appendChild(b);
    });
    container.appendChild(box);
  }

  /* one competitor: NAV with trade markers, event table synced to the chart */
  async function mountCompetitorChart(el) {
    const id = el.dataset.competitor; const days = parseInt(el.dataset.days || "90", 10);
    const r = await fetch(`/api/series/${id}?days=${days}`); if (!r.ok) return;
    const d = await r.json();
    if (!d.nav.t.length) { el.textContent = "Pas encore de livre pour cette période."; el.classList.add("muted"); return; }
    const colors = palette();
    const tIndex = new Map(d.nav.t.map((t, i) => [t, i]));
    const nearest = (t) => { if (tIndex.has(t)) return tIndex.get(t); let lo = 0, hi = d.nav.t.length - 1; while (lo < hi) { const m = (lo + hi) >> 1; if (d.nav.t[m] < t) lo = m + 1; else hi = m; } return lo; };
    const evAt = new Map(); for (const e of d.events) { const i = nearest(e.t); const k = d.nav.t[i]; if (!evAt.has(k)) evAt.set(k, []); evAt.get(k).push(e); }
    const opts = baseOpts(el, "€", [`${d.name} (€)`], colors);
    opts.plugins = [markersPlugin(() => d.events, nearest)];
    const u = new uPlot(opts, [d.nav.t, d.nav.nav], el);
    const tip = tooltipPlugin(u, el, [`${d.name}`], colors, (v) => fmtEur.format(v) + " €", (t) => evAt.get(t) || []);
    u.hooks.setCursor = [(u) => tip(u.cursor.idx)];
    charts.push({ u, el });
    renderEvents(document.getElementById("events-table"), d.events, u, nearest);
    onTick(async () => { const r2 = await fetch(`/api/series/${id}?days=${days}`); if (!r2.ok) return; const d2 = await r2.json(); d.events = d2.events; u.setData([d2.nav.t, d2.nav.nav]); renderEvents(document.getElementById("events-table"), d2.events, u, nearest); });
  }
  function renderEvents(tbody, events, u, nearest) {
    if (!tbody) return; tbody.textContent = "";
    for (const e of [...events].reverse().slice(0, 200)) {
      const tr = document.createElement("tr"); tr.className = "ev " + e.side; tr.tabIndex = 0;
      const cells = [fmtTs(e.t), `${KIND_GLYPH[e.kind] || "●"} ${e.kind}`, e.symbol, e.side === "long" ? "achat" : "vente", `${(e.before * 100).toFixed(1)} → ${(e.after * 100).toFixed(1)} %`, e.price != null ? String(e.price) : "—", Object.entries(e.reason).filter(([k]) => !["tranches", "vol_scale"].includes(k)).slice(0, 4).map(([k, v]) => `${k}=${typeof v === "number" ? v.toFixed(3) : v}`).join(" · ")];
      cells.forEach((c, i) => { const td = document.createElement("td"); td.textContent = c; if (i === 4 || i === 5) td.className = "n"; tr.appendChild(td); });
      const focus = () => { const i = nearest(e.t); u.setCursor({ left: u.valToPos(u.data[0][i], "x"), top: 10 }); };
      tr.addEventListener("mouseenter", focus); tr.addEventListener("focus", focus);
      tbody.appendChild(tr);
    }
  }

  /* the board: champions and benchmarks normalised to 100 */
  async function mountBoardChart(el) {
    const days = parseInt(el.dataset.days || "90", 10);
    const r = await fetch(`/api/board?days=${days}`); if (!r.ok) return;
    const d = await r.json(); if (!d.series.length) { el.textContent = "Rien à tracer."; return; }
    const all = Array.from(new Set(d.series.flatMap((s) => s.t))).sort((a, b) => a - b);
    const idx = new Map(all.map((t, i) => [t, i]));
    const cols = d.series.slice(0, 8).map((s) => { const col = new Array(all.length).fill(null); s.t.forEach((t, i) => (col[idx.get(t)] = s.v[i])); return col; });
    const names = d.series.slice(0, 8).map((s) => s.name);
    const colors = palette();
    const opts = baseOpts(el, "idx", names, colors);
    opts.series.slice(1).forEach((s, i) => { if (d.series[i].role !== "competitor") { s.dash = [6, 4]; s.width = 1.5; } });
    const u = new uPlot(opts, [all, ...cols], el);
    const tip = tooltipPlugin(u, el, names, colors, (v) => v.toFixed(1), null);
    u.hooks.setCursor = [(u) => tip(u.cursor.idx)];
    legend(el, names, colors, u);
    charts.push({ u, el });
    onTick(() => mountBoardChart.refresh && mountBoardChart.refresh());
  }

  /* ------------------------------------------------------------------ boot */
  function boot() {
    document.querySelectorAll("[data-chart=competitor]").forEach(mountCompetitorChart);
    document.querySelectorAll("[data-chart=board]").forEach(mountBoardChart);
    pollLive(); setInterval(pollLive, POLL_MS); setInterval(renderClocks, 1000);
    addEventListener("resize", () => charts.forEach(({ u, el }) => u.setSize({ width: el.clientWidth, height: u.height })));
    const toggle = document.getElementById("theme-toggle");
    if (toggle) toggle.addEventListener("click", () => { const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"); document.documentElement.dataset.theme = cur === "dark" ? "light" : "dark"; try { localStorage.setItem("arena-theme", document.documentElement.dataset.theme); } catch (e) {} location.reload(); });
  }
  try { const saved = localStorage.getItem("arena-theme"); if (saved) document.documentElement.dataset.theme = saved; } catch (e) {}
  window.Arena = { onTick, countdown, fmtTs, KIND_GLYPH, palette };
  if (document.readyState === "loading") addEventListener("DOMContentLoaded", boot); else boot();
})();
