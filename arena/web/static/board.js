/* The agents board: every stored book marked to market on the exchange's public price stream.
 *
 * Two clocks, kept apart on purpose. Perp prices are polled every two seconds
 * from Binance's public futures REST (one call, every pair; the futures
 * websocket opens but streams nothing from some regions, so the spot
 * market-data websocket is the fallback) and re-value each agent's *stored* book
 * (nav × (1 + Σ w·(p/p_ref − 1))) -- that is what the agent would be worth
 * right now, an honest mark, not a forecast. Decisions (buys, sells) are made
 * by the hourly tick and appear when /api/live reports a new version; the
 * board then refetches /api/book so the reference prices and weights move on.
 *
 * Symbols without a public live stream (equity tickers) keep their last hourly
 * close and the row says so. No library: a table, a websocket, FLIP reorder.
 */
(function () {
  "use strict";
  const A = window.Arena; if (!A) return;
  const body = document.getElementById("board-body"); if (!body) return;
  const table = document.getElementById("board");
  const fmtEur2 = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 0 });
  const fmtPx = (v) => (v >= 100 ? v.toLocaleString("fr-FR", { maximumFractionDigits: 2 }) : v.toLocaleString("fr-FR", { maximumSignificantDigits: 5 }));
  const eur = (v) => (v >= 0 ? "+" : "−") + fmtEur2.format(Math.abs(v)) + " €";
  const pct = (v) => (v >= 0 ? "+" : "−") + (Math.abs(v) * 100).toFixed(2).replace(".", ",") + " %";

  let agents = [], ws = null, wsUrl = null, restUrl = null, pairs = [], nextTick = null, backoff = 1000, restTimer = null, restFails = 0;
  const POLL_PRICES_MS = 2000;
  const prices = new Map(); // pair (BTCUSDT) -> last price
  const expanded = new Set(); // agent ids whose positions are shown
  const last = new Map(); // agent id -> last live nav, to flash up/down
  let sortKey = "all", sortDir = -1, renderQueued = false;

  /* mark-to-market of one stored book */
  function liveNav(a) {
    let f = 1.0;
    for (const p of a.positions) {
      const px = p.pair ? prices.get(p.pair) : null;
      if (px && p.ref_price) f += p.weight * (px / p.ref_price - 1);
    }
    return a.nav * f;
  }
  function posPnl(a, p) {
    const px = p.pair ? prices.get(p.pair) : null;
    if (!px || !p.ref_price) return null;
    return a.nav * p.weight * (px / p.ref_price - 1);
  }
  const ret = (a, key, nav) => (a.base[key] ? nav / a.base[key] - 1 : null);
  const metric = (a, key) => {
    const nav = liveNav(a);
    switch (key) {
      case "name": return a.name;
      case "live": return nav;
      case "today": return ret(a, "today", nav);
      case "d7": return ret(a, "7d", nav);
      case "d30": return ret(a, "30d", nav);
      case "all": return ret(a, "all", nav);
      case "psr": return a.psr;
      case "npos": return a.positions.length;
      default: return nav;
    }
  };
  function sorted() {
    const rows = agents.slice();
    rows.sort((x, y) => {
      if (x.role !== y.role) return x.role === "competitor" ? -1 : 1; // benchmarks after agents
      const a = metric(x, sortKey), b = metric(y, sortKey);
      if (a == null && b == null) return 0; if (a == null) return 1; if (b == null) return -1;
      return typeof a === "string" ? a.localeCompare(b) * -sortDir : (a - b) * sortDir;
    });
    return rows;
  }

  function cell(text, cls) { const td = document.createElement("td"); td.textContent = text; if (cls) td.className = cls; return td; }
  function retCell(a, key, nav) {
    const v = ret(a, key, nav);
    if (v == null) return cell("—", "n muted");
    const td = cell(pct(v), "n " + (v >= 0 ? "pos" : "neg"));
    const s = document.createElement("div"); s.className = "sub"; s.textContent = eur(nav - a.base[key]); td.appendChild(s);
    return td;
  }

  function rowFor(a, rank) {
    const nav = liveNav(a);
    const tr = document.createElement("tr"); tr.className = "agent" + (a.role !== "competitor" ? " bench" : ""); tr.dataset.id = a.id;
    tr.appendChild(cell(String(rank), "n rank"));
    const name = document.createElement("td");
    const link = document.createElement("a"); link.href = `/competitors/${a.id}`; link.textContent = a.name; link.addEventListener("click", (e) => e.stopPropagation());
    const pill = document.createElement("span"); pill.className = "pill " + (a.role === "competitor" ? a.status : "benchmark");
    pill.textContent = a.role !== "competitor" ? "repère" : a.status === "champion" ? "champion" : "prétendant";
    const fam = document.createElement("span"); fam.className = "muted small"; fam.textContent = " " + a.family;
    name.append(link, " ", pill, fam); tr.appendChild(name);
    const prev = last.get(a.id); last.set(a.id, nav);
    const live = cell(fmtEur2.format(nav) + " €", "n live" + (prev != null && nav !== prev ? (nav > prev ? " up" : " down") : ""));
    const lv = a.positions.length && !a.positions.some((p) => p.pair) ? "cours horaire" : "";
    if (lv) { const s = document.createElement("div"); s.className = "sub muted"; s.textContent = lv; live.appendChild(s); }
    tr.appendChild(live);
    tr.appendChild(retCell(a, "today", nav)); tr.appendChild(retCell(a, "7d", nav)); tr.appendChild(retCell(a, "30d", nav)); tr.appendChild(retCell(a, "all", nav));
    tr.appendChild(cell(a.psr == null ? "—" : Math.round(a.psr * 100) + " %", "n " + (a.psr >= 0.95 ? "pos" : "muted")));
    const np = cell(a.positions.length ? `${a.positions.length} ▸` : "cash", "n npos"); tr.appendChild(np);
    tr.addEventListener("click", () => { expanded.has(a.id) ? expanded.delete(a.id) : expanded.add(a.id); render(true); });
    return tr;
  }

  function detailFor(a) {
    const tr = document.createElement("tr"); tr.className = "detail"; tr.dataset.for = a.id;
    const td = document.createElement("td"); td.colSpan = 9;
    if (!a.positions.length) { td.textContent = "En cash : aucune position tenue."; td.className = "muted"; tr.appendChild(td); return tr; }
    const t = document.createElement("table"); t.className = "positions";
    const h = document.createElement("thead"); const hr = document.createElement("tr");
    for (const [txt, cls] of [["Actif", ""], ["Sens", ""], ["Poids", "n"], ["Prix de référence", "n"], ["Prix courant", "n"], ["Gain / perte", "n"], ["Tenue depuis", ""]]) { const th = document.createElement("th"); th.textContent = txt; th.className = cls; hr.appendChild(th); }
    h.appendChild(hr); t.appendChild(h);
    const tb = document.createElement("tbody");
    for (const p of a.positions) {
      const r = document.createElement("tr"); r.className = p.weight > 0 ? "long" : "short";
      const px = p.pair ? prices.get(p.pair) : null; const pnl = posPnl(a, p);
      r.appendChild(cell(p.symbol.replace(/USDT$/, ""), "sym"));
      r.appendChild(cell(p.weight > 0 ? "achat" : "vente", p.weight > 0 ? "pos" : "neg"));
      r.appendChild(cell((Math.abs(p.weight) * 100).toFixed(1) + " %", "n"));
      r.appendChild(cell(p.ref_price ? fmtPx(p.ref_price) : "—", "n muted"));
      r.appendChild(cell(px ? fmtPx(px) : p.pair ? "…" : "pas de cours live", "n" + (px ? "" : " muted")));
      r.appendChild(cell(pnl == null ? "—" : eur(pnl), "n " + (pnl == null ? "muted" : pnl >= 0 ? "pos" : "neg")));
      r.appendChild(cell(p.since ? new Date(p.since).toLocaleString("fr-FR", { timeZone: "UTC", day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" }) + " UTC" : "—", "muted small"));
      tb.appendChild(r);
    }
    t.appendChild(tb); td.appendChild(t); tr.appendChild(td);
    return tr;
  }

  /* FLIP: remember where each row was, re-render, slide it from there */
  function render(force) {
    renderQueued = false;
    const before = new Map(); for (const tr of body.querySelectorAll("tr.agent")) before.set(tr.dataset.id, tr.getBoundingClientRect().top);
    const frag = document.createDocumentFragment();
    sorted().forEach((a, i) => { frag.appendChild(rowFor(a, i + 1)); if (expanded.has(a.id)) frag.appendChild(detailFor(a)); });
    if (!agents.length) { const tr = document.createElement("tr"); tr.appendChild(cell("Aucun agent actif.", "muted")); frag.appendChild(tr); }
    body.replaceChildren(frag);
    for (const tr of body.querySelectorAll("tr.agent")) {
      const was = before.get(tr.dataset.id); if (was == null) continue;
      const dy = was - tr.getBoundingClientRect().top;
      if (Math.abs(dy) > 1) { tr.style.transform = `translateY(${dy}px)`; tr.style.transition = "none"; requestAnimationFrame(() => { tr.style.transition = "transform .45s ease"; tr.style.transform = ""; }); }
    }
    for (const th of table.querySelectorAll("th[data-sort]")) th.classList.toggle("sorted", th.dataset.sort === sortKey), th.classList.toggle("asc", th.dataset.sort === sortKey && sortDir === 1);
  }
  function queueRender() { if (renderQueued) return; renderQueued = true; setTimeout(render, 250); }

  /* the price source: futures REST first, spot websocket if that fails */
  function setWs(state, label) {
    const dot = document.getElementById("ws-dot"), lab = document.getElementById("ws-label");
    if (dot) dot.className = "live-dot " + state; if (lab) lab.textContent = label;
  }
  const wanted = () => new Set(pairs);
  async function pollPrices() {
    if (!restUrl) return;
    try {
      const r = await fetch(restUrl, { cache: "no-store" }); if (!r.ok) throw new Error(r.status);
      const want = wanted(); let n = 0;
      for (const row of await r.json()) if (want.has(row.symbol)) { prices.set(row.symbol, parseFloat(row.price)); n++; }
      restFails = 0; queueRender();
      setWs("ok", `cours perp Binance · ${n} actif${n > 1 ? "s" : ""} · toutes les 2 s`);
      if (ws) { try { ws.onclose = null; ws.close(); } catch (e) {} ws = null; }
    } catch (e) {
      if (++restFails === 3) { setWs("warn", "REST Binance injoignable, bascule sur le flux spot…"); connect(); }
    }
  }
  function connect() {
    if (ws) { try { ws.onclose = null; ws.close(); } catch (e) {} ws = null; }
    if (!wsUrl) return;
    try { ws = new WebSocket(wsUrl); } catch (e) { setWs("bad", "cours live indisponibles"); return; }
    ws.onopen = () => { backoff = 1000; setWs("ok", `cours spot Binance en direct · ${pairs.length} actif${pairs.length > 1 ? "s" : ""} (le livre est marqué sur les perps : écart de base possible)`); };
    ws.onmessage = (m) => {
      try { const d = JSON.parse(m.data); if (d.data && d.data.s && d.data.c) { prices.set(d.data.s, parseFloat(d.data.c)); queueRender(); } } catch (e) {}
    };
    ws.onclose = () => { setWs("warn", "flux spot coupé, reconnexion…"); setTimeout(connect, backoff); backoff = Math.min(backoff * 2, 30000); };
    ws.onerror = () => { try { ws.close(); } catch (e) {} };
  }
  function startPrices() {
    if (restTimer) clearInterval(restTimer);
    if (!restUrl && !wsUrl) { setWs("warn", "aucun cours live : valeurs au dernier cours horaire"); return; }
    setWs("warn", "connexion aux cours…");
    pollPrices(); restTimer = setInterval(pollPrices, POLL_PRICES_MS);
    if (!restUrl) connect();
  }

  async function load() {
    const r = await fetch("/api/book", { cache: "no-store" }); if (!r.ok) return;
    const d = await r.json();
    agents = d.agents; nextTick = d.next_tick;
    const same = d.ws === wsUrl && d.rest === restUrl; wsUrl = d.ws; restUrl = d.rest; pairs = d.pairs;
    render(true);
    if (!same) startPrices();
  }

  for (const th of table.querySelectorAll("th[data-sort]")) th.addEventListener("click", () => {
    const k = th.dataset.sort === "rank" ? sortKey : th.dataset.sort;
    if (k === sortKey) sortDir = -sortDir; else { sortKey = k; sortDir = k === "name" ? 1 : -1; }
    render(true);
  });
  setInterval(() => { const c = document.getElementById("board-countdown"); if (c && nextTick) c.textContent = A.countdown(nextTick); }, 1000);
  A.onTick(load); // a tick wrote: new weights, new reference prices, maybe new agents
  load();
})();
