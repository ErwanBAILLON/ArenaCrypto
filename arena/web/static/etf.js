/* The ETF lab: overlay any number of daily series, each rebased to 100 at its own start,
 * on a relative (days since start) or calendar axis; put buy/sell points on the active
 * curve; price them with user-defined fee profiles. All client side, on plain arrays
 * from /api/etf/series. Layers, points and profiles live in localStorage: a personal
 * scratchpad, not shared state.
 */
(function () {
  "use strict";
  const A = window.Arena; if (!A) return;
  const chartEl = document.getElementById("etf-chart"); if (!chartEl) return;
  const $ = (id) => document.getElementById(id);
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const fmtEur = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 0 });
  const eur = (v) => (v >= 0 ? "+" : "−") + fmtEur.format(Math.abs(v)) + " €";
  const pct = (v, d = 1) => (v >= 0 ? "+" : "−") + (Math.abs(v) * 100).toFixed(d).replace(".", ",") + " %";
  const dayStr = (d) => new Date(d * 86400000).toISOString().slice(0, 10);
  const frDate = (d) => new Date(d * 86400000).toLocaleDateString("fr-FR", { timeZone: "UTC", day: "2-digit", month: "2-digit", year: "numeric" });
  // a bar index `t` in units of `step` seconds -> a French stamp, with the time when the step is intraday
  const frStamp = (t, step) => step >= 86400 ? frDate(t) : new Date(t * step * 1000).toLocaleString("fr-FR", { timeZone: "Europe/Paris", day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
  const relLabel = (i, step) => step >= 86400 ? `j+${i}` : (i * step >= 3600 ? `${Math.floor(i * step / 3600)}h${String(Math.round((i * step % 3600) / 60)).padStart(2, "0")}` : `${Math.round(i * step / 60)} min`);

  const emptyRow = (span, text) => { const tr = document.createElement("tr"); const td = document.createElement("td"); td.colSpan = span; td.className = "muted"; td.textContent = text; tr.appendChild(td); return tr; };

  /* ---- state ---- */
  const load = (k, dflt) => { try { const v = JSON.parse(localStorage.getItem(k)); return v == null ? dflt : v; } catch (e) { return dflt; } };
  const save = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} };
  const DEFAULT_PROFILES = [
    { name: "PEA banque en ligne (exemple)", fixed: 0, pct: 0.5, spread: 5, fx: 0, ter: 0.2 },
    { name: "Courtier étranger (exemple)", fixed: 1, pct: 0, spread: 5, fx: 0.25, ter: 0.2 },
    { name: "Sans frais (référence)", fixed: 0, pct: 0, spread: 0, fx: 0, ter: 0 },
  ];
  let layers = load("etf-layers", []); // {id, symbol, label, start, end, currency, trades:[{d, side, price, stake}]}
  let profiles = load("etf-profiles", DEFAULT_PROFILES);
  let active = load("etf-active", null);
  let xmode = load("etf-xmode", "rel"), scale = load("etf-scale", "100");
  const cache = new Map(); // `${symbol}|${start}|${end}` -> {t, close}
  const colors = () => A.palette();
  let u = null;

  /* ---- data ---- */
  async function fetchSeries(l) {
    const key = `${l.symbol}|${l.tf || "1d"}|${l.start || ""}|${l.end || ""}`;
    if (cache.has(key)) return cache.get(key);
    const q = new URLSearchParams({ symbol: l.symbol, tf: l.tf || "1d" }); if (l.start) q.set("start", l.start); if (l.end) q.set("end", l.end + "T23:59:59");
    const r = await fetch(`/api/etf/series?${q}`); if (!r.ok) return null;
    const d = await r.json(); cache.set(key, d); return d;
  }
  const rebased = (d) => { const b = d.close[0]; return d.close.map((c) => (scale === "pct" ? c / b - 1 : (c / b) * 100)); };

  /* ---- chart ---- */
  async function draw() {
    const series = []; const cols = colors();
    for (const l of layers) { const d = await fetchSeries(l); if (d && d.t.length > 1) series.push({ l, d, y: rebased(d) }); }
    chartEl.textContent = "";
    if (u) { u.destroy(); u = null; }
    if (!series.length) { chartEl.textContent = "Ajoute une courbe."; renderLayers(series); renderPairs(series); return; }
    const step = series[0].d.step || 86400; const mixed = series.some((s) => (s.d.step || 86400) !== step);
    let xs, data;
    if (xmode === "rel") {
      const n = Math.max(...series.map((s) => s.d.t.length));
      xs = Array.from({ length: n }, (_, i) => i);
      data = [xs, ...series.map((s) => xs.map((i) => (i < s.y.length ? s.y[i] : null)))];
    } else {
      // calendar axis in seconds: a 1d and a 5m curve can share it
      const secs = (s) => s.d.t.map((t) => t * (s.d.step || 86400));
      xs = Array.from(new Set(series.flatMap(secs))).sort((a, b) => a - b);
      const idx = new Map(xs.map((t, i) => [t, i]));
      data = [xs, ...series.map((s) => { const col = new Array(xs.length).fill(null); secs(s).forEach((t, i) => (col[idx.get(t)] = s.y[i])); return col; })];
    }
    const names = series.map((s) => layerName(s.l));
    const fg = css("--fg2"), grid = css("--line");
    const fmtY = (v) => (scale === "pct" ? pct(v) : v.toFixed(1));
    const opts = {
      width: chartEl.clientWidth || 800, height: 380, padding: [12, 12, 0, 0],
      cursor: { drag: { x: true, y: false }, points: { size: 8, width: 2, stroke: (uu, i) => cols[(i - 1) % cols.length], fill: css("--card") || "#fff" } },
      legend: { show: false },
      scales: { x: { time: xmode === "abs" }, y: { distr: scale === "log" ? 3 : 1, log: 10 } },
      axes: [
        xmode === "abs"
          ? { stroke: fg, grid: { stroke: grid, width: 1 }, ticks: { stroke: grid }, font: "12px system-ui", values: (uu, vals) => vals.map((v) => step >= 86400 ? new Date(v * 1000).toLocaleDateString("fr-FR", { timeZone: "UTC", month: "short", year: "2-digit" }) : new Date(v * 1000).toLocaleString("fr-FR", { timeZone: "Europe/Paris", day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" })) }
          : { stroke: fg, grid: { stroke: grid, width: 1 }, ticks: { stroke: grid }, font: "12px system-ui", values: (uu, vals) => vals.map((v) => relLabel(v, mixed ? 86400 : step)) },
        { stroke: fg, grid: { stroke: grid, width: 1 }, ticks: { stroke: grid }, font: "12px system-ui", size: 64, values: (uu, vals) => vals.map(fmtY) },
      ],
      series: [{ label: "x" }, ...series.map((s, i) => ({ label: names[i], stroke: cols[i % cols.length], width: s.l.id === active ? 2.6 : 1.6, points: { show: false }, spanGaps: true }))],
      plugins: [pointsPlugin(series, cols)],
      hooks: { setCursor: [(uu) => tooltip(uu, series, names, cols, fmtY)] },
    };
    const plotData = data;
    u = new uPlot(opts, plotData, chartEl);
    const tip = document.createElement("div"); tip.className = "u-tip"; tip.hidden = true; chartEl.appendChild(tip);
    u.over.addEventListener("click", (e) => onChartClick(e, series));
    renderLayers(series); renderPairs(series); renderTrades(); renderResult(series);
  }
  function layerName(l) { return `${l.label}${(l.tf || "1d") === "5m" ? " · 5 min" : ""} · ${l.start || "début"} → ${l.end || "auj."}`; }
  function tooltip(uu, series, names, cols, fmtY) {
    const tip = chartEl.querySelector(".u-tip"); if (!tip) return;
    const i = uu.cursor.idx; if (i == null) { tip.hidden = true; return; }
    tip.textContent = "";
    const step0 = series[0].d.step || 86400;
    const head = document.createElement("div"); head.className = "t"; head.textContent = xmode === "abs" ? frStamp(uu.data[0][i] / step0, step0) : relLabel(i, step0); tip.appendChild(head);
    series.forEach((s, k) => { const v = uu.data[k + 1][i]; if (v == null) return; const row = document.createElement("div"); const sw = document.createElement("i"); sw.style.background = cols[k % cols.length]; const val = document.createElement("b"); val.textContent = fmtY(v); const nm = document.createElement("span"); nm.textContent = " " + names[k] + (xmode === "rel" && s.d.t[i] != null ? ` (${frStamp(s.d.t[i], s.d.step || 86400)})` : ""); row.append(sw, val, nm); tip.appendChild(row); });
    tip.hidden = false;
    const left = uu.cursor.left, w = uu.over.clientWidth; tip.style.left = (left > w * 0.6 ? left - tip.offsetWidth - 14 : left + 14) + "px"; tip.style.top = "8px";
  }
  function pointsPlugin(series, cols) {
    return { hooks: { draw: (uu) => { const ctx = uu.ctx; ctx.save();
      series.forEach((s, k) => { for (const tr of s.l.trades || []) { const i = s.d.t.indexOf(tr.d); if (i < 0) continue; const xv = xmode === "abs" ? tr.d * (s.d.step || 86400) : i; const x = uu.valToPos(xv, "x", true), y = uu.valToPos(s.y[i], "y", true);
        ctx.beginPath(); ctx.fillStyle = tr.side === "buy" ? css("--long") : css("--short"); ctx.strokeStyle = css("--card"); ctx.lineWidth = 2;
        if (tr.side === "buy") { ctx.moveTo(x, y - 9); ctx.lineTo(x - 8, y + 6); ctx.lineTo(x + 8, y + 6); } else { ctx.moveTo(x, y + 9); ctx.lineTo(x - 8, y - 6); ctx.lineTo(x + 8, y - 6); }
        ctx.closePath(); ctx.fill(); ctx.stroke(); } });
      ctx.restore(); } } };
  }
  function onChartClick(e, series) {
    const l = layers.find((x) => x.id === active) || layers[0]; if (!l) return;
    const s = series.find((x) => x.l.id === l.id); if (!s) return;
    const rect = u.over.getBoundingClientRect(); const xv = u.posToVal(e.clientX - rect.left, "x");
    let i;
    if (xmode === "abs") { const d = Math.round(xv / (s.d.step || 86400)); i = s.d.t.reduce((best, t, k) => (Math.abs(t - d) < Math.abs(s.d.t[best] - d) ? k : best), 0); }
    else { i = Math.max(0, Math.min(s.d.t.length - 1, Math.round(xv))); }
    const tr = { d: s.d.t[i], side: $("etf-side").value, price: s.d.close[i], stake: Number($("etf-stake").value) || 0 };
    l.trades = [...(l.trades || []), tr].sort((a, b) => a.d - b.d);
    $("etf-side").value = tr.side === "buy" ? "sell" : "buy"; // the natural next click
    save("etf-layers", layers); draw();
  }

  /* ---- layers table ---- */
  function renderLayers(series) {
    const tb = $("etf-layers"); tb.textContent = ""; const cols = colors();
    if (!layers.length) { tb.appendChild(emptyRow(6, "aucune courbe : ajoute-en une à gauche")); return; }
    layers.forEach((l, k) => {
      const s = series.find((x) => x.l.id === l.id);
      const tr = document.createElement("tr"); tr.className = l.id === active ? "active" : ""; tr.tabIndex = 0;
      const sw = document.createElement("td"); const i = document.createElement("i"); i.className = "sw"; i.style.background = cols[k % cols.length]; sw.appendChild(i);
      const name = document.createElement("td"); name.textContent = l.label;
      const per = document.createElement("td"); per.className = "small"; per.textContent = `${l.start || "début"} → ${l.end || "aujourd'hui"}`;
      const n = document.createElement("td"); n.className = "n"; n.textContent = s ? String(s.d.t.length) : "—";
      const perf = document.createElement("td"); const p = s ? s.d.close[s.d.close.length - 1] / s.d.close[0] - 1 : null; perf.className = "n " + (p == null ? "muted" : p >= 0 ? "pos" : "neg"); perf.textContent = p == null ? "—" : pct(p);
      const act = document.createElement("td"); const del = document.createElement("button"); del.className = "btn small"; del.textContent = "retirer"; del.addEventListener("click", (e) => { e.stopPropagation(); layers = layers.filter((x) => x.id !== l.id); if (active === l.id) active = layers[0]?.id ?? null; save("etf-layers", layers); save("etf-active", active); draw(); }); act.appendChild(del);
      tr.append(sw, name, per, n, perf, act);
      tr.addEventListener("click", () => { active = l.id; save("etf-active", active); $("etf-fx").checked = l.currency !== "EUR"; draw(); });
      tb.appendChild(tr);
    });
    const al = layers.find((x) => x.id === active); $("etf-active-label").textContent = "courbe active : " + (al ? layerName(al) : "aucune");
  }

  /* ---- pair statistics: numbers, not eyes ---- */
  function returns(y) { const r = []; for (let i = 1; i < y.length; i++) r.push(y[i] / y[i - 1] - 1); return r; }
  function renderPairs(series) {
    const tb = $("etf-pairs"); tb.textContent = "";
    if (series.length < 2) { tb.appendChild(emptyRow(5, "deux courbes au moins")); return; }
    for (let a = 0; a < series.length; a++) for (let b = a + 1; b < series.length; b++) {
      const A_ = series[a], B_ = series[b];
      let ya, yb;
      if (xmode === "rel") { const n = Math.min(A_.d.close.length, B_.d.close.length); ya = A_.d.close.slice(0, n); yb = B_.d.close.slice(0, n); }
      else { const mb = new Map(B_.d.t.map((t, i) => [t, B_.d.close[i]])); ya = []; yb = []; A_.d.t.forEach((t, i) => { if (mb.has(t)) { ya.push(A_.d.close[i]); yb.push(mb.get(t)); } }); }
      const ra = returns(ya), rb = returns(yb), n = ra.length;
      let corr = NaN, beta = NaN, z = NaN;
      if (n >= 20) {
        const ma = ra.reduce((p, q) => p + q, 0) / n, mbm = rb.reduce((p, q) => p + q, 0) / n;
        let sab = 0, saa = 0, sbb = 0; for (let i = 0; i < n; i++) { sab += (ra[i] - ma) * (rb[i] - mbm); saa += (ra[i] - ma) ** 2; sbb += (rb[i] - mbm) ** 2; }
        corr = sab / Math.sqrt(saa * sbb); beta = sab / saa;
        const spread = ya.map((v, i) => Math.log(v / ya[0]) - Math.log(yb[i] / yb[0])); const ms = spread.reduce((p, q) => p + q, 0) / spread.length; const sd = Math.sqrt(spread.reduce((p, q) => p + (q - ms) ** 2, 0) / spread.length); z = sd ? (spread[spread.length - 1] - ms) / sd : NaN;
      }
      const tr = document.createElement("tr");
      [`${A_.l.label} ↔ ${B_.l.label}`, Number.isFinite(corr) ? corr.toFixed(2) : "—", Number.isFinite(beta) ? beta.toFixed(2) : "—", Number.isFinite(z) ? (z >= 0 ? "+" : "") + z.toFixed(1) : "—", String(n)].forEach((c, k) => { const td = document.createElement("td"); td.textContent = c; if (k) td.className = "n" + (k === 1 && Number.isFinite(corr) ? (Math.abs(corr) > 0.7 ? " pos" : "") : ""); tr.appendChild(td); });
      tb.appendChild(tr);
    }
  }

  /* ---- trades, fees, result ---- */
  function renderTrades() {
    const tb = $("etf-trades"); tb.textContent = "";
    const l = layers.find((x) => x.id === active); const trades = (l && l.trades) || [];
    if (!trades.length) { tb.appendChild(emptyRow(5, "clique sur le graphe pour poser un point")); return; }
    trades.forEach((t, k) => { const tr = document.createElement("tr"); tr.className = t.side === "buy" ? "long" : "short";
      const del = document.createElement("button"); del.className = "btn small"; del.textContent = "×"; del.addEventListener("click", () => { l.trades.splice(k, 1); save("etf-layers", layers); draw(); });
      [frStamp(t.d, (l.tf || "1d") === "5m" ? 300 : 86400), t.side === "buy" ? "achat" : "vente", t.price.toLocaleString("fr-FR", { maximumFractionDigits: 4 }), fmtEur.format(t.stake) + " €"].forEach((c, i) => { const td = document.createElement("td"); td.textContent = c; if (i >= 2) td.className = "n"; tr.appendChild(td); });
      const td = document.createElement("td"); td.appendChild(del); tr.appendChild(td); tb.appendChild(tr); });
  }
  function profile() { return profiles[Number($("etf-profile").value) || 0] || DEFAULT_PROFILES[2]; }
  function renderResult(series) {
    const box = $("etf-result"); box.textContent = "";
    const l = layers.find((x) => x.id === active); const s = series.find((x) => l && x.l.id === l.id);
    const trades = (l && l.trades) || []; if (!s || !trades.length) return;
    const p = profile(); const fx = $("etf-fx").checked ? p.fx / 100 : 0; const ttf = $("etf-ttf").checked ? 0.003 : 0;
    let gross = 0, fixed = 0, pctFee = 0, spread = 0, ttfFee = 0, fxFee = 0, ter = 0, staked = 0;
    const open = []; // FIFO buys
    const legCost = (stake, isBuy) => { fixed += p.fixed; pctFee += stake * (p.pct / 100); spread += stake * (p.spread / 10000) / 2; fxFee += stake * fx; if (isBuy) ttfFee += stake * ttf; };
    for (const t of trades) {
      if (t.side === "buy") { open.push({ ...t }); staked += t.stake; legCost(t.stake, true); }
      else { let rem = t.stake; legCost(t.stake, false);
        while (rem > 0 && open.length) { const b = open[0]; const use = Math.min(rem, b.stake); gross += use * (t.price / b.price - 1); ter += use * (p.ter / 100) * (((t.d - b.d) * (s.d.step || 86400)) / 86400 / 365); b.stake -= use; rem -= use; if (b.stake <= 1e-9) open.shift(); } }
    }
    const lastP = s.d.close[s.d.close.length - 1], lastD = s.d.t[s.d.t.length - 1]; let unreal = 0, openStake = 0;
    for (const b of open) { unreal += b.stake * (lastP / b.price - 1); ter += b.stake * (p.ter / 100) * (((lastD - b.d) * (s.d.step || 86400)) / 86400 / 365); openStake += b.stake; }
    const fees = fixed + pctFee + spread + ttfFee + fxFee + ter; const net = gross + unreal - fees;
    const kpis = [["Brut réalisé", gross], ["Latent (position ouverte)", unreal], ["Frais, tous postes", -fees], ["Net", net]];
    for (const [label, v] of kpis) { const k = document.createElement("div"); k.className = "kpi"; const val = document.createElement("div"); val.className = "v " + (v >= 0 ? "pos" : "neg"); val.textContent = eur(v); const lb = document.createElement("div"); lb.className = "l"; lb.textContent = label + (label === "Net" && staked ? ` · ${pct(net / staked)} de la mise` : ""); k.append(val, lb); box.appendChild(k); }
    const det = document.createElement("div"); det.className = "small muted"; det.style.gridColumn = "1 / -1";
    det.textContent = `Frais : fixe ${fmtEur.format(fixed)} € · % ordre ${fmtEur.format(pctFee)} € · spread ${fmtEur.format(spread)} € · TTF ${fmtEur.format(ttfFee)} € · change ${fmtEur.format(fxFee)} € · TER ${fmtEur.format(ter)} € — mise engagée ${fmtEur.format(staked)} €${openStake ? `, dont ${fmtEur.format(openStake)} € encore ouverts` : ""}.`;
    box.appendChild(det);
  }

  /* ---- profiles ---- */
  function renderProfiles() {
    const sel = $("etf-profile"); const cur = sel.value; sel.textContent = "";
    profiles.forEach((p, i) => { const o = document.createElement("option"); o.value = String(i); o.textContent = p.name; sel.appendChild(o); }); if (cur) sel.value = cur;
    const tb = $("etf-profiles"); tb.textContent = "";
    profiles.forEach((p, i) => { const tr = document.createElement("tr");
      const fields = [["name", "text"], ["fixed", "number"], ["pct", "number"], ["spread", "number"], ["fx", "number"], ["ter", "number"]];
      fields.forEach(([f, type]) => { const td = document.createElement("td"); if (type === "number") td.className = "n"; const inp = document.createElement("input"); inp.type = type; inp.value = p[f]; inp.id = `etf-p-${i}-${f}`; if (type === "number") { inp.step = "0.01"; inp.min = "0"; inp.style.width = "5.5em"; } inp.addEventListener("change", () => { p[f] = type === "number" ? Number(inp.value) : inp.value; save("etf-profiles", profiles); renderProfiles(); draw(); }); td.appendChild(inp); tr.appendChild(td); });
      const td = document.createElement("td"); const del = document.createElement("button"); del.className = "btn small"; del.textContent = "×"; del.addEventListener("click", () => { profiles.splice(i, 1); if (!profiles.length) profiles = [...DEFAULT_PROFILES]; save("etf-profiles", profiles); renderProfiles(); draw(); }); td.appendChild(del); tr.appendChild(td);
      tb.appendChild(tr); });
  }

  /* ---- wiring ---- */
  $("etf-add").addEventListener("click", () => {
    const sel = $("etf-symbol"); const opt = sel.selectedOptions[0];
    const l = { id: Date.now().toString(36), symbol: sel.value, tf: $("etf-tf").value, label: opt.textContent.replace(/ \(pas encore de données\)$/, ""), start: $("etf-start").value || null, end: $("etf-end").value || null, currency: opt.dataset.currency || "USD", trades: [] };
    layers.push(l); active = l.id; $("etf-fx").checked = l.currency !== "EUR"; save("etf-layers", layers); save("etf-active", active); draw();
  });
  $("etf-symbol").addEventListener("change", () => { const o = $("etf-symbol").selectedOptions[0]; if (o.dataset.first && !$("etf-start").value) $("etf-start").min = o.dataset.first; });
  $("etf-xmode").value = xmode; $("etf-scale").value = scale;
  $("etf-xmode").addEventListener("change", () => { xmode = $("etf-xmode").value; save("etf-xmode", xmode); draw(); });
  $("etf-scale").addEventListener("change", () => { scale = $("etf-scale").value; save("etf-scale", scale); draw(); });
  ["etf-stake", "etf-profile", "etf-fx", "etf-ttf"].forEach((id) => $(id).addEventListener("change", () => draw()));
  // presets fill the dates; the person still picks the line and the granularity
  const iso = (d) => d.toISOString().slice(0, 10);
  const shift = (d, n) => new Date(d.getTime() + n * 86400000);
  const monday = (d) => shift(d, -((d.getUTCDay() + 6) % 7));
  document.querySelectorAll("[data-preset]").forEach((b) => b.addEventListener("click", () => {
    const now = new Date(); let s, e;
    switch (b.dataset.preset) {
      case "today": s = e = now; break;
      case "yesterday": s = e = shift(now, now.getUTCDay() === 1 ? -3 : -1); break;
      case "lastweekday": s = e = shift(now, -7); break;
      case "thisweek": s = monday(now); e = now; break;
      case "lastweek": s = shift(monday(now), -7); e = shift(monday(now), -3); break;
      case "ytd": s = new Date(Date.UTC(now.getUTCFullYear(), 0, 1)); e = now; break;
    }
    $("etf-start").value = iso(s); $("etf-end").value = iso(e);
    if (["today", "yesterday", "lastweekday", "thisweek", "lastweek"].includes(b.dataset.preset)) $("etf-tf").value = "5m";
  }));
  $("etf-profile-add").addEventListener("click", () => { profiles.push({ name: "Nouveau profil", fixed: 0, pct: 0, spread: 0, fx: 0, ter: 0 }); save("etf-profiles", profiles); renderProfiles(); });
  addEventListener("resize", () => { if (u) u.setSize({ width: chartEl.clientWidth, height: 380 }); });
  renderProfiles();
  if (!layers.length) { // a first look that shows what the lab does: the same index twice, on two different years
    const opts = [...$("etf-symbol").options]; const spy = opts.find((o) => o.value === "SPY") || opts[0];
    if (spy) layers = [
      { id: "a", symbol: spy.value, label: spy.textContent, start: "2022-01-03", end: "2022-12-30", currency: "USD", trades: [] },
      { id: "b", symbol: spy.value, label: spy.textContent, start: "2025-01-02", end: null, currency: "USD", trades: [] },
    ], active = "b";
  }
  draw();
})();
