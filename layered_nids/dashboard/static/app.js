(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var SEV = ["critical", "high", "medium", "low"];
  var SEV_COLOR = { critical: "var(--critical)", high: "var(--high)", medium: "var(--medium)", low: "var(--low)" };
  var LAYER_COLOR = { packets: "var(--c-packets)", rules: "var(--c-rules)", flows: "var(--c-flows)" };
  var LAYER_NAME = { packets: "Packet anomaly model", rules: "Signatures and rules", flows: "Flow model" };
  var REDUCED = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ------------------------------------------------------------------------------------------------ helpers
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  function svg(tag, attrs, text) {
    var e = document.createElementNS(NS, tag);
    for (var k in (attrs || {})) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }
  var SRC = "recorded"; try { if (localStorage.getItem("nids-src") === "live") SRC = "live"; } catch (e) { /* storage may be blocked */ }
  function api(path) {
    if (SRC === "live" && path.indexOf("/nids/api/") === 0 && path.indexOf("/nids/api/capture") !== 0) path += (path.indexOf("?") < 0 ? "?" : "&") + "source=live";
    return fetch(path, { credentials: "same-origin" }).then(function (r) {
      if (r.status === 401) { location.href = "/login"; throw new Error("signed out"); }
      return r.json();
    });
  }
  function two(n) { return n < 10 ? "0" + n : String(n); }
  function utc(ts) { var d = new Date(ts * 1000); return { date: d.getUTCFullYear() + "-" + two(d.getUTCMonth() + 1) + "-" + two(d.getUTCDate()), time: two(d.getUTCHours()) + ":" + two(d.getUTCMinutes()) + ":" + two(d.getUTCSeconds()) }; }
  function clock(ts) { return utc(ts).time.slice(0, 5); }
  function nines(p) { return p >= 1 ? 12 : Math.min(12, Math.max(0, -Math.log10(Math.max(1 - p, 1e-12)))); }
  function confidence(p) {
    var n = nines(p), box = el("div", "meter"), t = el("div", "t"), i = el("i");
    i.style.width = Math.round(100 * Math.min(1, n / 12)) + "%"; t.appendChild(i); box.appendChild(t);
    box.appendChild(el("span", null, p >= 0.999 ? Math.floor(n) + " nines" : (100 * p).toFixed(1) + "%"));
    box.title = "probability " + p;
    return box;
  }
  function pill(kind, text) { return el("span", "pill " + kind, text || kind); }
  function empty(title, hint) { var d = el("div", "empty"); d.appendChild(el("b", null, title)); d.appendChild(el("span", null, hint || "")); return d; }
  function skeleton(box, n) { box.textContent = ""; for (var i = 0; i < n; i++) box.appendChild(el("div", "skel")); }
  function thead(tab, cols) { tab.textContent = ""; var tr = el("tr"); cols.forEach(function (c) { tr.appendChild(el("th", null, c)); }); var h = el("thead"); h.appendChild(tr); tab.appendChild(h); var b = el("tbody"); tab.appendChild(b); return b; }
  function td(content, cls) { var c = el("td", cls); if (content instanceof Node) c.appendChild(content); else c.textContent = content; return c; }
  function whenCell(ts) { var u = utc(ts), c = el("td", "when"); c.appendChild(el("b", null, u.time)); c.appendChild(document.createTextNode(u.date)); return c; }
  var timer = null;
  function later(fn) { clearTimeout(timer); timer = setTimeout(fn, 250); }
  function segValue(id, attr) { var on = document.querySelector("#" + id + " button.on"); return on ? on.getAttribute(attr) : ""; }
  function wireSeg(id, fn) {
    document.querySelectorAll("#" + id + " button").forEach(function (b) {
      b.addEventListener("click", function () {
        document.querySelectorAll("#" + id + " button").forEach(function (x) { x.classList.toggle("on", x === b); });
        if (fn) fn();
      });
    });
  }
  function countUp(node, to) {
    if (REDUCED || to < 2) { node.textContent = String(to); return; }
    var t0 = Date.now(), ms = 800;
    function step() { var f = Math.min(1, (Date.now() - t0) / ms); node.textContent = String(Math.round(to * (1 - Math.pow(1 - f, 3)))); if (f < 1) setTimeout(step, 30); }
    step();
  }
  function toast(text) {
    var t = el("div", "toast", text); document.getElementById("toasts").appendChild(t);
    setTimeout(function () { t.style.transition = "opacity .3s"; t.style.opacity = "0"; setTimeout(function () { t.remove(); }, 320); }, 4200);
  }
  // one shared tooltip for anything with data-tip ("title|text")
  var tip = document.getElementById("tip");
  document.addEventListener("mousemove", function (e) {
    var n = e.target.closest ? e.target.closest("[data-tip]") : null;
    if (!n) { tip.classList.remove("on"); return; }
    var parts = n.getAttribute("data-tip").split("|"); tip.textContent = "";
    tip.appendChild(el("b", null, parts[0])); if (parts[1]) tip.appendChild(document.createTextNode(parts[1]));
    tip.classList.add("on");
    var x = e.clientX + 14, y = e.clientY + 14; if (x + 270 > window.innerWidth) x = e.clientX - 274;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  });

  // ------------------------------------------------------------------------------------------------ overview
  var STATS = null, SUMMARY = null;
  function overview(initial) {
    if (initial) { skeleton(document.getElementById("ch-time"), 6); }
    return Promise.all([api("/nids/api/summary"), api("/nids/api/stats"), api("/nids/api/alerts?limit=8")]).then(function (r) {
      SUMMARY = r[0]; STATS = r[1];
      statusText(SUMMARY);
      var rng = document.getElementById("range");
      rng.textContent = STATS.t1 > 0 ? "Data from " + utc(STATS.t0).date + " " + clock(STATS.t0) + " to " + utc(STATS.t1).date + " " + clock(STATS.t1) + " UTC" : "No data loaded yet.";
      gauge(STATS); kpis(SUMMARY, STATS); timeChart(STATS); donut(STATS); hostMap(STATS); hostBars(STATS); layerBars(STATS); ruleBars(STATS); latest(r[2]);
    });
  }
  function statusText(s) { LAST_COUNT = s.alerts + s.events; }
  var LAST_COUNT = -1;

  function spark(series, color) {
    var W = 200, H = 34, max = Math.max.apply(null, series) || 1, n = series.length;
    var s = svg("svg", { viewBox: "0 0 " + W + " " + H, class: "spark", preserveAspectRatio: "none", "aria-hidden": "true" });
    var pts = series.map(function (v, i) { return [(i / (n - 1)) * W, H - 3 - (H - 8) * (v / max)]; });
    var line = pts.map(function (p, i) { return (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1); }).join(" ");
    var id = "g" + Math.random().toString(36).slice(2, 8);
    var defs = svg("defs"), lg = svg("linearGradient", { id: id, x1: 0, y1: 0, x2: 0, y2: 1 });
    lg.appendChild(svg("stop", { offset: "0", "stop-color": color, "stop-opacity": 0.45 })); lg.appendChild(svg("stop", { offset: "1", "stop-color": color, "stop-opacity": 0 })); defs.appendChild(lg); s.appendChild(defs);
    s.appendChild(svg("path", { d: line + " L" + W + " " + H + " L0 " + H + " Z", fill: "url(#" + id + ")" }));
    s.appendChild(svg("path", { d: line, fill: "none", stroke: color, "stroke-width": 1.6, "stroke-linejoin": "round", "vector-effect": "non-scaling-stroke" }));
    return s;
  }
  function kpis(s, st) {
    var box = document.getElementById("kpis"); box.textContent = "";
    var h = st.hist;
    var items = [["Host alerts", s.alerts, "in " + s.alert_hosts + " host" + (s.alert_hosts === 1 ? "" : "s"), "var(--critical)", h.map(function (b) { return b.critical + b.high + b.medium + b.low; })],
                 ["Critical", (s.severity && s.severity.critical) || 0, "highest severity", "var(--high)", h.map(function (b) { return b.critical; })],
                 ["Rule events", s.events, "on " + s.event_hosts + " host" + (s.event_hosts === 1 ? "" : "s"), "var(--c-rules)", h.map(function (b) { return b.events; })],
                 ["Signature hits", s.signature_events, "known-bad patterns", "var(--c-packets)", h.map(function (b) { return b.signatures || 0; })]];
    items.forEach(function (x, i) {
      var c = el("div", "kpi rise"); c.style.setProperty("--kc", x[3]); c.style.setProperty("--i", i);
      var n = el("div", "n"); c.appendChild(n); countUp(n, x[1]);
      c.appendChild(el("div", "l", x[0])); c.appendChild(el("div", "s", x[2])); c.appendChild(spark(x[4], x[3]));
      box.appendChild(c);
    });
  }
  function gauge(st) {
    var box = document.getElementById("gauge"); box.textContent = "";
    var tot = st.hist.map(function (b) { return b.critical + b.high + b.medium + b.low; }), peak = Math.max.apply(null, tot) || 0, last = tot.length ? tot[tot.length - 1] : 0;
    var f = peak ? last / peak : 0;
    var s = svg("svg", { viewBox: "0 0 120 74", role: "img", "aria-label": "latest alert activity compared with the peak" });
    var arc = function (a0, a1) { var r = 50, cx = 60, cy = 62, p = function (a) { return [cx + r * Math.cos(Math.PI * (1 - a)), cy - r * Math.sin(Math.PI * (1 - a))]; }, A = p(a0), B = p(a1); return "M" + A[0] + " " + A[1] + " A" + r + " " + r + " 0 0 1 " + B[0] + " " + B[1]; };
    s.appendChild(svg("path", { d: arc(0, 1), fill: "none", stroke: "var(--line-strong)", "stroke-width": 9, "stroke-linecap": "round" }));
    var col = f > 0.66 ? "var(--critical)" : f > 0.33 ? "var(--medium)" : "var(--ok)";
    if (f > 0.01) { var p = svg("path", { d: arc(0, Math.max(f, 0.02)), fill: "none", stroke: col, "stroke-width": 9, "stroke-linecap": "round" }); if (!REDUCED) { p.style.setProperty("--len", "180"); p.classList.add("draw"); } s.appendChild(p); }
    box.appendChild(s);
    var t = el("div"); t.appendChild(el("b", null, Math.round(100 * f) + "%")); t.appendChild(el("span", null, "latest window vs peak")); t.appendChild(el("span", null, last + " alert" + (last === 1 ? "" : "s") + " now, peak " + peak));
    box.appendChild(t);
  }
  function timeChart(st) {
    var box = document.getElementById("ch-time"); box.textContent = "";
    var hist = st.hist, W = 760, H = 240, L = 34, B = 26, T = 10, R = 6;
    var max = 1; hist.forEach(function (b) { max = Math.max(max, b.critical + b.high + b.medium + b.low, b.events); });
    var ymax = Math.ceil(max / 4) * 4 || 4;
    var s = svg("svg", { viewBox: "0 0 " + W + " " + H, class: "chart", role: "img", "aria-label": "alerts over time" });
    var defs = svg("defs"), lg = svg("linearGradient", { id: "evg", x1: 0, y1: 0, x2: 0, y2: 1 });
    lg.appendChild(svg("stop", { offset: "0", "stop-color": "var(--c-rules)", "stop-opacity": 0.28 })); lg.appendChild(svg("stop", { offset: "1", "stop-color": "var(--c-rules)", "stop-opacity": 0 })); defs.appendChild(lg); s.appendChild(defs);
    for (var g = 0; g <= 4; g++) {
      var y = T + (H - T - B) * (1 - g / 4);
      s.appendChild(svg("line", { x1: L, x2: W - R, y1: y, y2: y, stroke: "var(--line)", "stroke-width": 1, "stroke-dasharray": g ? "3 5" : "0" }));
      s.appendChild(svg("text", { x: L - 6, y: y + 4, "text-anchor": "end" }, String(Math.round(ymax * g / 4))));
    }
    var bw = (W - L - R) / hist.length;
    var pts = hist.map(function (b, i) { return [L + i * bw + bw / 2, T + (H - T - B) * (1 - b.events / ymax)]; });
    var line = pts.map(function (p, i) { return (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1); }).join(" ");
    s.appendChild(svg("path", { d: line + " L" + pts[pts.length - 1][0] + " " + (H - B) + " L" + pts[0][0] + " " + (H - B) + " Z", fill: "url(#evg)" }));
    hist.forEach(function (b, i) {
      var x = L + i * bw, base = H - B, tot = b.critical + b.high + b.medium + b.low;
      var tip = clock(b.t0) + "–" + clock(b.t1) + " UTC|" + tot + " alert" + (tot === 1 ? "" : "s") + (tot ? " (" + b.critical + " critical)" : "") + ", " + b.events + " rule event" + (b.events === 1 ? "" : "s");
      var grp = svg("g", { "data-tip": tip });
      grp.appendChild(svg("rect", { x: x, y: T, width: bw, height: H - T - B, fill: "transparent" }));
      SEV.slice().reverse().forEach(function (k) {
        var h = (H - T - B) * (b[k] / ymax); if (h <= 0) return; base -= h;
        var r = svg("rect", { x: x + 1, y: base, width: Math.max(bw - 2, 1), height: h, fill: SEV_COLOR[k], rx: 2, class: REDUCED ? "" : "bar-grow", style: "--d:" + i }); grp.appendChild(r);
      });
      s.appendChild(grp);
    });
    var pl = svg("path", { d: line, fill: "none", stroke: "var(--c-rules)", "stroke-width": 2, "stroke-linejoin": "round", "pointer-events": "none" });
    if (!REDUCED) { pl.style.setProperty("--len", "2600"); pl.classList.add("draw"); }
    s.appendChild(pl);
    [0, 0.25, 0.5, 0.75, 1].forEach(function (f) {
      s.appendChild(svg("text", { x: L + (W - L - R) * f, y: H - 8, "text-anchor": f === 0 ? "start" : f === 1 ? "end" : "middle" }, clock(st.t0 + (st.t1 - st.t0) * f)));
    });
    box.appendChild(s);
    var lgd = document.getElementById("lg-time"); lgd.textContent = "";
    SEV.forEach(function (k) { var i = el("span"), sw = el("i"); sw.style.background = SEV_COLOR[k]; i.appendChild(sw); i.appendChild(document.createTextNode(k)); lgd.appendChild(i); });
    var ev = el("span"), ln = el("i"); ln.style.background = "var(--c-rules)"; ev.appendChild(ln); ev.appendChild(document.createTextNode("rule events (line)")); lgd.appendChild(ev);
  }
  function donut(st) {
    var box = document.getElementById("ch-sev"); box.textContent = "";
    var total = 0; SEV.forEach(function (k) { total += st.severity[k] || 0; });
    var s = svg("svg", { viewBox: "0 0 42 42", role: "img", "aria-label": "alerts by severity" });
    s.appendChild(svg("circle", { cx: 21, cy: 21, r: 15.9155, fill: "none", stroke: "var(--surface-2)", "stroke-width": 5 }));
    var off = 25;
    if (total) SEV.forEach(function (k) {
      var v = st.severity[k] || 0; if (!v) return; var share = 100 * v / total;
      s.appendChild(svg("circle", { cx: 21, cy: 21, r: 15.9155, fill: "none", stroke: SEV_COLOR[k], "stroke-width": 5, "stroke-dasharray": Math.max(share - 0.6, 0.1) + " " + (100 - share + 0.6), "stroke-dashoffset": off, "stroke-linecap": "round", "data-tip": k + "|" + v + " alerts (" + Math.round(share) + "%)" }));
      off -= share;
    });
    s.appendChild(svg("text", { x: 21, y: 22.5, "text-anchor": "middle", style: "fill:var(--text);font-size:8px;font-weight:700" }, String(total)));
    s.appendChild(svg("text", { x: 21, y: 28, "text-anchor": "middle", style: "font-size:3.4px;fill:var(--muted)" }, "alerts"));
    box.appendChild(s);
    var lg = el("div", "legend");
    SEV.forEach(function (k) { var i = el("span"), sw = el("i"); sw.style.background = SEV_COLOR[k]; i.appendChild(sw); i.appendChild(document.createTextNode(k + "  " + (st.severity[k] || 0))); lg.appendChild(i); });
    box.appendChild(lg);
  }
  function hostMap(st) {
    var box = document.getElementById("ch-map"); box.textContent = "";
    var hosts = st.top_hosts.slice(0, 8);
    if (!hosts.length) { box.appendChild(empty("No hosts yet")); return; }
    var W = 520, H = 300, cx = W / 2, cy = H / 2, R = 112, max = Math.max.apply(null, hosts.map(function (h) { return h.alerts + h.events * 0.05; })) || 1;
    var s = svg("svg", { viewBox: "0 0 " + W + " " + H, class: "map", role: "img", "aria-label": "hosts around the sensor" });
    [R * 0.55, R, R * 1.45].forEach(function (r) { s.appendChild(svg("ellipse", { cx: cx, cy: cy, rx: r * 1.55, ry: r * 0.92, fill: "none", stroke: "var(--line)", "stroke-dasharray": "2 6" })); });
    hosts.forEach(function (h, i) {
      var a = (i / hosts.length) * Math.PI * 2 - Math.PI / 2, x = cx + Math.cos(a) * R * 1.55, y = cy + Math.sin(a) * R * 0.92;
      var col = SEV_COLOR[h.max_severity] || "var(--c-other)";
      s.appendChild(svg("line", { x1: cx, y1: cy, x2: x, y2: y, stroke: col, "stroke-opacity": 0.35, "stroke-width": 1.2 }));
    });
    var core = svg("g"); core.appendChild(svg("circle", { cx: cx, cy: cy, r: 22, fill: "var(--accent-soft)", stroke: "var(--accent)" })); core.appendChild(svg("text", { x: cx, y: cy + 4, "text-anchor": "middle", style: "fill:var(--accent);font:700 11px system-ui" }, "NIDS")); s.appendChild(core);
    hosts.forEach(function (h, i) {
      var a = (i / hosts.length) * Math.PI * 2 - Math.PI / 2, x = cx + Math.cos(a) * R * 1.55, y = cy + Math.sin(a) * R * 0.92;
      var col = SEV_COLOR[h.max_severity] || "var(--c-other)", r = 7 + 14 * Math.sqrt((h.alerts + h.events * 0.05) / max);
      var g = svg("g", { class: "node", "data-tip": h.host + "|" + h.alerts + " alerts, " + h.events + " rule events" });
      g.appendChild(svg("circle", { class: "halo", cx: x, cy: y, r: r + 5, fill: col, opacity: 0.18 }));
      g.appendChild(svg("circle", { cx: x, cy: y, r: r, fill: col, opacity: 0.9 }));
      g.appendChild(svg("text", { x: x, y: y + r + 14, "text-anchor": "middle" }, h.host));
      g.addEventListener("click", function () { openHost(h.host); });
      s.appendChild(g);
    });
    box.appendChild(s);
  }
  function barRows(box, rows, color, onclick) {
    box.textContent = "";
    if (!rows.length) { box.appendChild(empty("Nothing here yet")); return; }
    var max = Math.max.apply(null, rows.map(function (r) { return r.value; })) || 1;
    rows.forEach(function (r, i) {
      var row = el("div", "bar-row"), lab = el("div", "lab");
      if (onclick) { var b = el("button", null, r.label); b.addEventListener("click", function () { onclick(r); }); lab.appendChild(b); } else lab.textContent = r.label;
      var tr = el("div", "track"), f = el("div", "fill"); f.style.background = r.color || color; f.style.setProperty("--w", Math.max(3, 100 * r.value / max) + "%"); f.style.setProperty("--d", i * 70 + "ms"); tr.appendChild(f);
      row.appendChild(lab); row.appendChild(tr); row.appendChild(el("div", "val", r.text != null ? r.text : String(r.value))); box.appendChild(row);
    });
  }
  function hostBars(st) {
    barRows(document.getElementById("ch-hosts"), st.top_hosts.map(function (h) { return { label: h.host, value: h.alerts + h.events * 0.01, text: h.alerts + (h.events ? " · " + h.events + " ev" : ""), color: SEV_COLOR[h.max_severity] || "var(--c-other)", host: h.host }; }), "var(--accent)", function (r) { openHost(r.host); });
  }
  function layerBars(st) {
    barRows(document.getElementById("ch-layers"), Object.keys(st.layers).map(function (k) { return { label: LAYER_NAME[k] || k, value: st.layers[k], text: Math.round(100 * st.layers[k]) + "%", color: LAYER_COLOR[k] || "var(--c-other)" }; }), "var(--accent)");
  }
  function ruleBars(st) {
    barRows(document.getElementById("ch-rules"), st.rules.map(function (r) { return { label: r.rule, value: r.count, color: r.signature ? "var(--critical)" : "var(--medium)" }; }), "var(--medium)");
  }
  function latest(r) { alertRows(document.getElementById("t-latest"), r.rows, new Set()); }

  // ------------------------------------------------------------------------------------------------ alerts
  function layerBar(layers) {
    var box = el("div"), st = el("div", "stack"), lab = el("div", "stack-l"), keys = Object.keys(layers || {});
    keys.forEach(function (k) { var i = el("i"); i.style.width = Math.round(100 * layers[k]) + "%"; i.style.background = LAYER_COLOR[k] || "var(--c-other)"; i.title = (LAYER_NAME[k] || k) + " " + Math.round(100 * layers[k]) + "%"; st.appendChild(i); });
    lab.textContent = keys.map(function (k) { return k + " " + Math.round(100 * layers[k]) + "%"; }).join(" · ");
    box.appendChild(st); box.appendChild(lab); return box;
  }
  var SEEN_A = null, SEEN_E = null;
  function rowKey(r) { return r.host + "|" + r.ts + "|" + (r.reasons ? r.reasons[0] : r.reason); }
  function alertRows(tab, rows, seen) {
    var b = thead(tab, ["Severity", "When (UTC)", "Host", "Confidence", "Evidence", "Why"]);
    if (!rows.length) { var tr = el("tr"), c = td(empty("No alerts", "Run the pipeline and start the dashboard with --data pointing at its output folder.")); c.colSpan = 6; tr.appendChild(c); b.appendChild(tr); return; }
    rows.forEach(function (a) {
      var tr = el("tr"); tr.tabIndex = 0;
      if (seen && seen.size && !seen.has(rowKey(a))) tr.className = "fresh";
      tr.appendChild(td(pill(a.severity))); tr.appendChild(whenCell(a.ts)); tr.appendChild(td(a.host, "host"));
      tr.appendChild(td(confidence(Number(a.probability)))); tr.appendChild(td(layerBar(a.layers)));
      var why = el("div"), rs = a.reasons || [];
      why.appendChild(el("div", "clamp", rs[0] || "")); if (rs.length > 1) why.appendChild(el("span", "more", "+" + (rs.length - 1) + " more"));
      tr.appendChild(td(why, "why"));
      tr.addEventListener("click", function () { openHost(a.host); });
      tr.addEventListener("keydown", function (e) { if (e.key === "Enter") openHost(a.host); });
      b.appendChild(tr);
    });
  }
  var aLimit = 150;
  function alerts(reset) {
    if (reset) { aLimit = 150; skeleton(document.getElementById("atab"), 5); }
    var q = new URLSearchParams({ q: document.getElementById("aq").value, severity: segValue("asev", "data-s"), limit: String(aLimit) });
    return api("/nids/api/alerts?" + q).then(function (r) {
      alertRows(document.getElementById("atab"), r.rows, reset ? null : SEEN_A);
      SEEN_A = new Set(r.rows.map(rowKey));
      document.getElementById("acount").textContent = r.total + " alert" + (r.total === 1 ? "" : "s");
      document.getElementById("amore").hidden = r.rows.length >= r.total;
    });
  }

  // ------------------------------------------------------------------------------------------------ events
  function eventRows(tab, rows, seen) {
    var b = thead(tab, ["When (UTC)", "Host", "Type", "Rule", "Score", "Reason"]);
    if (!rows.length) { var tr = el("tr"), c = td(empty("No events", "Nothing matched the rules, or the filter is too narrow.")); c.colSpan = 6; tr.appendChild(c); b.appendChild(tr); return; }
    rows.forEach(function (e) {
      var d = e.detail || {}, tr = el("tr"); tr.tabIndex = 0;
      if (seen && seen.size && !seen.has(rowKey(e))) tr.className = "fresh";
      tr.appendChild(whenCell(e.ts)); tr.appendChild(td(e.host, "host"));
      tr.appendChild(td(d.signature ? pill("sig", "signature") : e.layer === "flows" ? pill("flow", "flow model") : pill("heur", "heuristic")));
      tr.appendChild(td(String(d.trail || e.layer)));
      var m = el("div", "meter"), t = el("div", "t"), i = el("i"); i.style.width = Math.round(100 * Number(e.score)) + "%"; t.appendChild(i); m.appendChild(t); m.appendChild(el("span", null, Number(e.score).toFixed(2))); tr.appendChild(td(m));
      tr.appendChild(td(el("div", "clamp", e.reason), "why"));
      tr.addEventListener("click", function () { openHost(e.host); });
      tr.addEventListener("keydown", function (ev) { if (ev.key === "Enter") openHost(e.host); });
      b.appendChild(tr);
    });
  }
  var eLimit = 150;
  function events(reset) {
    if (reset) { eLimit = 150; skeleton(document.getElementById("etab"), 5); }
    var q = new URLSearchParams({ q: document.getElementById("eq").value, kind: segValue("ekind", "data-k"), limit: String(eLimit) });
    return api("/nids/api/events?" + q).then(function (r) {
      eventRows(document.getElementById("etab"), r.rows, reset ? null : SEEN_E);
      SEEN_E = new Set(r.rows.map(rowKey));
      document.getElementById("ecount").textContent = r.total + " event" + (r.total === 1 ? "" : "s");
      document.getElementById("emore").hidden = r.rows.length >= r.total;
    });
  }

  // ------------------------------------------------------------------------------------------------ host drawer
  function openHost(host) {
    api("/nids/api/host?host=" + encodeURIComponent(host)).then(function (r) {
      document.getElementById("d-title").textContent = host;
      var worst = "none"; r.alerts.forEach(function (a) { if (SEV.indexOf(a.severity) !== -1 && (worst === "none" || SEV.indexOf(a.severity) < SEV.indexOf(worst))) worst = a.severity; });
      var sv = document.getElementById("d-sev"); sv.textContent = ""; if (worst !== "none") sv.appendChild(pill(worst));
      var body = document.getElementById("d-body"); body.textContent = "";
      var k = el("div", "tiles");
      [[r.alert_total, "alerts"], [r.event_total, "rule events"]].forEach(function (x) { var t = el("div", "tile"); t.appendChild(el("div", "n", String(x[0]))); t.appendChild(el("div", "l", x[1])); k.appendChild(t); });
      body.appendChild(k);
      var items = r.alerts.map(function (a) { return { ts: a.ts, kind: "alert", a: a }; }).concat(r.events.map(function (e) { return { ts: e.ts, kind: "event", e: e }; }));
      items.sort(function (x, y) { return y.ts - x.ts; });
      if (!items.length) body.appendChild(empty("Nothing recorded for this host"));
      var ul = el("ul", "tl");
      items.slice(0, 120).forEach(function (it) {
        var li = el("li"), u = utc(it.ts);
        li.appendChild(el("div", "t", u.date + " " + u.time + " UTC"));
        var x = el("div", "x");
        if (it.kind === "alert") {
          li.style.setProperty("--dc", SEV_COLOR[it.a.severity] || "var(--low)");
          x.appendChild(pill(it.a.severity)); x.appendChild(document.createTextNode(" alert, " + Math.floor(nines(Number(it.a.probability))) + " nines"));
          var rs = el("div", "reasons"); (it.a.reasons || []).forEach(function (t) { rs.appendChild(el("div", null, t)); }); x.appendChild(rs); x.appendChild(layerBar(it.a.layers));
        } else {
          var d = it.e.detail || {}; li.style.setProperty("--dc", d.signature ? "var(--critical)" : "var(--medium)");
          x.appendChild(d.signature ? pill("sig", "signature") : pill("heur", "heuristic")); x.appendChild(document.createTextNode(" " + (d.trail || it.e.layer))); x.appendChild(el("div", null, it.e.reason));
        }
        li.appendChild(x); ul.appendChild(li);
      });
      body.appendChild(ul);
      var dr = document.getElementById("drawer"); dr.classList.add("on"); dr.setAttribute("aria-hidden", "false"); document.getElementById("scrim").classList.add("on");
    });
  }
  function closeAll() {
    var dr = document.getElementById("drawer"); dr.classList.remove("on"); dr.setAttribute("aria-hidden", "true");
    document.getElementById("scrim").classList.remove("on"); closePalette();
  }

  // ------------------------------------------------------------------------------------------------ command palette
  var PITEMS = [], PSEL = 0;
  function actions() {
    var a = [["▦", "Overview", "page", function () { location.hash = "#/overview"; }], ["⚠", "Alerts", "page", function () { location.hash = "#/alerts"; }],
             ["☰", "Rule events", "page", function () { location.hash = "#/events"; }], ["↑", "Upload a flow file", "page", function () { location.hash = "#/upload"; }], ["◉", "Live capture", "page", function () { location.hash = "#/live"; }],
             ["↗", "Flow analysis app", "page", function () { location.href = "/app/"; }], ["◐", "Toggle light / dark theme", "command", toggleTheme],
             ["●", "Turn live refresh on or off", "command", toggleLive], ["⇥", "Sign out", "command", signOut]];
    if (STATS) STATS.top_hosts.forEach(function (h) { a.push(["◎", "Host " + h.host, h.alerts + " alerts", function () { openHost(h.host); }]); });
    return a;
  }
  function renderPalette(filter) {
    var list = document.getElementById("plist"); list.textContent = ""; filter = (filter || "").trim().toLowerCase();
    PITEMS = actions().filter(function (x) { return !filter || x[1].toLowerCase().indexOf(filter) !== -1; });
    if (/^\d{1,3}(\.\d{1,3}){3}$/.test(filter)) PITEMS.unshift(["◎", "Open host " + filter, "host", function () { openHost(filter); }]);
    PSEL = 0;
    PITEMS.slice(0, 12).forEach(function (x, i) {
      var li = el("li", i === PSEL ? "sel" : ""); li.appendChild(el("span", "ico", x[0])); li.appendChild(el("span", null, x[1])); li.appendChild(el("small", null, x[2]));
      li.addEventListener("click", function () { closePalette(); x[3](); }); list.appendChild(li);
    });
    if (!PITEMS.length) list.appendChild(el("li", null, "No match"));
  }
  function movePalette(d) {
    var lis = document.querySelectorAll("#plist li"); if (!lis.length) return;
    lis[PSEL] && lis[PSEL].classList.remove("sel"); PSEL = (PSEL + d + lis.length) % lis.length; lis[PSEL].classList.add("sel"); lis[PSEL].scrollIntoView({ block: "nearest" });
  }
  function openPalette() { var p = document.getElementById("palette"); p.classList.add("on"); p.setAttribute("aria-hidden", "false"); document.getElementById("scrim").classList.add("on"); var i = document.getElementById("pin"); i.value = ""; renderPalette(""); i.focus(); }
  function closePalette() { var p = document.getElementById("palette"); if (p.classList.contains("on")) { p.classList.remove("on"); p.setAttribute("aria-hidden", "true"); if (!document.getElementById("drawer").classList.contains("on")) document.getElementById("scrim").classList.remove("on"); } }
  document.getElementById("pin").addEventListener("input", function (e) { renderPalette(e.target.value); });
  document.getElementById("pin").addEventListener("keydown", function (e) {
    if (e.key === "ArrowDown") { e.preventDefault(); movePalette(1); } else if (e.key === "ArrowUp") { e.preventDefault(); movePalette(-1); }
    else if (e.key === "Enter") { e.preventDefault(); var x = PITEMS[PSEL]; if (x) { closePalette(); x[3](); } }
  });

  // ------------------------------------------------------------------------------------------------ live refresh
  var liveOn = true, liveTimer = null;
  function setLive(on) {
    liveOn = on; var b = document.getElementById("live"); b.classList.toggle("on", on); document.getElementById("live-t").textContent = on ? "Live" : "Paused";
    clearInterval(liveTimer); if (on) liveTimer = setInterval(poll, 8000);
    try { localStorage.setItem("nids-live", on ? "1" : "0"); } catch (e) { /* ignore */ }
  }
  function toggleLive() { setLive(!liveOn); toast(liveOn ? "Live refresh on" : "Live refresh paused"); }
  function poll() {
    if (document.hidden) return;
    api("/nids/api/summary").then(function (s) {
      var n = s.alerts + s.events;
      if (LAST_COUNT >= 0 && n !== LAST_COUNT) {
        var d = n - LAST_COUNT; LAST_COUNT = n; toast("New data: " + (d > 0 ? "+" + d : d) + " items (" + s.alerts + " alerts, " + s.events + " events)");
        var v = currentView(); if (v === "overview") overview(); else if (v === "alerts") alerts(false); else if (v === "events") events(false);
      }
    }).catch(function () { /* keep the old numbers */ });
  }

  // ------------------------------------------------------------------------------------------------ upload
  function tile(box, n, l) { var t = el("div", "tile"); t.appendChild(el("div", "n", String(n))); t.appendChild(el("div", "l", l)); box.appendChild(t); }
  function showResult(r) {
    var box = document.getElementById("result"); box.textContent = "";
    if (r.error) { box.appendChild(el("div", "notice err", "Could not analyse the file: " + r.error)); toast("Upload failed"); return; }
    var pct = function (x) { return (100 * x).toFixed(1) + "%"; }, tiles = el("div", "tiles");
    tile(tiles, r.rows, "rows (" + r.schema + ")");
    if (r.flagged != null) { tile(tiles, r.flagged, "flagged as attack (" + pct(r.flagged_share) + ")"); if (r.severity) tile(tiles, r.severity.high + " / " + r.severity.medium, "high / medium"); if (r.hosts_flagged != null) tile(tiles, r.hosts_flagged, "hosts with flagged flows"); }
    else { tile(tiles, r.findings, "scan findings"); tile(tiles, r.hosts_flagged, "hosts involved"); }
    tile(tiles, r.events_added, "events added"); tile(tiles, r.seconds + " s", "analysis time");
    box.appendChild(tiles);
    (r.warnings || []).forEach(function (w) { box.appendChild(el("div", "notice warn", w)); });
    if (r.caution) box.appendChild(el("div", "notice info", r.caution));
    var d = el("div", "upload-row");
    (r.downloads || []).forEach(function (k) { var a = el("a", "btn ghost", "Download " + k + ".csv"); a.href = "/nids/api/download?id=" + encodeURIComponent(r.id) + "&kind=" + encodeURIComponent(k); d.appendChild(a); });
    box.appendChild(d);
    toast("Analysed " + r.rows + " rows: " + (r.flagged != null ? r.flagged + " flagged" : r.findings + " findings"));
    loaded.overview = false; loaded.events = false; LAST_COUNT = -1; status();
  }
  function upload() {
    var f = document.getElementById("file").files[0], box = document.getElementById("result"), go = document.getElementById("go");
    if (!f) { box.textContent = ""; box.appendChild(el("div", "notice warn", "Choose a file first.")); return; }
    if (f.size > 100 * 1024 * 1024) { box.textContent = ""; box.appendChild(el("div", "notice err", "The file is larger than 100 MB.")); return; }
    go.disabled = true; box.textContent = ""; box.appendChild(el("div", "notice info", "Uploading and analysing " + f.name + " ..."));
    fetch("/nids/api/upload", { method: "POST", credentials: "same-origin", body: f,
      headers: { "X-NIDS-Upload": "1", "X-Filename": encodeURIComponent(f.name), "X-Policy": segValue("policy", "data-p"), "Content-Type": "application/octet-stream" } })
      .then(function (r) { return r.json().catch(function () { return { error: "server answered " + r.status }; }); })
      .then(showResult, function (e) { showResult({ error: String(e) }); })
      .then(function () { go.disabled = false; });
  }

  // ------------------------------------------------------------------------------------------------ live capture
  var capTimer = null;
  function capHeaders() { return { "X-NIDS-Capture": "1", "Content-Type": "application/json" }; }
  function setSrc(v) {
    SRC = v; try { localStorage.setItem("nids-src", v); } catch (e) { /* ignore */ }
    var b = document.getElementById("src"); b.textContent = "Data: " + (v === "live" ? "Live captures" : "Recorded"); b.classList.toggle("src-live", v === "live");
    loaded = {}; LAST_COUNT = -1; status(); route();
  }
  function capRender(st) {
    var job = st.job || {}, nets = st.networks || [], net = document.getElementById("cap-net"), box = document.getElementById("cap-status"), busy = job.state === "capturing" || job.state === "analysing";
    net.textContent = "";
    net.appendChild(el("div", "notice " + (st.needs_acknowledgement ? "warn" : "info"),
      nets.length ? "Network: " + nets.map(function (n) { return n.name + " (" + n.category + ")"; }).join(", ") + (st.needs_acknowledgement ? " - not marked Private: other people's broadcast traffic may be on the wire, so you must confirm below." : " - marked Private.")
                  : "The network type could not be checked, so you must confirm below."));
    document.getElementById("ack-box").hidden = !st.needs_acknowledgement;
    document.getElementById("cap-go").disabled = busy;
    box.textContent = "";
    if (busy) {
      box.appendChild(el("div", "notice info", job.message + " (" + Math.max(0, Math.round(Date.now() / 1000 - job.started)) + " s since start)"));
    } else if (job.state === "done") {
      var tiles = el("div", "tiles"); tile(tiles, job.packets, "packets captured"); tile(tiles, job.events, "rule events"); tile(tiles, job.alerts, "alerts"); tile(tiles, job.signature_hits, "signature hits");
      box.appendChild(tiles); box.appendChild(el("div", "notice info", job.message));
      var row = el("div", "upload-row"), go = el("button", "btn", "Show live results"); go.addEventListener("click", function () { setSrc("live"); location.hash = "#/overview"; }); row.appendChild(go); box.appendChild(row);
    } else if (job.state === "failed") {
      box.appendChild(el("div", "notice err", job.message));
    }
    box.appendChild(el("p", "hint", "Stored live results: " + st.live_alerts + " alerts, " + st.live_events + " events."));
    var hb = document.getElementById("cap-hist"); hb.textContent = "";
    var hist = st.history || [];
    if (hist.length) {
      hb.appendChild(el("h2", null, "Recent captures")); var ul = el("ul", "plain");
      hist.slice().reverse().forEach(function (h) {
        var d = utc(h.finished);
        ul.appendChild(el("li", null, d.date + " " + d.time + " UTC: " + h.packets.toLocaleString() + " packets in " + h.seconds + " s, " + (h.events + h.alerts === 0 ? "nothing suspicious found" : h.events + " events, " + h.alerts + " alerts") + (h.test_request ? " (with the test request)" : "")));
      });
      hb.appendChild(ul);
    }
    clearTimeout(capTimer); if (busy) capTimer = setTimeout(capPoll, 1500);
  }
  function liveNote() {
    var box = document.getElementById("live-note"); box.textContent = ""; box.hidden = SRC !== "live";
    if (SRC !== "live") return;
    Promise.all([api("/nids/api/summary"), api("/nids/api/capture/status")]).then(function (a) {
      var s = a[0], st = a[1], h = st.history || [], last = h[h.length - 1], empty = s.alerts + s.events === 0;
      var t = (empty ? "No findings in your live captures yet. " : "Showing live captures only. ");
      t += last ? "Last capture: " + last.packets.toLocaleString() + " packets in " + last.seconds + " s, " + (last.events + last.alerts === 0 ? "nothing suspicious found, which is normal for ordinary traffic." : last.events + " events.") : "No capture has been run yet.";
      if (empty) t += " On the Live capture page you can tick the test request to see a finding appear.";
      var n = el("div", "notice info", t); box.appendChild(n);
    }).catch(function () { /* keep the page as it is */ });
  }
  function capPoll() { api("/nids/api/capture/status").then(capRender).catch(function () { /* keep the old status */ }); }
  function capPost(path, body) {
    return fetch(path, { method: "POST", credentials: "same-origin", headers: capHeaders(), body: JSON.stringify(body || {}) })
      .then(function (r) { return r.json().catch(function () { return { error: "server answered " + r.status }; }).then(function (j) { j.status = r.status; return j; }); });
  }
  function capStart() {
    var box = document.getElementById("cap-status"), s = parseFloat(document.getElementById("cap-s").value);
    if (!(s >= 5 && s <= 120)) { box.textContent = ""; box.appendChild(el("div", "notice warn", "Seconds must be between 5 and 120.")); return; }
    capPost("/nids/api/capture", { seconds: s, acknowledged: document.getElementById("ack").checked, test_request: document.getElementById("cap-test").checked }).then(function (j) {
      if (j.error) { box.textContent = ""; box.appendChild(el("div", "notice " + (j.needs_acknowledgement ? "warn" : "err"), j.error)); return; }
      toast("Capture started: approve the Windows prompt"); capPoll();
    });
  }
  function capClear() {
    if (!confirm("Delete all stored live-capture alerts and events?")) return;
    capPost("/nids/api/capture/clear").then(function (j) { toast(j.error ? j.error : "Live results cleared"); loaded = {}; LAST_COUNT = -1; capPoll(); status(); });
  }

  // ------------------------------------------------------------------------------------------------ routing and wiring
  var loaded = {};
  function currentView() { var v = location.hash.replace(/^#\//, "") || "overview"; if (v === "search") v = "overview"; return ["overview", "alerts", "events", "upload", "live"].indexOf(v) < 0 ? "overview" : v; }
  function status() { api("/nids/api/summary").then(function (s) { LAST_COUNT = s.alerts + s.events; }); }
  function route() {
    var raw = location.hash.replace(/^#\//, "") || "overview";
    if (raw.indexOf("host/") === 0) { var h = decodeURIComponent(raw.slice(5)); openHost(h); }
    if (raw === "search") openPalette();                                                                              // deep link: #/search
    var v = currentView();
    document.querySelectorAll(".view").forEach(function (s) { s.hidden = s.id !== "v-" + v; });
    document.querySelectorAll("nav.tabs a").forEach(function (a) { a.classList.toggle("on", a.getAttribute("data-v") === v); });
    if (v === "live") capPoll();
    liveNote();
    if (!loaded[v]) { loaded[v] = true; if (v === "overview") overview(true); if (v === "alerts") alerts(true); if (v === "events") events(true); }
    if (raw.indexOf("host/") !== 0) window.scrollTo(0, 0);
  }
  function toggleTheme() {
    var r = document.documentElement, cur = r.getAttribute("data-theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"), nx = cur === "dark" ? "light" : "dark";
    r.setAttribute("data-theme", nx); try { localStorage.setItem("nids-theme", nx); } catch (e) { /* storage may be blocked */ }
  }
  function signOut() { fetch("/logout", { method: "POST", credentials: "same-origin" }).then(function () { location.href = "/login"; }); }
  window.addEventListener("hashchange", route);
  document.getElementById("aq").addEventListener("input", function () { later(function () { alerts(true); }); });
  document.getElementById("eq").addEventListener("input", function () { later(function () { events(true); }); });
  wireSeg("asev", function () { alerts(true); }); wireSeg("ekind", function () { events(true); }); wireSeg("policy");
  document.getElementById("amore").addEventListener("click", function () { aLimit += 300; alerts(false); });
  document.getElementById("emore").addEventListener("click", function () { eLimit += 300; events(false); });
  document.getElementById("go").addEventListener("click", upload);
  document.getElementById("cap-go").addEventListener("click", capStart);
  document.getElementById("cap-clear").addEventListener("click", capClear);
  document.getElementById("src").addEventListener("click", function () { setSrc(SRC === "live" ? "recorded" : "live"); toast(SRC === "live" ? "Showing live captures" : "Showing recorded data"); });
  (function () { var b = document.getElementById("src"); b.textContent = "Data: " + (SRC === "live" ? "Live captures" : "Recorded"); b.classList.toggle("src-live", SRC === "live"); })();
  var drop = document.getElementById("drop"), file = document.getElementById("file");
  function picked() { var f = file.files[0]; document.getElementById("drop-t").textContent = f ? f.name : "Drop a file here or click to choose"; document.getElementById("drop-s").textContent = f ? (f.size / 1048576).toFixed(2) + " MB" : ".csv, .tsv, .binetflow"; }
  file.addEventListener("change", picked);
  ["dragenter", "dragover"].forEach(function (n) { drop.addEventListener(n, function (e) { e.preventDefault(); drop.classList.add("over"); }); });
  ["dragleave", "drop"].forEach(function (n) { drop.addEventListener(n, function (e) { e.preventDefault(); drop.classList.remove("over"); }); });
  drop.addEventListener("drop", function (e) { if (e.dataTransfer.files.length) { file.files = e.dataTransfer.files; picked(); } });
  document.getElementById("d-close").addEventListener("click", closeAll);
  document.getElementById("scrim").addEventListener("click", closeAll);
  document.getElementById("kbtn").addEventListener("click", openPalette);
  document.getElementById("live").addEventListener("click", toggleLive);
  document.getElementById("theme").addEventListener("click", toggleTheme);
  document.getElementById("out").addEventListener("click", signOut);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") closeAll();
    else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); openPalette(); }
    else if (e.key === "/" && !/INPUT|SELECT|TEXTAREA/.test((document.activeElement || {}).tagName || "")) {
      var id = currentView() === "events" ? "eq" : "aq"; if (currentView() === "alerts" || currentView() === "events") { e.preventDefault(); document.getElementById(id).focus(); }
    }
  });
  try { var th = localStorage.getItem("nids-theme"); if (th) document.documentElement.setAttribute("data-theme", th); } catch (e) { /* ignore */ }
  var liveSaved = "1"; try { liveSaved = localStorage.getItem("nids-live") || "1"; } catch (e) { /* ignore */ }
  setLive(liveSaved === "1");
  route();
})();
