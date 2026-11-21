(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var SEV = ["critical", "high", "medium", "low"];
  var SEV_COLOR = { critical: "var(--critical)", high: "var(--high)", medium: "var(--medium)", low: "var(--low)" };
  var LAYER_COLOR = { packets: "var(--c-packets)", rules: "var(--c-rules)", flows: "var(--c-flows)" };
  var LAYER_NAME = { packets: "Packet anomaly model", rules: "Signatures and rules", flows: "Flow model" };

  // ------------------------------------------------------------------------------------------------ helpers
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  function svg(tag, attrs, text) {
    var e = document.createElementNS(NS, tag);
    for (var k in (attrs || {})) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }
  function api(path) {
    return fetch(path, { credentials: "same-origin" }).then(function (r) {
      if (r.status === 401) { location.href = "/login"; throw new Error("signed out"); }
      return r.json();
    });
  }
  function two(n) { return n < 10 ? "0" + n : String(n); }
  function utc(ts) { var d = new Date(ts * 1000); return { date: d.getUTCFullYear() + "-" + two(d.getUTCMonth() + 1) + "-" + two(d.getUTCDate()), time: two(d.getUTCHours()) + ":" + two(d.getUTCMinutes()) + ":" + two(d.getUTCSeconds()) }; }
  function clock(ts) { var u = utc(ts); return u.time.slice(0, 5); }
  function nines(p) { return p >= 1 ? 12 : Math.min(12, Math.max(0, -Math.log10(Math.max(1 - p, 1e-12)))); }
  function confidence(p) {
    var n = nines(p), box = el("div", "meter");
    var t = el("div", "t"), i = el("i"); i.style.width = Math.round(100 * Math.min(1, n / 12)) + "%"; t.appendChild(i);
    box.appendChild(t);
    box.appendChild(el("span", null, p >= 0.999 ? Math.floor(n) + " nines" : (100 * p).toFixed(1) + "%"));
    box.title = "probability " + p;
    return box;
  }
  function pill(kind, text) { return el("span", "pill " + kind, text || kind); }
  function empty(title, hint) { var d = el("div", "empty"); d.appendChild(el("b", null, title)); d.appendChild(el("span", null, hint || "")); return d; }
  function thead(tab, cols) { tab.textContent = ""; var tr = el("tr"); cols.forEach(function (c) { tr.appendChild(el("th", null, c)); }); var h = el("thead"); h.appendChild(tr); tab.appendChild(h); var b = el("tbody"); tab.appendChild(b); return b; }
  function td(content, cls) { var c = el("td", cls); if (content instanceof Node) c.appendChild(content); else c.textContent = content; return c; }
  function whenCell(ts) { var u = utc(ts), c = el("td", "when"); c.appendChild(el("b", null, u.time)); c.appendChild(document.createTextNode(u.date)); return c; }
  function hostCell(h) { return td(h, "host"); }
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

  // ------------------------------------------------------------------------------------------------ overview
  var STATS = null;
  function overview() {
    Promise.all([api("/nids/api/summary"), api("/nids/api/stats"), api("/nids/api/alerts?limit=8")]).then(function (r) {
      var s = r[0]; STATS = r[1];
      document.getElementById("status-t").textContent = s.alerts + " alerts · " + s.events + " events";
      var rng = document.getElementById("range");
      rng.textContent = STATS.t1 > 0 ? "Data from " + utc(STATS.t0).date + " " + clock(STATS.t0) + " to " + utc(STATS.t1).date + " " + clock(STATS.t1) + " UTC." : "No data loaded yet.";
      kpis(s); timeChart(STATS); donut(STATS); hostBars(STATS); layerBars(STATS); ruleBars(STATS); latest(r[2]);
    });
  }
  function kpis(s) {
    var box = document.getElementById("kpis"); box.textContent = "";
    var items = [["Host alerts", s.alerts, "in " + s.alert_hosts + " host" + (s.alert_hosts === 1 ? "" : "s"), "var(--critical)"],
                 ["Critical", (s.severity && s.severity.critical) || 0, "highest severity", "var(--high)"],
                 ["Rule events", s.events, "on " + s.event_hosts + " host" + (s.event_hosts === 1 ? "" : "s"), "var(--c-rules)"],
                 ["Signature hits", s.signature_events, "known-bad patterns", "var(--c-packets)"]];
    items.forEach(function (x) {
      var c = el("div", "kpi"); c.style.setProperty("--kc", x[3]);
      c.appendChild(el("div", "n", String(x[1]))); c.appendChild(el("div", "l", x[0])); c.appendChild(el("div", "s", x[2])); box.appendChild(c);
    });
  }
  function timeChart(st) {
    var box = document.getElementById("ch-time"); box.textContent = "";
    var hist = st.hist, W = 760, H = 230, L = 34, B = 26, T = 8, R = 6;
    var max = 1; hist.forEach(function (b) { max = Math.max(max, b.critical + b.high + b.medium + b.low, b.events); });
    var ymax = Math.ceil(max / 4) * 4 || 4;
    var s = svg("svg", { viewBox: "0 0 " + W + " " + H, class: "chart", role: "img", "aria-label": "alerts over time" });
    for (var g = 0; g <= 4; g++) {
      var y = T + (H - T - B) * (1 - g / 4);
      s.appendChild(svg("line", { x1: L, x2: W - R, y1: y, y2: y, stroke: "var(--line)", "stroke-width": 1 }));
      s.appendChild(svg("text", { x: L - 6, y: y + 4, "text-anchor": "end" }, String(Math.round(ymax * g / 4))));
    }
    var bw = (W - L - R) / hist.length;
    hist.forEach(function (b, i) {
      var x = L + i * bw, base = H - B, tot = b.critical + b.high + b.medium + b.low;
      var tip = clock(b.t0) + "–" + clock(b.t1) + " UTC: " + tot + " alert" + (tot === 1 ? "" : "s") + ", " + b.events + " rule event" + (b.events === 1 ? "" : "s");
      var grp = svg("g"); grp.appendChild(svg("title", null, tip));
      grp.appendChild(svg("rect", { x: x, y: T, width: bw, height: H - T - B, fill: "transparent" }));
      SEV.slice().reverse().forEach(function (k) {
        var h = (H - T - B) * (b[k] / ymax); if (h <= 0) return; base -= h;
        grp.appendChild(svg("rect", { x: x + 1, y: base, width: Math.max(bw - 2, 1), height: h, fill: SEV_COLOR[k], rx: 2 }));
      });
      s.appendChild(grp);
    });
    // rule events as a line
    var pts = hist.map(function (b, i) { return (L + i * bw + bw / 2) + "," + (T + (H - T - B) * (1 - b.events / ymax)); }).join(" ");
    s.appendChild(svg("polyline", { points: pts, fill: "none", stroke: "var(--c-rules)", "stroke-width": 2, "stroke-linejoin": "round", opacity: 0.9 }));
    [0, 0.25, 0.5, 0.75, 1].forEach(function (f) {
      var t = st.t0 + (st.t1 - st.t0) * f;
      s.appendChild(svg("text", { x: L + (W - L - R) * f, y: H - 8, "text-anchor": f === 0 ? "start" : f === 1 ? "end" : "middle" }, clock(t)));
    });
    box.appendChild(s);
    var lg = document.getElementById("lg-time"); lg.textContent = "";
    SEV.forEach(function (k) { var i = el("span"); var sw = el("i"); sw.style.background = SEV_COLOR[k]; i.appendChild(sw); i.appendChild(document.createTextNode(k)); lg.appendChild(i); });
    var ev = el("span"); var line = el("i"); line.style.background = "var(--c-rules)"; ev.appendChild(line); ev.appendChild(document.createTextNode("rule events (line)")); lg.appendChild(ev);
  }
  function donut(st) {
    var box = document.getElementById("ch-sev"); box.textContent = "";
    var total = 0; SEV.forEach(function (k) { total += st.severity[k] || 0; });
    var s = svg("svg", { viewBox: "0 0 42 42", role: "img", "aria-label": "alerts by severity" });
    s.appendChild(svg("circle", { cx: 21, cy: 21, r: 15.9155, fill: "none", stroke: "var(--surface-2)", "stroke-width": 6 }));
    var off = 25;
    if (total) SEV.forEach(function (k) {
      var v = st.severity[k] || 0; if (!v) return; var share = 100 * v / total;
      s.appendChild(svg("circle", { cx: 21, cy: 21, r: 15.9155, fill: "none", stroke: SEV_COLOR[k], "stroke-width": 6, "stroke-dasharray": share + " " + (100 - share), "stroke-dashoffset": off }));
      off -= share;
    });
    s.appendChild(svg("text", { x: 21, y: 22.5, "text-anchor": "middle", "font-size": 8, "font-weight": 700, fill: "var(--text)", style: "fill:var(--text);font-size:8px;font-weight:700" }, String(total)));
    s.appendChild(svg("text", { x: 21, y: 28, "text-anchor": "middle", style: "font-size:3.4px" }, "alerts"));
    box.appendChild(s);
    var lg = el("div", "legend");
    SEV.forEach(function (k) { var i = el("span"); var sw = el("i"); sw.style.background = SEV_COLOR[k]; i.appendChild(sw); i.appendChild(document.createTextNode(k + "  " + (st.severity[k] || 0))); lg.appendChild(i); });
    box.appendChild(lg);
  }
  function barRows(box, rows, color, onclick) {
    box.textContent = "";
    if (!rows.length) { box.appendChild(empty("Nothing here yet")); return; }
    var max = Math.max.apply(null, rows.map(function (r) { return r.value; })) || 1;
    rows.forEach(function (r) {
      var row = el("div", "bar-row"), lab = el("div", "lab");
      if (onclick) { var b = el("button", null, r.label); b.addEventListener("click", function () { onclick(r); }); lab.appendChild(b); } else lab.textContent = r.label;
      var tr = el("div", "track"), f = el("div", "fill"); f.style.width = Math.max(3, 100 * r.value / max) + "%"; f.style.background = r.color || color; tr.appendChild(f);
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
  function latest(r) { alertRows(document.getElementById("t-latest"), r.rows, true); }

  // ------------------------------------------------------------------------------------------------ alerts
  function layerBar(layers) {
    var box = el("div"), st = el("div", "stack"), lab = el("div", "stack-l"), keys = Object.keys(layers || {});
    keys.forEach(function (k) { var i = el("i"); i.style.width = Math.round(100 * layers[k]) + "%"; i.style.background = LAYER_COLOR[k] || "var(--c-other)"; i.title = (LAYER_NAME[k] || k) + " " + Math.round(100 * layers[k]) + "%"; st.appendChild(i); });
    lab.textContent = keys.map(function (k) { return k + " " + Math.round(100 * layers[k]) + "%"; }).join(" · ");
    box.appendChild(st); box.appendChild(lab); return box;
  }
  function alertRows(tab, rows, compact) {
    var b = thead(tab, ["Severity", "When (UTC)", "Host", "Confidence", "Evidence", "Why"]);
    if (!rows.length) { var tr = el("tr"), c = td(empty("No alerts", "Run the pipeline and start the dashboard with --data pointing at its output folder.")); c.colSpan = 6; tr.appendChild(c); b.appendChild(tr); return; }
    rows.forEach(function (a) {
      var tr = el("tr"); tr.tabIndex = 0;
      tr.appendChild(td(pill(a.severity)));
      tr.appendChild(whenCell(a.ts)); tr.appendChild(hostCell(a.host));
      tr.appendChild(td(confidence(Number(a.probability))));
      tr.appendChild(td(layerBar(a.layers)));
      var why = el("div"), rs = a.reasons || [];
      why.appendChild(el("div", "clamp", rs[0] || ""));
      if (rs.length > 1) why.appendChild(el("span", "more", "+" + (rs.length - 1) + " more"));
      tr.appendChild(td(why, "why"));
      tr.addEventListener("click", function () { openHost(a.host); });
      tr.addEventListener("keydown", function (e) { if (e.key === "Enter") openHost(a.host); });
      b.appendChild(tr);
    });
  }
  var aLimit = 150;
  function alerts(reset) {
    if (reset) aLimit = 150;
    var q = new URLSearchParams({ q: document.getElementById("aq").value, severity: segValue("asev", "data-s"), limit: String(aLimit) });
    api("/nids/api/alerts?" + q).then(function (r) {
      alertRows(document.getElementById("atab"), r.rows);
      document.getElementById("acount").textContent = r.total + " alert" + (r.total === 1 ? "" : "s");
      var m = document.getElementById("amore"); m.hidden = r.rows.length >= r.total;
    });
  }

  // ------------------------------------------------------------------------------------------------ events
  function eventRows(tab, rows) {
    var b = thead(tab, ["When (UTC)", "Host", "Type", "Rule", "Score", "Reason"]);
    if (!rows.length) { var tr = el("tr"), c = td(empty("No events", "Nothing matched the rules, or the filter is too narrow.")); c.colSpan = 6; tr.appendChild(c); b.appendChild(tr); return; }
    rows.forEach(function (e) {
      var d = e.detail || {}, tr = el("tr"); tr.tabIndex = 0;
      tr.appendChild(whenCell(e.ts)); tr.appendChild(hostCell(e.host));
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
    if (reset) eLimit = 150;
    var q = new URLSearchParams({ q: document.getElementById("eq").value, kind: segValue("ekind", "data-k"), limit: String(eLimit) });
    api("/nids/api/events?" + q).then(function (r) {
      eventRows(document.getElementById("etab"), r.rows);
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
      if (!items.length) { body.appendChild(empty("Nothing recorded for this host")); }
      var ul = el("ul", "tl");
      items.slice(0, 120).forEach(function (it) {
        var li = el("li"), u = utc(it.ts);
        li.appendChild(el("div", "t", u.date + " " + u.time + " UTC"));
        var x = el("div", "x");
        if (it.kind === "alert") {
          li.style.setProperty("--dc", SEV_COLOR[it.a.severity] || "var(--low)");
          x.appendChild(pill(it.a.severity)); x.appendChild(document.createTextNode(" alert, " + Math.floor(nines(Number(it.a.probability))) + " nines"));
          var rs = el("div", "reasons"); (it.a.reasons || []).forEach(function (t) { rs.appendChild(el("div", null, t)); }); x.appendChild(rs);
          x.appendChild(layerBar(it.a.layers));
        } else {
          var d = it.e.detail || {}; li.style.setProperty("--dc", d.signature ? "var(--critical)" : "var(--medium)");
          x.appendChild(d.signature ? pill("sig", "signature") : pill("heur", "heuristic")); x.appendChild(document.createTextNode(" " + (d.trail || it.e.layer)));
          x.appendChild(el("div", null, it.e.reason));
        }
        li.appendChild(x); ul.appendChild(li);
      });
      body.appendChild(ul);
      document.getElementById("drawer").classList.add("on"); document.getElementById("drawer").setAttribute("aria-hidden", "false"); document.getElementById("scrim").classList.add("on");
    });
  }
  function closeHost() { document.getElementById("drawer").classList.remove("on"); document.getElementById("drawer").setAttribute("aria-hidden", "true"); document.getElementById("scrim").classList.remove("on"); }

  // ------------------------------------------------------------------------------------------------ upload
  function tile(box, n, l) { var t = el("div", "tile"); t.appendChild(el("div", "n", String(n))); t.appendChild(el("div", "l", l)); box.appendChild(t); }
  function showResult(r) {
    var box = document.getElementById("result"); box.textContent = "";
    if (r.error) { box.appendChild(el("div", "notice err", "Could not analyse the file: " + r.error)); return; }
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
    overview(); events(true);
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

  // ------------------------------------------------------------------------------------------------ routing and wiring
  var loaded = {};
  function status() {
    api("/nids/api/summary").then(function (s) { document.getElementById("status-t").textContent = s.alerts + " alerts · " + s.events + " events"; });
  }
  function route() {
    var v = (location.hash.replace(/^#\//, "") || "overview");
    if (v.indexOf("host/") === 0) { var h = decodeURIComponent(v.slice(5)); v = "overview"; openHost(h); }          // deep link: #/host/<address>
    if (["overview", "alerts", "events", "upload"].indexOf(v) < 0) v = "overview";
    document.querySelectorAll(".view").forEach(function (s) { s.hidden = s.id !== "v-" + v; });
    document.querySelectorAll("nav.tabs a").forEach(function (a) { a.classList.toggle("on", a.getAttribute("data-v") === v); });
    if (!loaded[v]) { loaded[v] = true; if (v === "overview") overview(); if (v === "alerts") alerts(true); if (v === "events") events(true); }
    window.scrollTo(0, 0);
  }
  window.addEventListener("hashchange", route);
  document.getElementById("aq").addEventListener("input", function () { later(function () { alerts(true); }); });
  document.getElementById("eq").addEventListener("input", function () { later(function () { events(true); }); });
  wireSeg("asev", function () { alerts(true); }); wireSeg("ekind", function () { events(true); }); wireSeg("policy");
  document.getElementById("amore").addEventListener("click", function () { aLimit += 300; alerts(false); });
  document.getElementById("emore").addEventListener("click", function () { eLimit += 300; events(false); });
  document.getElementById("go").addEventListener("click", upload);
  var drop = document.getElementById("drop"), file = document.getElementById("file");
  function picked() { var f = file.files[0]; document.getElementById("drop-t").textContent = f ? f.name : "Drop a file here or click to choose"; document.getElementById("drop-s").textContent = f ? (f.size / 1048576).toFixed(2) + " MB" : ".csv, .tsv, .binetflow"; }
  file.addEventListener("change", picked);
  ["dragenter", "dragover"].forEach(function (n) { drop.addEventListener(n, function (e) { e.preventDefault(); drop.classList.add("over"); }); });
  ["dragleave", "drop"].forEach(function (n) { drop.addEventListener(n, function (e) { e.preventDefault(); drop.classList.remove("over"); }); });
  drop.addEventListener("drop", function (e) { if (e.dataTransfer.files.length) { file.files = e.dataTransfer.files; picked(); } });
  document.getElementById("d-close").addEventListener("click", closeHost);
  document.getElementById("scrim").addEventListener("click", closeHost);
  document.addEventListener("keydown", function (e) { if (e.key === "Escape") closeHost(); });
  document.getElementById("out").addEventListener("click", function () { fetch("/logout", { method: "POST", credentials: "same-origin" }).then(function () { location.href = "/login"; }); });
  document.getElementById("theme").addEventListener("click", function () {
    var r = document.documentElement, cur = r.getAttribute("data-theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"), nx = cur === "dark" ? "light" : "dark";
    r.setAttribute("data-theme", nx); try { localStorage.setItem("nids-theme", nx); } catch (e) { /* storage may be blocked */ }
  });
  try { var th = localStorage.getItem("nids-theme"); if (th) document.documentElement.setAttribute("data-theme", th); } catch (e) { /* ignore */ }
  status();
  route();
})();
