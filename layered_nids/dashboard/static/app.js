(function () {
  "use strict";
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  function api(path) {
    return fetch(path, { credentials: "same-origin" }).then(function (r) {
      if (r.status === 401) { location.href = "/login"; throw new Error("signed out"); }
      return r.json();
    });
  }
  function when(ts) { return new Date(ts * 1000).toISOString().replace("T", " ").slice(0, 19); }
  function td(content, cls) { var c = el("td", cls); if (content instanceof Node) c.appendChild(content); else c.textContent = content; return c; }
  function head(tab, cols) {
    tab.textContent = "";
    var tr = el("tr");
    cols.forEach(function (c) { tr.appendChild(el("th", null, c)); });
    var th = el("thead");
    th.appendChild(tr);
    tab.appendChild(th);
    tab.appendChild(el("tbody"));
  }
  function empty(tab, msg) {
    var tr = el("tr"), c = td(msg, "empty");
    c.colSpan = 8;
    tr.appendChild(c);
    tab.querySelector("tbody").appendChild(tr);
  }
  var timer = null;
  function later(fn) { clearTimeout(timer); timer = setTimeout(fn, 250); }

  function cards() {
    api("/nids/api/summary").then(function (s) {
      var box = document.getElementById("cards");
      box.textContent = "";
      [["Host alerts", s.alerts], ["Hosts alerted", s.alert_hosts], ["Rule events", s.events], ["Signature hits", s.signature_events]].forEach(function (x) {
        var c = el("div", "card");
        c.appendChild(el("div", "n", String(x[1])));
        c.appendChild(el("div", "l", x[0]));
        box.appendChild(c);
      });
    });
  }
  function alerts() {
    var q = new URLSearchParams({ q: document.getElementById("aq").value, severity: document.getElementById("asev").value });
    api("/nids/api/alerts?" + q).then(function (r) {
      var t = document.getElementById("atab");
      head(t, ["Time (UTC)", "Host", "Severity", "Probability", "Layers", "Reason"]);
      document.getElementById("acount").textContent = r.total + " alerts" + (r.total > r.rows.length ? " (showing " + r.rows.length + ")" : "");
      if (!r.rows.length) return empty(t, "No alerts.");
      var tb = t.querySelector("tbody");
      r.rows.forEach(function (a) {
        var tr = el("tr");
        tr.appendChild(td(when(a.ts)));
        tr.appendChild(td(a.host));
        tr.appendChild(td(el("span", "badge " + a.severity, a.severity)));
        tr.appendChild(td(Number(a.probability).toFixed(6)));
        var ch = el("span");
        Object.keys(a.layers || {}).forEach(function (k) { ch.appendChild(el("span", "chip", k + " " + Math.round(a.layers[k] * 100) + "%")); });
        tr.appendChild(td(ch));
        tr.appendChild(td((a.reasons || []).join(" | "), "reason"));
        tb.appendChild(tr);
      });
    });
  }
  function events() {
    var q = new URLSearchParams({ q: document.getElementById("eq").value, kind: document.getElementById("ekind").value });
    api("/nids/api/events?" + q).then(function (r) {
      var t = document.getElementById("etab");
      head(t, ["Time (UTC)", "Host", "Type", "Rule", "Score", "Reason"]);
      document.getElementById("ecount").textContent = r.total + " events" + (r.total > r.rows.length ? " (showing " + r.rows.length + ")" : "");
      if (!r.rows.length) return empty(t, "No events.");
      var tb = t.querySelector("tbody");
      r.rows.forEach(function (e) {
        var d = e.detail || {}, tr = el("tr");
        tr.appendChild(td(when(e.ts)));
        tr.appendChild(td(e.host));
        tr.appendChild(td(el("span", "badge " + (d.signature ? "sig" : "heur"), d.signature ? "signature" : "heuristic")));
        tr.appendChild(td(String(d.trail || e.layer)));
        tr.appendChild(td(Number(e.score).toFixed(2)));
        tr.appendChild(td(e.reason, "reason"));
        tb.appendChild(tr);
      });
    });
  }
  function row(box, k, v) {
    var r = el("div", "row");
    r.appendChild(el("span", null, k));
    r.appendChild(el("span", null, String(v)));
    box.appendChild(r);
  }
  function show(box, r) {
    box.textContent = "";
    if (r.error) { box.appendChild(el("div", "err", "Error: " + r.error)); return; }
    var pct = function (x) { return (100 * x).toFixed(1) + "%"; };
    row(box, "format", r.schema);
    row(box, "rows", r.rows);
    if (r.flagged != null) {
      row(box, "flagged as attack", r.flagged + " (" + pct(r.flagged_share) + ")");
      if (r.severity) row(box, "high / medium", r.severity.high + " / " + r.severity.medium);
      row(box, "false-alarm target", r.policy);
    } else {
      row(box, "scan findings", r.findings);
      row(box, "hosts involved", r.hosts_flagged);
    }
    row(box, "events added", r.events_added);
    row(box, "time", r.seconds + " s");
    (r.warnings || []).forEach(function (w) { box.appendChild(el("div", "warn", "Warning: " + w)); });
    if (r.caution) box.appendChild(el("div", "note", r.caution));
    var d = el("div", "note");
    (r.downloads || []).forEach(function (k) {
      var a = el("a", null, "download " + k + ".csv");
      a.href = "/nids/api/download?id=" + encodeURIComponent(r.id) + "&kind=" + encodeURIComponent(k);
      a.style.marginRight = "12px";
      d.appendChild(a);
    });
    box.appendChild(d);
    cards();
    events();
  }
  function upload() {
    var f = document.getElementById("file").files[0], box = document.getElementById("result"), go = document.getElementById("go");
    if (!f) { box.textContent = "Choose a file first."; return; }
    if (f.size > 100 * 1024 * 1024) { box.textContent = "The file is larger than 100 MB."; return; }
    go.disabled = true;
    box.textContent = "Uploading and analysing " + f.name + " ...";
    fetch("/nids/api/upload", {
      method: "POST", credentials: "same-origin", body: f,
      headers: { "X-NIDS-Upload": "1", "X-Filename": encodeURIComponent(f.name), "X-Policy": document.getElementById("policy").value, "Content-Type": "application/octet-stream" }
    })
      .then(function (r) { return r.json().catch(function () { return { error: "server answered " + r.status }; }); })
      .then(function (j) { show(box, j); }, function (e) { show(box, { error: String(e) }); })
      .then(function () { go.disabled = false; });
  }
  document.querySelectorAll(".tabs button").forEach(function (b) {
    b.addEventListener("click", function () {
      document.querySelectorAll(".tabs button").forEach(function (x) { x.classList.toggle("on", x === b); });
      ["alerts", "events", "upload"].forEach(function (t) { document.getElementById("t-" + t).hidden = (t !== b.dataset.t); });
    });
  });
  ["aq", "asev"].forEach(function (i) { document.getElementById(i).addEventListener("input", function () { later(alerts); }); });
  ["eq", "ekind"].forEach(function (i) { document.getElementById(i).addEventListener("input", function () { later(events); }); });
  document.getElementById("go").addEventListener("click", upload);
  document.getElementById("out").addEventListener("click", function () {
    fetch("/logout", { method: "POST", credentials: "same-origin" }).then(function () { location.href = "/login"; });
  });
  document.getElementById("theme").addEventListener("click", function () {
    var r = document.documentElement;
    var cur = r.getAttribute("data-theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    var nx = cur === "dark" ? "light" : "dark";
    r.setAttribute("data-theme", nx);
    try { localStorage.setItem("nids-theme", nx); } catch (e) { /* storage may be blocked */ }
  });
  try { var th = localStorage.getItem("nids-theme"); if (th) document.documentElement.setAttribute("data-theme", th); } catch (e) { /* ignore */ }
  cards();
  alerts();
  events();
})();
