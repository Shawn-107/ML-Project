/* NIDS: "Upload CSV" button and dialog. Sends a flow file to /nids/csv/upload (signed-in users only) and shows the summary. */
(function () {
  "use strict";
  function el(tag, attrs, text) {
    var e = document.createElement(tag);
    for (var k in (attrs || {})) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }
  function row(label, value) {
    var p = el("div", { style: "display:flex;justify-content:space-between;gap:12px;padding:3px 0;border-bottom:1px solid var(--line-solid)" });
    p.appendChild(el("span", { style: "color:var(--muted)" }, label));
    p.appendChild(el("span", null, value));
    return p;
  }
  function show(box, r) {
    box.textContent = "";
    if (r.error) { box.appendChild(el("div", { style: "color:#e5534b" }, "Error: " + r.error)); return; }
    var pct = function (x) { return (100 * x).toFixed(1) + "%"; };
    box.appendChild(row("format", r.schema));
    box.appendChild(row("rows", String(r.rows)));
    if (r.flagged != null) {
      box.appendChild(row("flagged as attack", r.flagged + " (" + pct(r.flagged_share) + ")"));
      if (r.severity) box.appendChild(row("high / medium", r.severity.high + " / " + r.severity.medium));
      box.appendChild(row("false-alarm target", r.policy));
    } else {
      box.appendChild(row("scan findings", String(r.findings)));
      box.appendChild(row("hosts involved", String(r.hosts_flagged)));
    }
    if (r.hosts_flagged != null && r.flagged != null) box.appendChild(row("hosts with flagged flows", String(r.hosts_flagged)));
    box.appendChild(row("events added to the table", String(r.events_added)));
    box.appendChild(row("time", r.seconds + " s"));
    (r.warnings || []).forEach(function (w) { box.appendChild(el("div", { style: "color:#d4a72c;margin-top:6px" }, "Warning: " + w)); });
    if (r.caution) box.appendChild(el("div", { style: "color:var(--muted);margin-top:6px;font-size:12px" }, r.caution));
    var d = el("div", { style: "margin-top:10px;display:flex;gap:10px;flex-wrap:wrap" });
    (r.downloads || []).forEach(function (k) {
      var a = el("a", { href: "/nids/csv/download?id=" + encodeURIComponent(r.id) + "&kind=" + encodeURIComponent(k), class: "foot-link" }, "download " + k + ".csv");
      d.appendChild(a);
    });
    box.appendChild(d);
    if (r.events_added) {
      var b = el("button", { class: "ghost-btn", style: "margin-top:10px" }, "Reload table");
      b.onclick = function () { location.reload(); };
      box.appendChild(b);
    }
  }
  function open() {
    if (document.getElementById("nids_upload_overlay")) return;
    var o = el("div", { id: "nids_upload_overlay", class: "modal-overlay" });
    var m = el("div", { class: "modal", style: "width:460px" });
    m.appendChild(el("div", { class: "modal-h" }, "Upload a flow CSV"));
    m.appendChild(el("div", { style: "color:var(--muted);font-size:12px;margin-bottom:10px" },
      "Accepted: UNSW-NB15 (34 features), NetFlow v2 (40 features), Argus flow files. The format is detected from the columns. Up to 100 MB."));
    var f = el("input", { type: "file", accept: ".csv,.tsv,.txt,.binetflow" });
    var sel = el("select", { style: "margin:10px 0;display:block" });
    [["fpr_3pct", "3% false-alarm target (default)"], ["fpr_1pct", "1% false-alarm target (stricter)"]].forEach(function (p) { sel.appendChild(el("option", { value: p[0] }, p[1])); });
    var out = el("div", { style: "margin-top:10px;font-size:13px;max-height:48vh;overflow:auto" });
    var go = el("button", { class: "ghost-btn" }, "Analyse");
    var close = el("button", { class: "ghost-btn", style: "margin-left:8px" }, "Close");
    close.onclick = function () { o.remove(); };
    o.addEventListener("mousedown", function (e) { if (e.target === o) o.remove(); });
    go.onclick = function () {
      var file = f.files && f.files[0];
      if (!file) { out.textContent = "Choose a file first."; return; }
      if (file.size > 100 * 1024 * 1024) { out.textContent = "File is larger than 100 MB."; return; }
      go.disabled = true; out.textContent = "Uploading and analysing " + file.name + " …";
      fetch("/nids/csv/upload", {
        method: "POST", credentials: "same-origin", body: file,
        headers: { "X-NIDS-Upload": "1", "X-Filename": encodeURIComponent(file.name), "X-Policy": sel.value, "Content-Type": "application/octet-stream" }
      }).then(function (r) { return r.json().catch(function () { return { error: "server answered " + r.status }; }); })
        .then(function (j) { show(out, j); }, function (e) { show(out, { error: String(e) }); })
        .then(function () { go.disabled = false; });
    };
    [f, sel, go, close, out].forEach(function (x) { m.appendChild(x); });
    o.appendChild(m); document.body.appendChild(o);
  }
  function init() {
    var login = document.getElementById("login_link");
    var b = el("button", { class: "ghost-btn", id: "nids_upload_btn", title: "analyse a flow CSV file" }, "Upload CSV");
    b.onclick = open;
    if (login && login.parentNode) login.parentNode.insertBefore(b, login); else document.querySelector("header").appendChild(b);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
