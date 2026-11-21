(function () {
  "use strict";
  var REDUCED = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
  try { var th = localStorage.getItem("nids-theme"); if (th) document.documentElement.setAttribute("data-theme", th); } catch (e) { /* storage may be blocked */ }

  // ------------------------------------------------------------------------------------------------ sign-in form
  var f = document.getElementById("f"), u = document.getElementById("u"), p = document.getElementById("p"), err = document.getElementById("e");
  var go = document.getElementById("go"), goT = document.getElementById("go-t");
  document.getElementById("eye").addEventListener("click", function () {
    var show = p.type === "password"; p.type = show ? "text" : "password"; this.setAttribute("aria-label", show ? "hide password" : "show password"); this.classList.toggle("on", show);
  });
  p.addEventListener("keydown", function (e) { document.getElementById("caps").hidden = !(e.getModifierState && e.getModifierState("CapsLock")); });
  p.addEventListener("keyup", function (e) { document.getElementById("caps").hidden = !(e.getModifierState && e.getModifierState("CapsLock")); });
  document.getElementById("theme").addEventListener("click", function () {
    var r = document.documentElement, cur = r.getAttribute("data-theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"), nx = cur === "dark" ? "light" : "dark";
    r.setAttribute("data-theme", nx); try { localStorage.setItem("nids-theme", nx); } catch (e) { /* ignore */ } draw(true);
  });
  function fail(msg) {
    err.textContent = msg; f.classList.remove("shake"); void f.offsetWidth; f.classList.add("shake"); go.disabled = false; go.classList.remove("busy"); goT.textContent = "Sign in"; p.select();
  }
  f.addEventListener("submit", function (ev) {
    ev.preventDefault(); err.textContent = "";
    if (!u.value || !p.value) { fail("Enter your user name and password."); return; }
    go.disabled = true; go.classList.add("busy"); goT.textContent = "Signing in";
    fetch("/login", { method: "POST", headers: { "Content-Type": "application/json" }, credentials: "same-origin", body: JSON.stringify({ user: u.value, password: p.value }) })
      .then(function (r) { return r.json().then(function (j) { return [r.ok, j]; }); })
      .then(function (x) { if (x[0]) { goT.textContent = "Welcome"; f.classList.add("ok"); location.href = "/"; } else fail(x[1].error || "Sign-in failed."); })
      .catch(function () { fail("Could not reach the server."); });
  });

  // ------------------------------------------------------------------------------------------------ animated network (canvas)
  var cv = document.getElementById("net"), cx = cv.getContext("2d"), W = 0, H = 0, DPR = Math.min(window.devicePixelRatio || 1, 2);
  var nodes = [], packets = [], ripples = [], last = 0, running = true;
  function css(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim() || "#8b93ff"; }
  function resize() {
    W = window.innerWidth; H = window.innerHeight; cv.width = W * DPR; cv.height = H * DPR; cv.style.width = W + "px"; cv.style.height = H + "px"; cx.setTransform(DPR, 0, 0, DPR, 0, 0);
    var n = Math.max(24, Math.min(64, Math.round(W * H / 24000)));
    nodes = []; for (var i = 0; i < n; i++) nodes.push({ x: Math.random() * W, y: Math.random() * H, vx: (Math.random() - .5) * .18, vy: (Math.random() - .5) * .18, r: 1.6 + Math.random() * 2.2, hot: 0 });
    draw(true);
  }
  function links() {
    var out = [], D = 150;
    for (var i = 0; i < nodes.length; i++) for (var j = i + 1; j < nodes.length; j++) {
      var dx = nodes[i].x - nodes[j].x, dy = nodes[i].y - nodes[j].y, d = Math.sqrt(dx * dx + dy * dy); if (d < D) out.push([i, j, d / D]);
    }
    return out;
  }
  var L = [];
  function step(dt) {
    nodes.forEach(function (n) { n.x += n.vx * dt * .06; n.y += n.vy * dt * .06; if (n.x < -20) n.x = W + 20; if (n.x > W + 20) n.x = -20; if (n.y < -20) n.y = H + 20; if (n.y > H + 20) n.y = -20; n.hot = Math.max(0, n.hot - dt * .0006); });
    L = links();
    if (L.length && Math.random() < dt * .004) { var l = L[(Math.random() * L.length) | 0]; packets.push({ a: l[0], b: l[1], t: 0, sp: .0006 + Math.random() * .0009 }); }
    packets = packets.filter(function (q) { q.t += q.sp * dt; return q.t < 1; });
    if (Math.random() < dt * .00045) { var n = nodes[(Math.random() * nodes.length) | 0]; n.hot = 1; ripples.push({ x: n.x, y: n.y, t: 0 }); }
    ripples = ripples.filter(function (r) { r.t += dt * .0007; return r.t < 1; });
  }
  function draw(static_) {
    var acc = css("--accent"), acc2 = css("--accent-2"), bad = css("--critical"), line = css("--line-strong");
    cx.clearRect(0, 0, W, H);
    if (static_) L = links();
    L.forEach(function (l) {
      var a = nodes[l[0]], b = nodes[l[1]]; cx.globalAlpha = (1 - l[2]) * .55; cx.strokeStyle = line; cx.lineWidth = 1; cx.beginPath(); cx.moveTo(a.x, a.y); cx.lineTo(b.x, b.y); cx.stroke();
    });
    nodes.forEach(function (n) {
      cx.globalAlpha = .9; cx.fillStyle = n.hot > 0 ? bad : acc; cx.shadowColor = n.hot > 0 ? bad : acc; cx.shadowBlur = n.hot > 0 ? 18 : 8; cx.beginPath(); cx.arc(n.x, n.y, n.r + n.hot * 2.4, 0, 6.2832); cx.fill();
    });
    cx.shadowBlur = 0;
    packets.forEach(function (q) {
      var a = nodes[q.a], b = nodes[q.b], x = a.x + (b.x - a.x) * q.t, y = a.y + (b.y - a.y) * q.t; cx.globalAlpha = 1; cx.fillStyle = acc2; cx.shadowColor = acc2; cx.shadowBlur = 12; cx.beginPath(); cx.arc(x, y, 2.4, 0, 6.2832); cx.fill();
    });
    cx.shadowBlur = 0;
    ripples.forEach(function (r) { cx.globalAlpha = (1 - r.t) * .7; cx.strokeStyle = bad; cx.lineWidth = 1.5; cx.beginPath(); cx.arc(r.x, r.y, 8 + r.t * 70, 0, 6.2832); cx.stroke(); });
    cx.globalAlpha = 1;
  }
  function frame(t) {
    if (!running) return;
    var dt = Math.min(50, t - last || 16); last = t; step(dt); draw(false); requestAnimationFrame(frame);
  }
  window.addEventListener("resize", function () { clearTimeout(resize.t); resize.t = setTimeout(resize, 150); });
  document.addEventListener("visibilitychange", function () { running = !document.hidden && !REDUCED; if (running) { last = 0; requestAnimationFrame(frame); } });
  resize();
  if (!REDUCED) requestAnimationFrame(frame);
})();
