document.getElementById("f").addEventListener("submit", function (ev) {
  ev.preventDefault();
  var e = document.getElementById("e");
  e.textContent = "";
  fetch("/login", {
    method: "POST", headers: { "Content-Type": "application/json" }, credentials: "same-origin",
    body: JSON.stringify({ user: document.getElementById("u").value, password: document.getElementById("p").value })
  })
    .then(function (r) { return r.json().then(function (j) { return [r.ok, j]; }); })
    .then(function (x) { if (x[0]) location.href = "/"; else e.textContent = x[1].error || "sign-in failed"; })
    .catch(function () { e.textContent = "could not reach the server"; });
});
