(function () {
  var b = document.getElementById("nids-out");
  if (b) b.addEventListener("click", function () { fetch("/logout", { method: "POST", credentials: "same-origin" }).then(function () { location.href = "/login"; }); });
})();
