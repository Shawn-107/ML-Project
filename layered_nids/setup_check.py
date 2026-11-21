"""Check a fresh clone: what is present, what is missing, and the exact command that fixes each missing piece.

    python -m layered_nids.setup_check

Exit code 0 when everything REQUIRED for the tests and the CSV mode is present, 1 otherwise. Optional pieces are reported but never fail the check.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

from layered_nids.common.paths import DATA, MODELS, ROOT

PY = r".venv313\Scripts\python.exe"


def _have(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except Exception:                                                  # a missing or blocked package both mean "not usable"
        return False


def checks() -> list[tuple[str, bool, bool, str]]:
    """(what, ok, required, how to fix)"""
    unsw = DATA / "unsw"
    rows = [
        ("Python 3.13", sys.version_info[:2] == (3, 13), False, "use the Python 3.13 environment described in requirements-313.txt"),
        ("packages: numpy, pandas, scikit-learn, torch, scapy", all(_have(m) for m in ("numpy", "pandas", "sklearn", "torch", "scapy")), True,
         f"{PY} -m pip install --only-binary=:all: -r requirements-313.txt"),
        ("package layered_nids and the legacy flow package importable", _have("nids"), True, f"{PY} -m pip install --no-deps -e . -e legacy/nids_backend"),
        ("UNSW-NB15 training file", (unsw / "official_training_175341.csv").exists(), True, "copy the file into data/unsw/ (see data/SOURCES.md)"),
        ("UNSW-NB15 testing file", (unsw / "official_testing_82332.csv").exists(), True, "copy the file into data/unsw/ (see data/SOURCES.md)"),
        ("CSV model: UNSW flow model", (MODELS / "csv" / "unsw34.joblib").exists(), True, f"{PY} -m layered_nids.csvmode.models train --which unsw34"),
        ("CSV model: NetFlow", (MODELS / "csv" / "netflow_v2.joblib").exists(), False,
         f"put NF-ToN-IoT-v2-test.csv in data/netflow/ (or set NIDS_NETFLOW_CSV), then {PY} -m layered_nids.csvmode.models train --which netflow_v2"),
        ("packet capture for the pipeline and the benchmarks", (DATA / "mirai" / "mirai.pcap").exists(), False, "put the public Mirai capture at data/mirai/mirai.pcap (see data/SOURCES.md)"),
        ("DGA domain list", (DATA / "dga" / "dga_legit.csv").exists(), False, "download the Mendeley DGA dataset into data/dga/ (see data/SOURCES.md)"),
        ("known-bad indicator file", (DATA / "trails" / "trails.csv").exists(), False, "optional: place your licensed indicator file at data/trails/trails.csv (see data/SOURCES.md)"),
        ("dashboard user", (ROOT / "local" / "users.json").exists(), False, f"{PY} -m layered_nids.dashboard --init-user NAME"),
        ("external sensor configured", bool(os.environ.get("NIDS_SENSOR_WSL_ROOT") and os.environ.get("NIDS_SENSOR_SHARE")), False,
         "optional: set NIDS_SENSOR_WSL_ROOT and NIDS_SENSOR_SHARE (see layered_nids/signatures/external.py)"),
    ]
    return rows


def main() -> int:
    bad = 0
    for what, ok, required, fix in checks():
        tag = "ok     " if ok else ("MISSING" if required else "optional")
        print(f"[{tag}] {what}" + ("" if ok else f"\n           fix: {fix}"))
        bad += (not ok) and required
    print("\nready for the tests and CSV mode" if not bad else f"\n{bad} required item(s) missing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
