"""Dashboard CSV upload helper: input checks and download-path safety (no server, no models needed)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "dashboard"))
import nids_csv  # noqa: E402


def test_rejects_empty_unknown_policy_and_too_large(tmp_path):
    assert "empty" in nids_csv.handle_upload(b"", "a.csv", "fpr_3pct", tmp_path)["error"]
    assert "policy" in nids_csv.handle_upload(b"x", "a.csv", "fpr_99pct", tmp_path)["error"]
    big = type("B", (bytes,), {})(b"x")
    nids_csv.MAX_BYTES, old = 0, nids_csv.MAX_BYTES
    try:
        assert "larger" in nids_csv.handle_upload(big, "a.csv", "fpr_3pct", tmp_path)["error"]
    finally:
        nids_csv.MAX_BYTES = old


@pytest.mark.parametrize("uid,kind", [("../../server.conf", "scored"), ("0" * 31, "scored"), ("g" * 32, "scored"), ("0" * 32, "../x"), ("", "scored"), (None, "scored")])
def test_download_ids_are_strictly_validated(uid, kind):
    assert nids_csv.result_file(uid, kind) is None


def test_unreadable_file_gives_an_error_not_a_crash(tmp_path):
    r = nids_csv.handle_upload(b"hello,world\n1,2\n", "a.csv", "fpr_3pct", tmp_path)
    assert "unrecognised flow file" in r["error"]
    assert not list(nids_csv.UPLOADS.glob("*.input"))          # the upload itself is not kept
