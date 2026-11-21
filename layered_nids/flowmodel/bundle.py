"""Bundle = everything needed to serve; saved with a sha256 sidecar and a schema/version check."""
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from .contract import SCHEMA_VERSION

log = logging.getLogger(__name__)
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "models" / "csv" / "unsw_flow_model.joblib"


@dataclass
class Bundle:
    prep: object
    model: object
    kmeans: object
    thresholds: dict
    novelty_threshold: float
    background: pd.DataFrame
    meta: dict
    adapt_ref: object = None          # validation features/scores used by Detector.adapt (None in older bundles)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def _meta(path):
    return Path(path).with_suffix(".meta.json")


def save_bundle(b: Bundle, path=DEFAULT_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    joblib.dump(b, tmp, compress=3)
    tmp.replace(path)
    meta = dict(b.meta, sha256=_sha(path), thresholds=b.thresholds, novelty_threshold=b.novelty_threshold)
    _meta(path).write_text(json.dumps(meta, indent=2, default=float))
    return path


def load_bundle(path=DEFAULT_PATH, verify=True) -> Bundle:
    """SECURITY: bundles are pickles - only load files you trained. The checksum detects corruption, not a malicious author."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"model bundle not found: {path}")
    if verify:
        if not _meta(path).exists():
            raise RuntimeError(f"missing {_meta(path).name}; cannot verify bundle")
        if json.loads(_meta(path).read_text()).get("sha256") != _sha(path):
            raise RuntimeError("bundle checksum mismatch: file corrupted or modified after training")
    b = joblib.load(path)
    if b.meta.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"bundle schema v{b.meta.get('schema_version')} != code schema v{SCHEMA_VERSION}")
    if b.meta.get("sklearn") != sklearn.__version__:
        log.warning("bundle trained with scikit-learn %s, running %s", b.meta.get("sklearn"), sklearn.__version__)
    return b
