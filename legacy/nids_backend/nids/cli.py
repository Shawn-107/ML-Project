"""Command line: python -m nids <train|evaluate|predict|explain|recalibrate|serve|unseen|stability|recal-curve|report|convert> ..."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from .config import REPORT_DIR, Config, DATA_DIR
from .data import load_splits, read_table
from .evaluate import benchmark_latency, evaluate, save_report
from .inference import Detector
from .pipeline import DEFAULT_BUNDLE, save_bundle, train


def _cfg(a) -> Config:
    return Config.fast() if getattr(a, "fast", False) else Config()


def _report_dir(a) -> Path:
    return Path(getattr(a, "report_dir", None) or REPORT_DIR)


def _meta_path(a) -> Path:
    return Path(a.bundle).with_suffix(".meta.json")


def _detector(a) -> Detector:
    d = Detector.load(a.bundle, policy=a.policy, use_novelty=a.novelty)
    return d


def cmd_train(a):
    bundle = train(_cfg(a))
    p = save_bundle(bundle, a.bundle)
    print(f"saved {p}")
    print("validation metrics (default policy):", json.dumps(bundle.meta["validation_metrics"], indent=2, default=float))
    print("thresholds:", json.dumps(bundle.thresholds, indent=2))


def cmd_evaluate(a):
    test = load_splits(_cfg(a)).test
    d = _detector(a)
    res = evaluate(d, test)
    res["latency"] = benchmark_latency(d, test)
    p = save_report(res, f"evaluation_{d.policy}{'_novelty' if d.use_novelty else ''}.json", _report_dir(a))
    print(json.dumps({k: v for k, v in res.items() if k != "per_class"}, indent=2, default=float))
    for k, v in res["per_class"].items():
        print(f"  {k:15s} n={v['n']:6d} flagged={v['flagged']:.3f}")
    print(f"report: {p}")


def cmd_predict(a):
    df = read_table(a.input)
    out = _detector(a).predict_frame(df)
    res = pd.concat([df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)
    res.to_csv(a.output, index=False)
    print(f"wrote {len(res)} predictions to {a.output}; flagged {int(out.is_attack.sum())}")


def cmd_explain(a):
    df = read_table(a.input)
    if not 0 <= a.row < len(df):
        raise ValueError(f"--row must be between 0 and {len(df) - 1}")
    res = _detector(a).explain(df.iloc[a.row].to_dict(), n_permutations=a.permutations, top=a.top)
    print(json.dumps(res, indent=2, default=float))


def _load_for_experiment(a):
    cfg = _cfg(a)
    return cfg, load_splits(cfg)


def cmd_unseen(a):
    from .experiments import leave_one_attack_out
    cfg, sp = _load_for_experiment(a)
    seen = _detector(a) if Path(a.bundle).exists() else None
    attacks = [x.strip() for x in a.attacks.split(",") if x.strip()] if a.attacks else None
    res = leave_one_attack_out(cfg, sp, attacks, seen_detector=seen, policy=a.policy, progress=print)
    p = save_report(res, "unseen_attacks.json", _report_dir(a))
    print(json.dumps(res["summary_mean_over_folds"], indent=2)); print(f"report: {p}")


def cmd_stability(a):
    from .experiments import stability
    res = stability(_cfg(a), n_repeats=a.repeats, policy=a.policy, progress=print)
    p = save_report(res, "stability.json", _report_dir(a))
    print(json.dumps(res["summary"], indent=2)); print(f"report: {p}")


def cmd_recal_curve(a):
    from .experiments import recalibration_curve
    test = load_splits(_cfg(a)).test
    res = recalibration_curve(_detector(a), test, draws=a.draws)
    p = save_report(res, "recalibration_curve.json", _report_dir(a))
    print(json.dumps(res, indent=2, default=float)); print(f"report: {p}")


def cmd_report(a):
    from .report import write_report
    print(f"wrote {write_report(_report_dir(a), _meta_path(a))}")


def cmd_recalibrate(a):
    d = _detector(a)
    rep = d.recalibrate(read_table(a.input), fpr_cap=a.fpr_cap)
    print(json.dumps(rep, indent=2, default=float))
    if rep["applied"]:
        d.save_override(a.override)
        print(f"saved override to {a.override} (load it with Detector.load_override)")


def cmd_serve(a):
    from .server import make_server
    d = _detector(a)
    if a.override and Path(a.override).exists():
        d.load_override(a.override)
    srv = make_server(d, a.host, a.port, allow_admin=a.allow_admin or a.allow_recalibrate)
    print(f"serving on http://{a.host}:{a.port}  (policy={d.policy}, threshold={d.threshold:.4f}) - Ctrl+C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def cmd_convert(a):
    df = read_table(a.input)
    df.to_csv(a.output, index=False)
    print(f"{a.input} -> {a.output} {df.shape}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nids", description="Network intrusion detection backend")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
        p.add_argument("--policy", default=None, help="e.g. fpr_1pct, fpr_3pct, fpr_5pct, f1")
        p.add_argument("--novelty", action="store_true", help="also flag flows the autoencoder finds novel")

    def reports(p):
        p.add_argument("--report-dir", default=str(REPORT_DIR))

    def fast(p):
        p.add_argument("--fast", action="store_true", help="small/quick configuration (for tests and smoke runs)")

    p = sub.add_parser("train"); fast(p); p.add_argument("--bundle", default=str(DEFAULT_BUNDLE)); p.set_defaults(f=cmd_train)
    p = sub.add_parser("evaluate"); common(p); fast(p); reports(p); p.set_defaults(f=cmd_evaluate)
    p = sub.add_parser("explain"); common(p); p.add_argument("--input", required=True); p.add_argument("--row", type=int, default=0)
    p.add_argument("--permutations", type=int, default=16); p.add_argument("--top", type=int, default=10); p.set_defaults(f=cmd_explain)
    p = sub.add_parser("unseen"); common(p); fast(p); reports(p); p.add_argument("--attacks", default=None, help="comma-separated families (default: all)"); p.set_defaults(f=cmd_unseen)
    p = sub.add_parser("stability"); common(p); fast(p); reports(p); p.add_argument("--repeats", type=int, default=3); p.set_defaults(f=cmd_stability)
    p = sub.add_parser("recal-curve"); common(p); fast(p); reports(p); p.add_argument("--draws", type=int, default=30); p.set_defaults(f=cmd_recal_curve)
    p = sub.add_parser("report"); common(p); reports(p); p.set_defaults(f=cmd_report)
    p = sub.add_parser("predict"); common(p); p.add_argument("--input", required=True); p.add_argument("--output", required=True); p.set_defaults(f=cmd_predict)
    p = sub.add_parser("recalibrate"); common(p); p.add_argument("--input", required=True)
    p.add_argument("--fpr-cap", type=float, default=0.08); p.add_argument("--override", default=str(DEFAULT_BUNDLE.with_name("override.json"))); p.set_defaults(f=cmd_recalibrate)
    p = sub.add_parser("serve"); common(p); p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8008)
    p.add_argument("--allow-admin", action="store_true", help="enable /config, /recalibrate and /reset")
    p.add_argument("--allow-recalibrate", action="store_true", help=argparse.SUPPRESS); p.add_argument("--override", default=str(DEFAULT_BUNDLE.with_name("override.json"))); p.set_defaults(f=cmd_serve)
    p = sub.add_parser("convert"); p.add_argument("input"); p.add_argument("output"); p.set_defaults(f=cmd_convert)

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        a.f(a)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
