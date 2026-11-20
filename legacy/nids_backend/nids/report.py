"""Build reports/REPORT.md from the JSON reports and the bundle metadata (numbers are copied, never typed by hand)."""
from __future__ import annotations

import json
from pathlib import Path


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _pct(x) -> str:
    return f"{100 * x:.1f}%"


def build_report(report_dir: Path, bundle_meta_path: Path) -> str:
    meta = _load(bundle_meta_path)
    if meta is None:
        raise FileNotFoundError(f"bundle metadata not found or unreadable: {bundle_meta_path}")
    L: list[str] = ["# Network Intrusion Detection - results", ""]
    di, vm = meta["data_info"], meta["validation_metrics"]
    L += ["## Data", "",
          f"UNSW-NB15 official split. Training {di['train_raw']:,} rows -> {di['train_dedup']:,} after removing rows whose "
          f"feature vector repeats or carries conflicting labels ({di['train_cleaning']['dropped_conflicting_label']:,} conflicting). "
          f"Test {di['test_raw']:,} -> {di['test_after_overlap_removal']:,} after the same cleaning and removing rows that also occur in training.",
          f"Model fitting used {di['n_train']:,} rows, threshold selection {di['n_val']:,} validation rows, final evaluation {di['n_test']:,} test rows (touched only by `evaluate`).", ""]
    L += ["## Validation (same distribution as training)", "",
          f"Policy `{meta['default_policy']}`: accuracy {_pct(vm['accuracy'])}, recall {_pct(vm['recall'])}, "
          f"false-alarm rate {_pct(vm['false_alarm_rate'])}, F1 {_pct(vm['f1'])}.", ""]

    evals = sorted(report_dir.glob("evaluation_*.json"))
    if evals:
        L += ["## Official test set", "", "| Policy | Novelty | Accuracy | Balanced acc. | Recall | False alarms | F1 | Brier | ECE |",
              "|---|---|---|---|---|---|---|---|---|"]
        for p in evals:
            e = _load(p)
            if not e:
                continue
            c = e.get("calibration", {})
            L.append(f"| {e['policy']} | {'yes' if e['use_novelty'] else 'no'} | {_pct(e['accuracy'])} | {_pct(e['balanced_accuracy'])} | "
                     f"{_pct(e['recall'])} | {_pct(e['false_alarm_rate'])} | {_pct(e['f1'])} | "
                     f"{c.get('brier', float('nan')):.3f} | {c.get('ece', float('nan')):.3f} |")
        L += ["", "The official test set is harder than the training distribution (known distribution shift), so test accuracy is "
              "lower than validation accuracy. Report both.", ""]
        first = _load(evals[0])
        if first and "latency" in first:
            lt = first["latency"]
            L += [f"Latency: {lt['single_flow_ms']:.1f} ms per flow one at a time, {lt['batched_ms_per_flow']:.2f} ms per flow in batches.", ""]

    un = _load(report_dir / "unseen_attacks.json")
    if un:
        L += ["## Attacks the model never saw (leave-one-attack-out)", "",
              f"Each row retrains without that attack family, then measures how many of its test flows are caught "
              f"(policy `{un['policy']}`, novelty detector at {_pct(un['novelty_false_alarm_target'])} validation false alarms).", "",
              "| Held-out attack | Flows | Classifier | Novelty | Combined | Detected when trained on it |", "|---|---|---|---|---|---|"]
        for a, f in un["folds"].items():
            seen = _pct(f["detect_when_seen"]) if "detect_when_seen" in f else "-"
            L.append(f"| {a} | {f['n_test_attack']:,} | {_pct(f['detect_classifier'])} | {_pct(f['detect_novelty'])} | {_pct(f['detect_combined'])} | {seen} |")
        s = un["summary_mean_over_folds"]
        L += ["", f"Mean over families: classifier {_pct(s['detect_classifier'])}, combined {_pct(s['detect_combined'])}. "
              f"False alarms on Normal test flows: {_pct(s['false_alarm_classifier'])} (classifier) vs {_pct(s['false_alarm_combined'])} (with novelty).", ""]

    st = _load(report_dir / "stability.json")
    if st:
        L += ["## Stability over repeated training runs", "", f"{st['n_repeats']} runs with different seeds and validation splits.", "",
              "| Metric | Validation mean ± std | Test mean ± std |", "|---|---|---|"]
        for k in ("accuracy", "recall", "false_alarm_rate", "f1"):
            v, t = st["summary"]["validation"][k], st["summary"]["test"][k]
            L.append(f"| {k} | {_pct(v['mean'])} ± {_pct(v['std'])} | {_pct(t['mean'])} ± {_pct(t['std'])} |")
        L.append("")

    rc = _load(report_dir / "recalibration_curve.json")
    if rc:
        n0 = rc["no_target_labels"]
        L += ["## Local recalibration", "",
              f"Without any local labels: accuracy {_pct(n0['accuracy'])}, false alarms {_pct(n0['false_alarm_rate'])}. "
              "Accuracy after tuning the threshold on n labelled local flows:", "",
              "| Labelled flows | Accuracy | Recall | False alarms |", "|---|---|---|---|"]
        for r in rc["curve"]:
            L.append(f"| {r['n_labelled']} | {_pct(r['accuracy']['mean'])} ± {_pct(r['accuracy']['std'])} | "
                     f"{_pct(r['recall']['mean'])} | {_pct(r['false_alarm_rate']['mean'])} |")
        L.append("")

    L += ["## Limits", "",
          "- Trained on UNSW-NB15 (2015, synthetic lab traffic). Real networks differ; recalibrate on local labelled flows.",
          "- Works on flow records, not raw packets. The 'real-time' demo replays stored flows; there is no live capture.",
          "- The drift monitor did not detect the shift between UNSW-NB15 training and test data; it is only a sanity check.",
          "- Explanations are Shapley-value estimates from a Monte-Carlo sampler written for this project (the `shap` package was not available).",
          "- The HTTP service has no authentication and must stay on localhost.", ""]
    return "\n".join(L)


def write_report(report_dir: Path, bundle_meta_path: Path, name: str = "REPORT.md") -> Path:
    text = build_report(report_dir, bundle_meta_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    p = report_dir / name
    p.write_text(text)
    return p
