"""Collect finished runs from experiments/ into one results table.

    python scripts/collect_results.py                 # print per-run and per-group tables (test split)
    python scripts/collect_results.py --split val
    python scripts/collect_results.py --write         # also update the tables in experiments/README.md

Each run lives in experiments/<run>/ with the small files copied from results/<run>/
(run_config.json, metrics.csv, eval_val_best.json, eval_test_best.json). Runs named
<group>_s<seed> (e.g. unet_a_s42, unet_a_s43) are grouped and reported as mean ± std
over seeds (sample std, ddof=1).
"""
import argparse
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
CLASSES = ["bg", "intact", "damaged", "destroyed"]
COLS = ["mIoU", "event mIoU"] + [f"IoU {c}" for c in CLASSES] + ["F1_loc", "F1_clf", "OA"]
START, END = "<!-- results:{}:start -->", "<!-- results:{}:end -->"


def load_runs(split):
    runs = []
    for d in sorted(p for p in EXP.iterdir() if p.is_dir()):
        f = d / f"eval_{split}_best.json"
        if not f.exists():
            continue
        e = json.loads(f.read_text())
        m = e["metrics"]
        cfg_f = d / "run_config.json"
        cfg = json.loads(cfg_f.read_text()) if cfg_f.exists() else {}
        git = cfg.get("git", {})
        commit = (git.get("commit") or "")[:7] or "?"
        if git.get("dirty"):
            commit += " (dirty)"
        vals = [m["miou"], m["event_miou"], *m["iou_per_class"], m["loc_f1"], m["clf_f1"], m["oa"]]
        mt = re.fullmatch(r"(.+)_s(\d+)", d.name)
        runs.append({"run": d.name, "group": mt.group(1) if mt else d.name,
                     "seed": int(mt.group(2)) if mt else None, "epoch": e.get("epoch"),
                     "commit": commit, "gpu": cfg.get("machine", {}).get("gpu") or "?",
                     "vals": [100 * v for v in vals]})
    return runs


def run_table(runs):
    head = ["run", "seed", "best epoch", *COLS, "commit", "GPU"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in runs:
        cells = [r["run"], str(r["seed"] if r["seed"] is not None else "–"), str(r["epoch"]),
                 *[f"{v:.2f}" for v in r["vals"]], r["commit"], r["gpu"].replace("NVIDIA ", "")]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def group_table(runs):
    groups = {}
    for r in runs:
        groups.setdefault(r["group"], []).append(r)
    head = ["group", "seeds", *COLS]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for g, rs in groups.items():
        cols = list(zip(*(r["vals"] for r in rs)))
        if len(rs) > 1:
            cells = [f"{statistics.mean(c):.2f} ± {statistics.stdev(c):.2f}" for c in cols]
        else:
            cells = [f"{c[0]:.2f}" for c in cols]
        seeds = ",".join(str(r["seed"]) for r in rs if r["seed"] is not None) or "–"
        lines.append("| " + " | ".join([g, seeds, *cells]) + " |")
    return "\n".join(lines)


def replace_block(text, tag, body):
    s, e = START.format(tag), END.format(tag)
    if s not in text or e not in text:
        raise SystemExit(f"markers {s} / {e} not found in experiments/README.md")
    i, j = text.index(s) + len(s), text.index(e)
    return text[:i] + "\n" + body + "\n" + text[j:]


def main():
    p = argparse.ArgumentParser(description="Results table from experiments/")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--write", action="store_true", help="update the tables in experiments/README.md")
    args = p.parse_args()

    runs = load_runs(args.split)
    if not runs:
        raise SystemExit(f"no experiments/*/eval_{args.split}_best.json found")
    per_run, per_group = run_table(runs), group_table(runs)
    print(f"## {args.split}: per group (mean ± std over seeds)\n\n{per_group}\n\n## {args.split}: per run\n\n{per_run}")

    if args.write:
        readme = EXP / "README.md"
        text = readme.read_text()
        text = replace_block(text, f"{args.split}-groups", per_group)
        text = replace_block(text, f"{args.split}-runs", per_run)
        readme.write_text(text)
        print(f"\nupdated {readme}")


if __name__ == "__main__":
    main()
