"""How often exact exposure parity is reachable within a single ranking.

A candidate's exposure is capped by the first position's discount while a merit ratio is not, so a
large enough imbalance leaves the exposure LP falling back to the closest achievable point. This
reports how often that happens, which bounds what the intervention in tab:fairness-ranking claims.

    uv run tools/analyze_exposure_feasibility.py --output-dir outputs/fairness_analysis
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tjm.metrics.exposure_lp import _normalised_merit, _position_discounts, _solve_exposure_lp

GROUPS = ("Female", "Male")
PROTECTED = "Female"
LP_WINDOW = 100
PARITY_TOLERANCE = 1e-4
SYSTEMS = [
    ("bm25", "BM25"),
    ("mpnet", "MPNet"),
    ("sheared_llama", "Sheared-LLaMA"),
    ("qwen3_embedding", "Qwen3-Emb.$_{0.6B}$"),
    ("qwen8", "Qwen3-Rerank$_{8B}$"),
]
POOLS = [("full", "Full-Pool"), ("observed", "Application-Pool")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=str, default="outputs/zeroshot")
    parser.add_argument("--split", type=str, default="triplets")
    parser.add_argument("--demographics", type=str, default="data/talents_demographics.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/fairness_analysis")
    return parser.parse_args()


def achieved_parity_ratio(window: list[str], scores: dict[str, float], gender: dict[str, str]) -> float | None:
    """The exposure-per-merit ratio the LP attains, or None when the query cannot define one."""
    merit = _normalised_merit(window, scores)
    protected = np.array([gender[candidate] == PROTECTED for candidate in window])
    if protected.all() or not protected.any():
        return None
    if merit[protected].sum() <= 0.0 or merit[~protected].sum() <= 0.0:
        return None

    exposure = _solve_exposure_lp(merit, protected) @ _position_discounts(len(window))
    return float((exposure[protected].sum() / merit[protected].sum())
                 / (exposure[~protected].sum() / merit[~protected].sum()))


def query_ratios(payload: dict, gender: dict[str, str]) -> np.ndarray:
    ratios = []
    for entries in payload.values():
        window = [e["candidate_id"] for e in entries if e["candidate_id"] in gender][:LP_WINDOW]
        if len(entries) < 2 or not window:
            continue
        ratio = achieved_parity_ratio(window, {e["candidate_id"]: e["score"] for e in entries}, gender)
        if ratio is not None:
            ratios.append(ratio)
    return np.array(ratios)


def main() -> None:
    args = parse_args()
    demo = pd.read_csv(args.demographics)
    demo["id"] = demo["id"].astype(str)
    gender = {r.id: r.gender for r in demo.itertuples() if r.gender in GROUPS}

    rows = []
    for system, label in SYSTEMS:
        for pool, pool_label in POOLS:
            path = Path(args.runs_dir) / system / args.split / f"job_to_talent_{pool}.json.gz"
            with gzip.open(path, "rt") as handle:
                values = query_ratios(json.load(handle), gender)

            unreachable = np.abs(values - 1.0) > PARITY_TOLERANCE
            rows.append({
                "model": label,
                "pool": pool_label,
                "n_queries": len(values),
                "n_unreachable": int(unreachable.sum()),
                "pct_unreachable": 100.0 * unreachable.mean(),
                "median_ratio": float(np.median(values)),
                "max_ratio": float(values.max()),
            })
            print(f"{label:22s} {pool_label:17s} parity unreachable on {unreachable.sum():4d}/{len(values)} "
                  f"({100.0 * unreachable.mean():5.1f}%)", flush=True)

    frame = pd.DataFrame(rows)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "exposure_feasibility.csv", index=False)
    print("\n" + frame.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nWritten to {out / 'exposure_feasibility.csv'}")


if __name__ == "__main__":
    main()
