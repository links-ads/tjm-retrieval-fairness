"""Ranking-fairness measures for the Job-to-Talent direction, where the talent is the ranked item.

All of these measures require the protected attribute on the items being ranked, so they apply only
to Job$\\to$Talent: in Talent$\\to$Job the ranked items are vacancies, which carry no gender. Reported:

    MaxSkew@k, MinSkew@k, NDKL   Geyik et al. 2019 -- distributional, target = the pool itself
    DTR, DIR                     Singh & Joachims 2018 -- exposure per unit of relevance
    attention gap, L1            Biega et al. 2018 -- equity of attention amortised over the queries
    FA*IR                        Zehlike et al. 2017 -- post-processing, applied to the saved rankings

    uv run tools/analyze_ranking_fairness.py --output-dir outputs/fairness_analysis
"""

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd

from tjm.metrics import (
    amortized_attention,
    build_references,
    exposure_ratios,
    fair_rerank,
    individual_classification_rates,
    ndkl,
    skew_at_k,
)

SYSTEMS = [("qwen3_embedding", "Qwen3-Emb.$_{0.6B}$"), ("qwen8", "Qwen3-Rerank$_{8B}$")]
POOLS = [("full", "Full-Pool", 10), ("observed", "Application-Pool", 1)]
GROUPS = ("Female", "Male")
PROTECTED = "Female"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=str, default="outputs/zeroshot")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="triplets")
    parser.add_argument("--demographics", type=str, default="data/talents_demographics.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/fairness_analysis")
    parser.add_argument("--alpha", type=float, default=0.1, help="FA*IR significance level")
    return parser.parse_args()


def find_run(runs_dir: Path, system: str, split: str, mode: str, pool: str) -> Path | None:
    for path in sorted(p for ext in ("*.json", "*.json.gz")
                       for p in (runs_dir / system).rglob(ext)):
        if "reference" in path.name or "publications" in str(path) or split not in str(path):
            continue
        if path.name.startswith(f"{mode}_{pool}"):
            return path
    return None


def load_rankings(path: Path) -> dict[str, list[str]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        payload = json.load(handle)
    return {q: [e["candidate_id"] for e in r] for q, r in payload.items()}


def measure(rankings, references, gender, k, protected_share) -> dict:
    """Every measure for one set of rankings, so the original and the FA*IR output are comparable."""
    max_skews, min_skews, divergences = [], [], []
    for ranked in rankings.values():
        skews = skew_at_k(ranked, gender, k=k, desired=protected_share)
        if skews:
            max_skews.append(max(skews.values()))
            min_skews.append(min(skews.values()))
        divergences.append(ndkl(ranked, gender, desired=protected_share))
    ratios = exposure_ratios(rankings, references, gender, PROTECTED, "Male")
    attention = amortized_attention(rankings, references, gender)
    rates = individual_classification_rates(rankings, references, gender, k=k, group_side="candidate")
    return {
        "MaxSkew": sum(max_skews) / len(max_skews),
        "MinSkew": sum(min_skews) / len(min_skews),
        "NDKL": sum(divergences) / len(divergences),
        "DTR": ratios["dtr"],
        "DIR": ratios["dir"],
        "AttnGap": (attention.get(f"attention_share_{PROTECTED}", 0.0)
                    - attention.get(f"relevance_share_{PROTECTED}", 0.0)) * 100,
        "AttnL1": attention["unfairness"],
        "EO": abs(rates[PROTECTED]["true_positive_rate"] - rates["Male"]["true_positive_rate"]) * 100,
    }


def main() -> None:
    args = parse_args()
    demo = pd.read_csv(args.demographics)
    demo["id"] = demo["id"].astype(str)
    gender = {r.id: r.gender for r in demo.itertuples() if r.gender in GROUPS}
    references = build_references(args.data_dir, args.split, "job_to_talent")
    runs_dir = Path(args.runs_dir)

    rows = []
    for system, label in SYSTEMS:
        for pool, pool_label, k in POOLS:
            rankings = load_rankings(find_run(runs_dir, system, args.split, "job_to_talent", pool))
            rankings = {q: r for q, r in rankings.items() if len(r) >= 2}
            # target distribution: the pool the ranking was drawn from, so the measures isolate the
            # ranking rather than restating the corpus imbalance
            labelled = [gender[c] for r in rankings.values() for c in r if c in gender]
            share = {g: labelled.count(g) / len(labelled) for g in GROUPS}

            base = measure(rankings, references, gender, k, share)
            rows.append({"model": label, "pool": pool_label, "ranking": "original", **base})

            reranked = {
                q: fair_rerank(r, gender, protected=PROTECTED, p=share[PROTECTED],
                               k=min(len(r), 100), alpha=args.alpha)
                for q, r in rankings.items()
            }
            after = measure(reranked, references, gender, k, share)
            rows.append({"model": label, "pool": pool_label, "ranking": "FA*IR", **after})
            print(f"{label:22s} {pool_label:17s} target Female share {share[PROTECTED]*100:.1f}%", flush=True)

    frame = pd.DataFrame(rows)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "ranking_fairness.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n" + frame.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWritten to {out / 'ranking_fairness.csv'}")


if __name__ == "__main__":
    main()
