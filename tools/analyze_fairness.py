"""Gender fairness analysis for both evaluation directions and both candidate pools.

Talents sit on a different side of the ranking in each direction, so the two directions answer
different questions and use different units of analysis:

    talent_to_job  -- the talent IS the query, so the attribute describes who is being served.
                      Unit: one query. Measure: quality-of-service parity, i.e. do the metrics a
                      talent receives depend on their gender?

    job_to_talent  -- the talent is a CANDIDATE, so the attribute describes who is being ranked.
                      Unit: one (query, relevant talent) pair for equal opportunity, and one query
                      for top-k composition. Measures: is a relevant talent surfaced as readily
                      regardless of gender, and does the shortlist over- or under-represent a group
                      relative to the pool it was drawn from?

Both are reported for the Full-Pool and the Application-Pool. The pools share an identical query set,
and differ only in which candidates are available, which is what makes the contrast interpretable:
Full-Pool composition is fixed and known, so disparity there is the model's; the Application-Pool is
already filtered by who applied, so disparity there mixes the model with that pipeline.

    uv run tools/analyze_fairness.py --runs-dir outputs/zeroshot --data-dir data \
        --demographics data/talents_demographics.csv --output-dir outputs/fairness_analysis
"""

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd

from tjm.metrics import (
    build_references,
    permutation_group_test,
    relevant_candidate_outcomes,
    representation_gap,
    summarise_groups,
)

EVALUATION_MODES = ("job_to_talent", "talent_to_job")
CANDIDATE_POOLS = ("full", "observed")
# Two groups large enough to support inference; every other value is reported as excluded instead of
# being folded into one of these, which would misattribute those talents.
COMPARED_GROUPS = ("Female", "Male")
QUALITY_METRICS = ("recall@10", "ndcg@10", "recall@1", "mrr")
DEFAULT_K = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=str, default="outputs/zeroshot")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="triplets")
    parser.add_argument("--demographics", type=str, default="data/talents_demographics.csv")
    parser.add_argument("--per-query", type=str, default="outputs/analysis/per_query_metrics.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/fairness_analysis")
    parser.add_argument("--systems", type=str, nargs="+", default=["bm25", "qwen8"])
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-pool",
        type=int,
        default=2,
        help="Skip queries whose candidate pool is smaller than this: with one candidate there is no "
        "ranking decision to be fair or unfair about.",
    )
    return parser.parse_args()


def load_gender(path: str) -> tuple[dict[str, str], dict[str, int]]:
    """Map talent id to gender, and report how many talents each value covers."""
    frame = pd.read_csv(path)
    frame["id"] = frame["id"].astype(str)
    counts = frame["gender"].value_counts(dropna=False).to_dict()
    gender = {
        row.id: row.gender
        for row in frame.itertuples()
        if isinstance(row.gender, str) and row.gender in COMPARED_GROUPS
    }
    return gender, {str(key): int(value) for key, value in counts.items()}


def find_run(runs_dir: Path, system: str, split: str, mode: str, pool: str) -> Path | None:
    for path in sorted(p for ext in ("*.json", "*.json.gz")
                       for p in (runs_dir / system).rglob(ext)):
        name = path.name
        if "reference" in name or "publications" in str(path):
            continue
        if split not in str(path):
            continue
        if name.startswith(f"{mode}_{pool}"):
            return path
    return None


def load_rankings(path: Path) -> dict[str, list[str]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        payload = json.load(handle)
    return {query_id: [entry["candidate_id"] for entry in ranked] for query_id, ranked in payload.items()}


def compare(values_by_group: dict[str, list[float]], n_resamples: int, seed: int) -> dict[str, float] | None:
    first, second = (values_by_group.get(group, []) for group in COMPARED_GROUPS)
    if not first or not second:
        return None
    return permutation_group_test(first, second, n_resamples=n_resamples, seed=seed)


def main() -> None:
    args = parse_args()
    gender, coverage = load_gender(args.demographics)
    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Gender values in the demographics file:", coverage, flush=True)
    print(f"Talents usable for the {COMPARED_GROUPS[0]}-vs-{COMPARED_GROUPS[1]} comparison: {len(gender)}", flush=True)

    per_query = pd.read_csv(args.per_query, dtype={"query_id": str})
    summary_rows: list[dict] = []
    test_rows: list[dict] = []

    for system in args.systems:
        for pool in CANDIDATE_POOLS:
            # --- talent_to_job: the talent is the query, so stratify the per-query metrics ---
            served = per_query[
                (per_query.system == system)
                & (per_query.evaluation_mode == "talent_to_job")
                & (per_query.candidate_pool == pool)
            ]
            run = find_run(runs_dir, system, args.split, "talent_to_job", pool)
            pool_size = {qid: len(ranked) for qid, ranked in load_rankings(run).items()} if run else {}
            eligible = served[served.query_id.map(lambda q: pool_size.get(q, 0) >= args.min_pool)]

            for metric in QUALITY_METRICS:
                if metric not in eligible.columns:
                    continue
                values = {
                    group: eligible[eligible.query_id.map(gender.get) == group][metric].tolist()
                    for group in COMPARED_GROUPS
                }
                for row in summarise_groups(values, n_resamples=args.n_resamples, seed=args.seed):
                    summary_rows.append(
                        {
                            "system": system,
                            "evaluation_mode": "talent_to_job",
                            "candidate_pool": pool,
                            "analysis": "quality_of_service",
                            "measure": metric,
                            "unit": "query",
                            **row,
                        }
                    )
                result = compare(values, args.n_resamples, args.seed)
                if result:
                    test_rows.append(
                        {
                            "system": system,
                            "evaluation_mode": "talent_to_job",
                            "candidate_pool": pool,
                            "analysis": "quality_of_service",
                            "measure": metric,
                            "unit": "query",
                            "group_a": COMPARED_GROUPS[0],
                            "group_b": COMPARED_GROUPS[1],
                            **result,
                        }
                    )

            # --- job_to_talent: the talent is a candidate, so use the rankings ---
            run = find_run(runs_dir, system, args.split, "job_to_talent", pool)
            if run is None:
                print(f"  no saved run for {system} job_to_talent {pool}, skipping", flush=True)
                continue
            rankings = load_rankings(run)
            rankings = {qid: ranked for qid, ranked in rankings.items() if len(ranked) >= args.min_pool}
            references = build_references(args.data_dir, args.split, "job_to_talent")

            # equal opportunity: among relevant talents, is retrieval independent of gender?
            jobs = [(k, measure) for k in (1, args.k) for measure in ("retrieved",)]
            jobs.append((args.k, "reciprocal_rank"))
            for cutoff, measure in jobs:
                outcomes = relevant_candidate_outcomes(rankings, references, k=cutoff)
                values = {group: [] for group in COMPARED_GROUPS}
                for outcome in outcomes:
                    group = gender.get(outcome["candidate_id"])
                    if group in values:
                        values[group].append(outcome[measure])
                label = f"retrieved@{cutoff}" if measure == "retrieved" else "mrr"
                for row in summarise_groups(values, n_resamples=args.n_resamples, seed=args.seed):
                    summary_rows.append(
                        {
                            "system": system,
                            "evaluation_mode": "job_to_talent",
                            "candidate_pool": pool,
                            "analysis": "equal_opportunity",
                            "measure": label,
                            "unit": "relevant_pair",
                            **row,
                        }
                    )
                result = compare(values, args.n_resamples, args.seed)
                if result:
                    test_rows.append(
                        {
                            "system": system,
                            "evaluation_mode": "job_to_talent",
                            "candidate_pool": pool,
                            "analysis": "equal_opportunity",
                            "measure": label,
                            "unit": "relevant_pair",
                            "group_a": COMPARED_GROUPS[0],
                            "group_b": COMPARED_GROUPS[1],
                            **result,
                        }
                    )

            # representation: does the top-k mirror the pool it was drawn from? Paired within
            # query, and reported per group rather than as a Female-vs-Male gap, since comparing the
            # two groups' shares would test a fifty-fifty null and only restate the pool imbalance.
            gaps = representation_gap(rankings, gender, k=args.k)
            for row in summarise_groups(gaps, n_resamples=args.n_resamples, seed=args.seed):
                summary_rows.append(
                    {
                        "system": system,
                        "evaluation_mode": "job_to_talent",
                        "candidate_pool": pool,
                        "analysis": "representation",
                        "measure": f"share@{args.k}_minus_pool_share",
                        "unit": "query",
                        **row,
                    }
                )

    summary = pd.DataFrame(summary_rows)
    tests = pd.DataFrame(test_rows)
    summary.to_csv(output_dir / "group_summary.csv", index=False)
    tests.to_csv(output_dir / "group_tests.csv", index=False)
    print(f"\nGroup summaries: {len(summary)} rows | group tests: {len(tests)} rows", flush=True)
    print(f"Written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
