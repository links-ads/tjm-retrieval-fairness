from pathlib import Path

from .cascade import cascade_rankings, first_stage_recall, load_run
from .ranking import evaluate_run
from .significance import DEFAULT_RESAMPLES, DEFAULT_SEED, bootstrap_ci, bootstrap_std


def _run_path(runs_dir: str | Path, system: str, split: str, evaluation_mode: str, candidate_pool: str) -> Path:
    """Locate a saved run, accepting either a plain or a gzipped predictions file.

    A plain .json shadows a .json.gz of the same name, matching ``discover_runs``. The plain path is
    returned when neither exists, so the caller's own existence check reports the expected name.
    """
    split_dir = Path(runs_dir) / system / split
    plain = split_dir / f"{evaluation_mode}_{candidate_pool}.json"
    gzipped = split_dir / f"{evaluation_mode}_{candidate_pool}.json.gz"
    return gzipped if not plain.exists() and gzipped.exists() else plain


def build_cascade_grid(
    runs_dir: str | Path,
    split: str,
    first_stage_systems: list[str],
    reranker_systems: list[str],
    depths: list[int],
    evaluation_mode: str,
    references: dict[str, list[str]],
    k_values: list[int],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    per_query_out: dict[tuple[str, str, int], dict] | None = None,
) -> list[dict]:
    """Reconstruct every (first stage, reranker, depth) cascade from saved full-pool runs.

    Every cascade is exact, since a saved cross-encoder score does not depend on the candidate set it
    was computed alongside. Pairs whose runs are not both present on disk are skipped rather than
    failing the whole grid, since the paper-scoped grid and the appendix grid share this code but not
    every model combination exists for every direction.

    Args:
        runs_dir: Root holding one subdirectory per model.
        split: Split to read, e.g. "triplets".
        first_stage_systems: Retriever system names to use as shortlist sources.
        reranker_systems: Reranker system names to re-score the shortlist.
        depths: Shortlist sizes to evaluate.
        evaluation_mode: job_to_talent or talent_to_job.
        references: Maps query_id -> relevant candidate ids.
        k_values: Rank cutoffs for the cascade's own metrics.
        n_resamples: Bootstrap resamples per confidence interval.
        seed: Seed for the bootstrap.
        per_query_out: Optional dict populated with per-query values keyed by
            (first_stage, reranker, depth), for significance testing against the bare first stage.

    Returns:
        One row per (first_stage, reranker, depth) triple that could be built, with cascade metrics,
        their confidence intervals, and the first-stage recall ceiling at that depth.
    """
    rows: list[dict] = []

    for first_stage in first_stage_systems:
        first_path = _run_path(runs_dir, first_stage, split, evaluation_mode, "full")
        if not first_path.exists():
            continue
        first_run = load_run(first_path)

        for reranker in reranker_systems:
            reranker_path = _run_path(runs_dir, reranker, split, evaluation_mode, "full")
            if not reranker_path.exists():
                continue
            reranker_run = load_run(reranker_path)

            for depth in depths:
                try:
                    cascaded = cascade_rankings(first_run, reranker_run, depth)
                except ValueError:
                    # reranker run does not cover every shortlisted candidate at this depth
                    continue

                means, per_query = evaluate_run(cascaded, references, k_values)
                if not per_query:
                    continue
                if per_query_out is not None:
                    per_query_out[(first_stage, reranker, depth)] = per_query

                ceiling = first_stage_recall(first_run, references, depth)
                ceiling_values = [ceiling[qid] for qid in per_query if qid in ceiling]

                row = {
                    "first_stage": first_stage,
                    "reranker": reranker,
                    "depth": depth,
                    "evaluation_mode": evaluation_mode,
                    "n_queries": len(per_query),
                    "first_stage_recall": sum(ceiling_values) / len(ceiling_values) if ceiling_values else 0.0,
                }
                for metric, mean in means.items():
                    values = [values[metric] for values in per_query.values()]
                    _, lo, hi = bootstrap_ci(values, n_resamples=n_resamples, seed=seed)
                    row[metric] = mean
                    row[f"{metric}_ci_low"] = lo
                    row[f"{metric}_ci_high"] = hi
                    row[f"{metric}_std"] = bootstrap_std(values, n_resamples=n_resamples, seed=seed)
                rows.append(row)

    return rows
