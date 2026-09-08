from collections import defaultdict
from pathlib import Path

import pandas as pd

from .cascade import load_run
from .ranking import evaluate_run
from .significance import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    bootstrap_ci,
    bootstrap_std,
    holm_bonferroni,
    paired_bootstrap_test,
)

PerQuery = dict[str, dict[str, float]]
RunKey = tuple[str, str, str]

FIT_LABEL = "Fit"
EVALUATION_MODES = ("job_to_talent", "talent_to_job")
CANDIDATE_POOLS = ("full", "observed")


def build_references(data_dir: str, split: str, evaluation_mode: str) -> dict[str, list[str]]:
    """Rebuild relevance judgements from the dataset CSVs, without loading a model.

    Mirrors the filtering the evaluation harness applies: a pair only counts when both documents have
    text, so positives whose resume failed to parse are excluded rather than being counted as
    unreachable relevants.

    Args:
        data_dir: Directory holding jobs.csv, resumes.csv and the split file.
        split: Split name, e.g. "triplets".
        evaluation_mode: job_to_talent or talent_to_job.

    Returns:
        Maps query_id -> relevant candidate ids, omitting queries with no usable positives.
    """
    jobs = pd.read_csv(f"{data_dir}/jobs.csv")
    resumes = pd.read_csv(f"{data_dir}/resumes.csv")
    triplets = pd.read_csv(f"{data_dir}/{split}.csv")

    job_ids = {int(i) for i, text in zip(jobs["id"], jobs["text"].fillna("")) if str(text).strip()}
    talent_ids = {int(i) for i, text in zip(resumes["id"], resumes["cleaned_text"].fillna("")) if str(text).strip()}

    references: dict[str, list[str]] = defaultdict(list)
    for _, row in triplets[triplets["label"] == FIT_LABEL].iterrows():
        job_id, talent_id = int(row["vacancy_id"]), int(row["talent_id"])
        if job_id not in job_ids or talent_id not in talent_ids:
            continue
        if evaluation_mode == "job_to_talent":
            references[str(job_id)].append(str(talent_id))
        else:
            references[str(talent_id)].append(str(job_id))

    return {qid: sorted(candidates) for qid, candidates in references.items()}


def discover_runs(
    runs_dir: str | Path,
    split: str,
    evaluation_mode: str | None = None,
    candidate_pool: str | None = None,
) -> dict[RunKey, Path]:
    """Locate saved prediction files under a results directory.

    Args:
        runs_dir: Root holding one subdirectory per model, e.g. outputs/zeroshot. Predictions
            may be stored as ``.json`` or gzipped ``.json.gz``.
        split: Split subdirectory to read, e.g. "triplets".
        evaluation_mode: Optional filter.
        candidate_pool: Optional filter.

    Returns:
        Maps (system, evaluation_mode, candidate_pool) -> predictions path.
    """
    modes = [evaluation_mode] if evaluation_mode else list(EVALUATION_MODES)
    pools = [candidate_pool] if candidate_pool else list(CANDIDATE_POOLS)

    found: dict[RunKey, Path] = {}
    for model_dir in sorted(Path(runs_dir).iterdir()):
        split_dir = model_dir / split
        if not split_dir.is_dir():
            continue
        for mode in modes:
            for pool in pools:
                # plain .json wins over .json.gz so an uncompressed working copy shadows the
                # committed archive rather than being silently ignored
                for suffix in (".json", ".json.gz"):
                    path = split_dir / f"{mode}_{pool}{suffix}"
                    if path.exists():
                        found[(model_dir.name, mode, pool)] = path
                        break

    return found


def summarise_runs(
    runs: dict[RunKey, Path],
    references: dict[str, list[str]],
    k_values: list[int],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    per_query_out: dict[RunKey, PerQuery] | None = None,
) -> list[dict]:
    """Score every run, reporting means with bootstrap confidence intervals.

    Intervals are per-system descriptive statistics and carry no multiple-comparison correction;
    corrected inference belongs to ``compare_against_baseline``.

    Args:
        runs: Output of ``discover_runs``.
        references: Maps query_id -> relevant candidate ids.
        k_values: Rank cutoffs to evaluate.
        n_resamples: Bootstrap resamples per interval.
        seed: Seed for the bootstrap.
        per_query_out: Optional dict populated with per-query values per run, for significance tests.

    Returns:
        One row per run with means and ``<metric>_ci_low`` / ``_ci_high`` / ``_std`` columns. The
        std is the standard deviation of the bootstrap distribution of the mean -- a compact
        alternative to the interval for the same quantity, not the per-query standard deviation.
    """
    rows: list[dict] = []

    for key, path in runs.items():
        system, mode, pool = key
        means, per_query = evaluate_run(load_run(path), references, k_values)
        if not per_query:
            continue
        if per_query_out is not None:
            per_query_out[key] = per_query

        row = {"system": system, "evaluation_mode": mode, "candidate_pool": pool, "n_queries": len(per_query)}
        for metric, mean in means.items():
            values = [values[metric] for values in per_query.values()]
            _, lo, hi = bootstrap_ci(values, n_resamples=n_resamples, seed=seed)
            row[metric] = mean
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
            row[f"{metric}_std"] = bootstrap_std(values, n_resamples=n_resamples, seed=seed)
        rows.append(row)

    return rows


def correct_family(rows: list[dict], alpha: float = 0.05) -> list[dict]:
    """Holm-correct a family of hypotheses tested independently of one another.

    ``compare_against_baseline`` corrects across the contenders passed to a single call, which is
    right when they share one baseline. It cannot correct across contrasts that each have their own
    baseline — such as many cascades, each tested against its own bare first stage — since those
    come from separate calls. Calling Holm on one hypothesis at a time is a no-op, so any family
    assembled from repeated single-hypothesis calls needs this correction applied afterwards, across
    the whole family at once.

    Args:
        rows: Dicts each carrying a "p_value" key; all other keys are passed through unchanged.
        alpha: Family-wise error rate.

    Returns:
        The same rows, each augmented with "p_value_corrected" and "significant".
    """
    if not rows:
        return []

    corrected, reject = holm_bonferroni([row["p_value"] for row in rows], alpha=alpha)
    return [
        {**row, "p_value_corrected": adjusted, "significant": significant}
        for row, adjusted, significant in zip(rows, corrected, reject)
    ]


def align_per_query(systems: dict[str, PerQuery], metric: str) -> tuple[list[str], dict[str, list[float]]]:
    """Restrict several systems to the queries they all cover.

    Paired significance testing and any delta between systems are only meaningful on a shared query
    population. Averaging over different denominators is what made the published enrichment
    comparisons invalid, so alignment is explicit and mismatches are visible to the caller.

    Args:
        systems: Maps system name -> per-query metric values.
        metric: Metric to extract.

    Returns:
        query_ids: The shared queries, sorted for a stable order.
        values: Maps system name -> metric values in ``query_ids`` order.

    Raises:
        ValueError: If the systems share no queries.
    """
    shared: set[str] | None = None
    for per_query in systems.values():
        keys = set(per_query)
        shared = keys if shared is None else shared & keys

    query_ids = sorted(shared or [])
    if not query_ids:
        raise ValueError(f"no shared queries across systems {sorted(systems)} for metric {metric}")

    values = {name: [per_query[qid][metric] for qid in query_ids] for name, per_query in systems.items()}
    return query_ids, values


def compare_against_baseline(
    systems: dict[str, PerQuery],
    baseline: str,
    metric: str,
    alpha: float = 0.05,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> list[dict]:
    """Paired-bootstrap every system against one baseline, Holm-corrected within the family.

    Args:
        systems: Maps system name -> per-query metric values; must include ``baseline``.
        baseline: Name of the reference system.
        metric: Metric to compare on.
        alpha: Family-wise error rate.
        n_resamples: Bootstrap resamples per contrast.
        seed: Seed for the bootstrap.

    Returns:
        One row per non-baseline system, carrying the delta, its confidence interval, the raw and
        Holm-corrected p-values, and the shared query count.
    """
    if baseline not in systems:
        raise ValueError(f"baseline {baseline!r} is not among the systems {sorted(systems)}")

    query_ids, values = align_per_query(systems, metric)
    contenders = [name for name in systems if name != baseline]

    tests = [
        paired_bootstrap_test(values[name], values[baseline], n_resamples=n_resamples, seed=seed)
        for name in contenders
    ]
    corrected, reject = holm_bonferroni([test["p_value"] for test in tests], alpha=alpha)

    return [
        {
            "system": name,
            "baseline": baseline,
            "metric": metric,
            "n_queries": len(query_ids),
            "delta": test["delta"],
            "ci_low": test["ci_low"],
            "ci_high": test["ci_high"],
            "p_value": test["p_value"],
            "p_value_corrected": adjusted,
            "significant": significant,
        }
        for name, test, adjusted, significant in zip(contenders, tests, corrected, reject)
    ]
