import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

METRIC_ORDER = ("hit", "recall", "ndcg", "mrr")


def sort_metric_names(names) -> list[str]:
    """Order metric columns as hit@k, recall@k, ndcg@k, then mrr variants, ascending in k."""

    def key(name: str) -> tuple[int, int, str]:
        family, _, cutoff = name.partition("@")
        rank = METRIC_ORDER.index(family) if family in METRIC_ORDER else len(METRIC_ORDER)
        return (rank, int(cutoff) if cutoff.isdigit() else 0, name)

    return sorted(names, key=key)


def append_row(results_file: Path, row: dict) -> Path:
    """Append a result row, widening the file when the row carries new columns.

    Appending in text mode would write values positionally under a stale header, silently shifting
    metrics into the wrong columns as soon as the metric set changes. Rewriting the file with the
    union of columns keeps older rows intact and leaves new columns empty for them.

    Args:
        results_file: Destination CSV.
        row: Column name -> value for a single run.

    Returns:
        The path written.
    """
    results_file.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])

    if results_file.exists():
        existing = pd.read_csv(results_file)
        columns = list(existing.columns) + [c for c in frame.columns if c not in existing.columns]
        frame = pd.concat([existing, frame], ignore_index=True).reindex(columns=columns)

    frame.to_csv(results_file, index=False)
    return results_file


def save_predictions_json(
    predictions: dict[str, dict[str, float]],
    results_dir: Path,
    evaluation_mode: str,
    candidate_pool: str,
) -> Path:
    """Save per-query predictions as {evaluation_mode}_{candidate_pool}.json, sorted by score desc."""
    predictions_json: dict[str, list] = {}
    for qid, cand_scores in predictions.items():
        sorted_candidates = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)
        predictions_json[qid] = [{"candidate_id": cid, "score": score} for cid, score in sorted_candidates]

    os.makedirs(results_dir, exist_ok=True)
    json_file = results_dir / f"{evaluation_mode}_{candidate_pool}.json"
    with open(json_file, "w") as f:
        json.dump(predictions_json, f, indent=4)
    return json_file


def save_references_json(references: dict[str, list[str]], results_dir: Path, evaluation_mode: str) -> Path:
    """Save the relevance judgements beside the predictions so analysis need not re-derive them."""
    os.makedirs(results_dir, exist_ok=True)
    json_file = results_dir / f"{evaluation_mode}_references.json"
    with open(json_file, "w") as f:
        json.dump(references, f, indent=4)
    return json_file


def save_eval_results(
    metrics: dict[str, float],
    results_file: Path,
    model_name: str,
    model_version: str,
    collaborator: str,
    evaluation_mode: str,
    candidate_pool: str,
    n_queries: int | None = None,
    seed: int | None = None,
    reference_date: str | None = None,
) -> Path:
    """Append a row of aggregate metrics to the shared results CSV.

    Every metric present in ``metrics`` is written, so adding a metric needs no change here.

    Args:
        metrics: Maps metric name -> aggregate value.
        results_file: Destination CSV.
        model_name: Model identifier.
        model_version: Version label for the run.
        collaborator: Preprocessing variant the run used.
        evaluation_mode: job_to_talent or talent_to_job.
        candidate_pool: observed or full.
        n_queries: Number of queries scored, needed to interpret small-sample results.
        seed: Seed recorded for reproducibility.
        reference_date: Publication recency reference date recorded for reproducibility.

    Returns:
        The path written.
    """
    row = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M"),
        "model": model_name,
        "model_version": model_version,
        "collaborator": collaborator,
        "evaluation_mode": evaluation_mode,
        "candidate_pool": candidate_pool,
        **{name: metrics[name] for name in sort_metric_names(metrics)},
    }
    provenance = {"n_queries": n_queries, "seed": seed, "reference_date": reference_date}
    row.update({key: value for key, value in provenance.items() if value is not None})

    return append_row(results_file, row)
