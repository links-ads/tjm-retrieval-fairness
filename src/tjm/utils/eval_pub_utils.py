import json
from datetime import datetime
from pathlib import Path

from .eval_utils import append_row, sort_metric_names


def _normalize_publication_sequence(sequence_value) -> str:
    if sequence_value is None:
        return "none"
    if isinstance(sequence_value, (list, tuple, set)):
        normalized = sorted(str(v) for v in sequence_value)
        return "+".join(normalized) if normalized else "none"
    return str(sequence_value)


def _publication_config_key(publication_config: dict | None) -> str:
    if not publication_config:
        return "pub_default"

    include_publications = publication_config.get("include_publications", False)
    publication_sequence = _normalize_publication_sequence(
        publication_config.get("publication_sequence")
    )
    publication_selection_mode = publication_config.get("publication_selection_mode", "random")
    publication_text_fields = publication_config.get("publication_text_fields", "both")
    max_publications = publication_config.get("max_publications", 5)

    return (
        f"pub-{int(bool(include_publications))}"
        f"_seq-{publication_sequence}"
        f"_sel-{publication_selection_mode}"
        f"_txt-{publication_text_fields}"
        f"_max-{max_publications}"
    )

def save_predictions_pub_json(
    predictions: dict[str, dict[str, float]],
    results_dir: Path,
    evaluation_mode: str,
    candidate_pool: str,
    publication_config: dict | None = None,
) -> Path:
    """Save per-query predictions for publication-enriched evaluations.

    Uses a dedicated filename suffix to avoid collisions with baseline outputs.
    """
    predictions_json: dict[str, list] = {}
    for qid, cand_scores in predictions.items():
        sorted_candidates = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)
        predictions_json[qid] = [{"candidate_id": cid, "score": score} for cid, score in sorted_candidates]

    results_dir.mkdir(parents=True, exist_ok=True)
    config_key = _publication_config_key(publication_config)
    json_file = results_dir / (
        f"{evaluation_mode}_{candidate_pool}_with_publications_{config_key}.json"
    )
    with open(json_file, "w") as f:
        json.dump(predictions_json, f, indent=4)
    return json_file


def save_eval_pub_results(
    metrics: dict[str, float],
    results_file: Path,
    model_name: str,
    model_version: str,
    collaborator: str,
    evaluation_mode: str,
    candidate_pool: str,
    publication_config: dict | None = None,
    n_queries: int | None = None,
    seed: int | None = None,
    reference_date: str | None = None,
) -> Path:
    """Append a row of publication-enrichment metrics.

    Args:
        metrics: Maps metric name -> aggregate value; all are written.
        results_file: Destination CSV.
        model_name: Ranking model identifier.
        model_version: Version label for the run.
        collaborator: Preprocessing variant the run used.
        evaluation_mode: job_to_talent or talent_to_job.
        candidate_pool: observed or full.
        publication_config: Enrichment settings, flattened into descriptive columns.
        n_queries: Number of queries scored. Essential here, since the enrichment study runs on tens
            of queries and comparisons are only valid on a shared query set.
        seed: Seed recorded for reproducibility.
        reference_date: Publication recency reference date recorded for reproducibility.

    Returns:
        The path written.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    config_key = _publication_config_key(publication_config)

    include_publications = None
    publication_sequence = None
    publication_selection_mode = None
    publication_text_fields = None
    max_publications = None
    if publication_config:
        include_publications = publication_config.get("include_publications")
        publication_sequence = _normalize_publication_sequence(
            publication_config.get("publication_sequence")
        )
        publication_selection_mode = publication_config.get("publication_selection_mode")
        publication_text_fields = publication_config.get("publication_text_fields")
        max_publications = publication_config.get("max_publications")

    row = {
        "timestamp": timestamp,
        "model": model_name,
        "model_version": model_version,
        "collaborator": collaborator,
        "evaluation_mode": evaluation_mode,
        "candidate_pool": candidate_pool,
        "use_publications": True,
        "publication_config_key": config_key,
        "include_publications": include_publications,
        "publication_sequence": publication_sequence,
        "publication_selection_mode": publication_selection_mode,
        "publication_text_fields": publication_text_fields,
        "max_publications": max_publications,
        **{name: metrics[name] for name in sort_metric_names(metrics)},
    }
    provenance = {"n_queries": n_queries, "seed": seed, "reference_date": reference_date}
    row.update({key: value for key, value in provenance.items() if value is not None})

    return append_row(results_file, row)