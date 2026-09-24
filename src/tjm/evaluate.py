import argparse
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from tjm.data.dataloader import InnoTripletForValidationDataset
from tjm.metrics.ranking import evaluate_run
from tjm.models.load_model import load_model
from tjm.utils import save_eval_results, save_predictions_json, save_references_json


def build_ir_payload(
    dataset,
    model,
    candidates_pool: dict,
    candidate_pool: str,
    show_progress: bool = False,
    batch_size: int = 8,
) -> tuple[dict[str, dict[str, float]], dict[str, list[str]]]:
    """Build predictions and references for IR evaluation.

    Args:
            dataset: HF Dataset with columns: query_id, query, candidate_id, candidate, label.
            model: A model with a score(query, candidates) method.
            candidates_pool: Maps candidate_id -> candidate data for "full" pool mode.
            candidate_pool: "observed" (only pairs in dataset) or "full" (all candidates).
            show_progress: Whether to display a progress bar.
            batch_size: Number of candidates to score in a batch.

    Returns:
            predictions: Dict mapping query_id -> {candidate_id: score}.
            references: Dict mapping query_id -> [positive_candidate_ids].
    """
    # Lexical scorers need collection-level term statistics, which a per-batch call cannot supply.
    if hasattr(model, "index_corpus"):
        model.index_corpus(candidates_pool)

    grouped: dict[str, list] = defaultdict(list)
    for sample in dataset:
        grouped[str(sample["query_id"])].append(sample)

    predictions: dict[str, dict[str, float]] = {}
    references: dict[str, list[str]] = {}

    items = list(grouped.items())
    iterator = tqdm(items, desc="Scoring queries", total=len(items), disable=not show_progress)

    for query_id, samples in iterator:
        query = samples[0]["query"]
        positive_ids = {str(sample["candidate_id"]) for sample in samples if int(sample["label"]) == 1}

        if not positive_ids:
            continue

        if candidate_pool == "full":
            candidate_ids = list(candidates_pool.keys())
            candidates = [candidates_pool[cid] for cid in candidate_ids]
        else:
            candidate_ids = [str(sample["candidate_id"]) for sample in samples]
            candidates = [sample["candidate"] for sample in samples]

        if not candidates:
            continue

        scores = []
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i : i + batch_size]
            batch_scores = model.score(query, batch)
            scores.extend(batch_scores)

        predictions[query_id] = {cid: float(score) for cid, score in zip(candidate_ids, scores)}
        references[query_id] = list(positive_ids)

    return predictions, references


def parse_k_list(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x]


def compute_recall_at_k(
    predictions: dict[str, dict[str, float]],
    references: dict[str, list[str]],
    k_values: list[int],
) -> dict[str, float]:
    """Aggregate metrics for a run.

    Retained under its original name so existing callers keep working; the metric set now also
    includes hit@k, ndcg@k and MRR. Use ``metrics.ranking.evaluate_run`` directly when the per-query
    values are needed, for instance to run significance tests.
    """
    means, _ = evaluate_run(predictions, references, k_values)
    return means


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate IR models on talent-job triplets")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing data",
    )
    parser.add_argument(
        "--dataloader",
        type=str,
        default="text",
        help="Which dataloader to use (registered in data/dataloaders/factory.py)",
    )
    parser.add_argument(
        "--preprocessing",
        type=str,
        default="base",
        help="Preprocessing variant label recorded alongside the run",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test", "triplets"],
        help="Which split to evaluate on",
    )
    parser.add_argument(
        "--evaluation-mode",
        type=str,
        default="job_to_talent",
        choices=["job_to_talent", "talent_to_job"],
        help="job_to_talent ranks resumes per job; talent_to_job ranks jobs per resume",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for model scoring",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mpnet",
        help="Which model implementation to run (registered in models/factory.py)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Optional HF hub path or local checkpoint to override the default model weights",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override (cpu|cuda); defaults to cuda if available else cpu",
    )
    parser.add_argument(
        "--k",
        type=str,
        default="1,5,10,20,50,100",
        help="Comma-separated cutoff values for recall@k",
    )
    parser.add_argument(
        "--candidate-pool",
        type=str,
        default="observed",
        choices=["observed", "full"],
        help="observed=only pairs from triplets; full=score against every candidate",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show a progress bar over queries",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        required=True,
        help="Version identifier for the model (e.g. v1.0, v2.3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
    )

    args = parser.parse_args()
    k_values = parse_k_list(args.k)

    import torch

    resolved_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = InnoTripletForValidationDataset(
        data_dir=args.data_dir,
        split=args.split,
        evaluation_mode=args.evaluation_mode,
    )
    dataset, candidates_pool = dataloader.load()

    model = load_model(
        args.model,
        device=resolved_device,
        pretrained=args.pretrained,
        evaluation_mode=args.evaluation_mode
    )

    print(f"Model: {model.model_name}, Device: {resolved_device}")

    predictions, references = build_ir_payload(
        dataset,
        model,
        candidates_pool=candidates_pool,
        candidate_pool=args.candidate_pool,
        show_progress=args.progress,
        batch_size=args.batch_size,
    )
    if not predictions:
        raise RuntimeError("No queries with positive labels found for evaluation.")

    metrics, per_query = evaluate_run(predictions, references, k_values)

    print(f"Queries evaluated: {len(per_query)}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    safe_model_name = args.model.replace("/", "_")
    results_dir = Path(args.output_dir) / safe_model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)

    json_file = save_predictions_json(
        predictions,
        results_dir / f"{args.split}",
        args.evaluation_mode,
        args.candidate_pool,
    )
    save_references_json(references, results_dir / f"{args.split}", args.evaluation_mode)
    results_csv = save_eval_results(
        metrics,
        output_dir / f"{args.split}_eval_results.csv",
        safe_model_name,
        args.model_version,
        args.preprocessing,
        args.evaluation_mode,
        args.candidate_pool,
        n_queries=len(per_query),
    )

    print(f"\nResults saved to:")
    print(f"  - {json_file}")
    print(f"  - {results_csv}")
