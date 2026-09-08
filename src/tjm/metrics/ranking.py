from math import log2

MRR_CUTOFF = 10


def rank_candidates(scores: dict[str, float]) -> list[str]:
    """Order candidate ids by relevance score.

    Ties are broken by ascending candidate id so that rankings are reproducible regardless of
    insertion order. This matters for lexical scorers such as BM25, which assign identical zero
    scores to most of the pool.

    Args:
        scores: Maps candidate_id -> relevance score.

    Returns:
        Candidate ids ordered by descending score, then ascending id.
    """
    return [cid for cid, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def _dcg(ranked: list[str], relevant: set[str], k: int) -> float:
    return sum(1.0 / log2(rank + 1) for rank, cid in enumerate(ranked[:k], start=1) if cid in relevant)


def _ideal_dcg(n_relevant: int, k: int) -> float:
    return sum(1.0 / log2(rank + 1) for rank in range(1, min(n_relevant, k) + 1))


def _reciprocal_rank(ranked: list[str], relevant: set[str], cutoff: int | None = None) -> float:
    limit = len(ranked) if cutoff is None else cutoff
    for rank, cid in enumerate(ranked[:limit], start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def per_query_metrics(ranked: list[str], relevant: set[str], k_values: list[int]) -> dict[str, float]:
    """Compute ranking metrics for a single query.

    Args:
        ranked: Candidate ids ordered best-first.
        relevant: Ids of the relevant candidates; must be non-empty.
        k_values: Rank cutoffs to evaluate.

    Returns:
        Maps metric name -> value. Includes hit@k, recall@k, ndcg@k for every k, plus mrr over the
        full ranking and mrr@10.
    """
    metrics: dict[str, float] = {}
    n_relevant = len(relevant)

    for k in k_values:
        hits = len(relevant.intersection(ranked[:k]))
        ideal = _ideal_dcg(n_relevant, k)
        metrics[f"hit@{k}"] = 1.0 if hits else 0.0
        metrics[f"recall@{k}"] = hits / n_relevant
        metrics[f"ndcg@{k}"] = _dcg(ranked, relevant, k) / ideal if ideal else 0.0

    metrics["mrr"] = _reciprocal_rank(ranked, relevant)
    metrics[f"mrr@{MRR_CUTOFF}"] = _reciprocal_rank(ranked, relevant, MRR_CUTOFF)
    return metrics


def evaluate_run(
    predictions: dict[str, dict[str, float]],
    references: dict[str, list[str]],
    k_values: list[int],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Score a run, returning both aggregates and the per-query values behind them.

    Queries with at least one relevant candidate are always counted. A query absent from
    ``predictions`` scores zero rather than being dropped, so a system cannot raise its average by
    failing to answer.

    Args:
        predictions: Maps query_id -> {candidate_id: score}.
        references: Maps query_id -> relevant candidate ids.
        k_values: Rank cutoffs to evaluate.

    Returns:
        means: Maps metric name -> mean over evaluated queries.
        per_query: Maps query_id -> {metric name: value}, the input required for paired significance
            testing.
    """
    per_query: dict[str, dict[str, float]] = {}

    for query_id, relevants in references.items():
        if not relevants:
            continue
        ranked = rank_candidates(predictions.get(query_id, {}))
        per_query[query_id] = per_query_metrics(ranked, set(relevants), k_values)

    if not per_query:
        return {}, {}

    names = next(iter(per_query.values())).keys()
    means = {name: sum(values[name] for values in per_query.values()) / len(per_query) for name in names}
    return means, per_query
