import gzip
import json
from pathlib import Path

from .ranking import rank_candidates

Run = dict[str, dict[str, float]]


def load_run(path: str | Path) -> Run:
    """Load a saved predictions JSON into query_id -> {candidate_id: score}.

    Args:
        path: Path to a predictions file written by ``save_predictions_json``, either plain
            ``.json`` or gzipped ``.json.gz``.

    Returns:
        Maps query_id -> {candidate_id: score}.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        payload = json.load(handle)
    return {
        qid: {entry["candidate_id"]: float(entry["score"]) for entry in entries} for qid, entries in payload.items()
    }


def cascade_rankings(first_stage: Run, reranker: Run, depth: int) -> Run:
    """Reconstruct a retrieve-then-rerank cascade from two exhaustive runs.

    A cross-encoder's score for a (query, document) pair does not depend on the rest of the candidate
    set, so re-sorting the first stage's top-``depth`` shortlist by the reranker's saved scores
    reproduces the cascade exactly rather than approximating it.

    Args:
        first_stage: Retriever run supplying the shortlist.
        reranker: Reranker run supplying the final scores; must cover every shortlisted pair.
        depth: Shortlist size handed to the reranker.

    Returns:
        Maps query_id -> {candidate_id: reranker score} restricted to the shortlist.

    Raises:
        ValueError: If the reranker run does not cover a query or a shortlisted candidate, since
            scoring partial coverage would silently understate the cascade.
    """
    cascaded: Run = {}

    for query_id, scores in first_stage.items():
        if query_id not in reranker:
            raise ValueError(f"reranker run is missing query {query_id}")

        shortlist = rank_candidates(scores)[:depth]
        reranker_scores = reranker[query_id]
        missing = [cid for cid in shortlist if cid not in reranker_scores]
        if missing:
            raise ValueError(
                f"reranker run is missing {len(missing)} shortlisted candidates for query {query_id}: "
                f"{', '.join(missing[:5])}"
            )

        cascaded[query_id] = {cid: reranker_scores[cid] for cid in shortlist}

    return cascaded


def first_stage_recall(first_stage: Run, references: dict[str, list[str]], depth: int) -> dict[str, float]:
    """Per-query recall of the first stage at the shortlist depth.

    This is the ceiling no reranker operating on that shortlist can exceed, which is what makes a
    depth sweep interpretable.

    Args:
        first_stage: Retriever run supplying the shortlist.
        references: Maps query_id -> relevant candidate ids.
        depth: Shortlist size.

    Returns:
        Maps query_id -> recall@depth, omitting queries with no relevant candidates.
    """
    recalls: dict[str, float] = {}

    for query_id, relevants in references.items():
        if not relevants:
            continue
        shortlist = set(rank_candidates(first_stage.get(query_id, {}))[:depth])
        relevant_set = set(relevants)
        recalls[query_id] = len(relevant_set & shortlist) / len(relevant_set)

    return recalls
