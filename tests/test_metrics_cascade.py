import json

import pytest

from tjm.metrics.cascade import cascade_rankings, first_stage_recall, load_run


def test_load_run_reads_the_saved_prediction_format(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(
        json.dumps({"q1": [{"candidate_id": "a", "score": 0.9}, {"candidate_id": "b", "score": 0.2}]})
    )
    assert load_run(path) == {"q1": {"a": 0.9, "b": 0.2}}


def test_cascade_keeps_only_the_first_stage_top_n():
    first = {"q1": {"a": 0.9, "b": 0.8, "c": 0.1}}
    reranker = {"q1": {"a": 0.1, "b": 0.2, "c": 0.99}}
    assert set(cascade_rankings(first, reranker, depth=2)["q1"]) == {"a", "b"}


def test_cascade_reorders_the_shortlist_by_reranker_score():
    first = {"q1": {"a": 0.9, "b": 0.8}}
    reranker = {"q1": {"a": 0.1, "b": 0.7}}
    assert cascade_rankings(first, reranker, depth=2)["q1"] == {"a": 0.1, "b": 0.7}


def test_cascade_cannot_recover_a_candidate_the_first_stage_missed():
    first = {"q1": {"a": 0.9, "b": 0.8, "gold": 0.01}}
    reranker = {"q1": {"a": 0.1, "b": 0.2, "gold": 0.99}}
    assert "gold" not in cascade_rankings(first, reranker, depth=2)["q1"]


def test_cascade_depth_beyond_pool_size_keeps_everything():
    first = {"q1": {"a": 0.9, "b": 0.8}}
    reranker = {"q1": {"a": 0.1, "b": 0.7}}
    assert set(cascade_rankings(first, reranker, depth=100)["q1"]) == {"a", "b"}


def test_cascade_rejects_a_reranker_missing_shortlisted_candidates():
    first = {"q1": {"a": 0.9, "b": 0.8}}
    reranker = {"q1": {"a": 0.1}}
    with pytest.raises(ValueError, match="b"):
        cascade_rankings(first, reranker, depth=2)


def test_cascade_rejects_a_reranker_missing_a_query():
    first = {"q1": {"a": 0.9}, "q2": {"a": 0.5}}
    reranker = {"q1": {"a": 0.1}}
    with pytest.raises(ValueError, match="q2"):
        cascade_rankings(first, reranker, depth=1)


def test_first_stage_recall_is_the_ceiling_at_that_depth():
    first = {"q1": {"a": 0.9, "b": 0.8, "gold": 0.01}}
    references = {"q1": ["gold"]}
    assert first_stage_recall(first, references, depth=2) == {"q1": 0.0}
    assert first_stage_recall(first, references, depth=3) == {"q1": 1.0}


def test_first_stage_recall_is_fractional_with_multiple_relevants():
    first = {"q1": {"g1": 0.9, "x": 0.5, "g2": 0.1}}
    references = {"q1": ["g1", "g2"]}
    assert first_stage_recall(first, references, depth=2) == pytest.approx({"q1": 0.5})


def test_first_stage_recall_skips_queries_without_relevants():
    first = {"q1": {"a": 0.9}, "q2": {"a": 0.9}}
    references = {"q1": ["a"], "q2": []}
    assert first_stage_recall(first, references, depth=1) == {"q1": 1.0}
