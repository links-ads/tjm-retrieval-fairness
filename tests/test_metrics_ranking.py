import pytest

from tjm.metrics.ranking import evaluate_run, per_query_metrics, rank_candidates


def test_ranks_by_descending_score():
    assert rank_candidates({"a": 0.1, "b": 0.9, "c": 0.5}) == ["b", "c", "a"]


def test_breaks_score_ties_by_ascending_candidate_id():
    assert rank_candidates({"b": 1.0, "a": 1.0, "c": 1.0}) == ["a", "b", "c"]


def test_hit_at_k_is_one_when_any_relevant_in_top_k():
    m = per_query_metrics(["x", "y", "rel"], {"rel"}, [1, 3])
    assert m["hit@1"] == 0.0
    assert m["hit@3"] == 1.0


def test_recall_at_k_is_fraction_of_relevants_retrieved():
    m = per_query_metrics(["a", "b", "x", "c"], {"a", "b", "c", "d"}, [2, 4])
    assert m["recall@2"] == pytest.approx(0.5)
    assert m["recall@4"] == pytest.approx(0.75)


def test_hit_and_recall_differ_with_multiple_positives():
    m = per_query_metrics(["a", "x", "y"], {"a", "b"}, [3])
    assert m["hit@3"] == 1.0
    assert m["recall@3"] == pytest.approx(0.5)


def test_ndcg_is_one_when_all_relevants_rank_first():
    m = per_query_metrics(["a", "b", "x", "y"], {"a", "b"}, [4])
    assert m["ndcg@4"] == pytest.approx(1.0)


def test_ndcg_matches_hand_computed_value():
    # single relevant at rank 2: DCG = 1/log2(3), IDCG = 1/log2(2) = 1
    import math

    m = per_query_metrics(["x", "rel", "y"], {"rel"}, [3])
    assert m["ndcg@3"] == pytest.approx(1.0 / math.log2(3))


def test_ndcg_idcg_caps_at_k_when_relevants_exceed_k():
    # 3 relevants but k=2: IDCG = 1/log2(2) + 1/log2(3); ranking puts 2 relevants first
    import math

    m = per_query_metrics(["a", "b", "c"], {"a", "b", "c"}, [2])
    ideal = 1.0 + 1.0 / math.log2(3)
    assert m["ndcg@2"] == pytest.approx((1.0 + 1.0 / math.log2(3)) / ideal)
    assert m["ndcg@2"] == pytest.approx(1.0)


def test_mrr_is_reciprocal_of_first_relevant_rank():
    assert per_query_metrics(["x", "y", "rel"], {"rel"}, [1])["mrr"] == pytest.approx(1.0 / 3.0)


def test_mrr_is_zero_when_no_relevant_retrieved():
    assert per_query_metrics(["x", "y"], {"rel"}, [1])["mrr"] == 0.0


def test_mrr_at_10_ignores_relevants_beyond_rank_10():
    ranked = [f"x{i}" for i in range(10)] + ["rel"]
    m = per_query_metrics(ranked, {"rel"}, [1])
    assert m["mrr"] == pytest.approx(1.0 / 11.0)
    assert m["mrr@10"] == 0.0


def test_evaluate_run_returns_means_and_per_query_values():
    predictions = {"q1": {"a": 0.9, "b": 0.1}, "q2": {"a": 0.1, "b": 0.9}}
    references = {"q1": ["a"], "q2": ["a"]}
    means, per_query = evaluate_run(predictions, references, [1])
    assert per_query["q1"]["hit@1"] == 1.0
    assert per_query["q2"]["hit@1"] == 0.0
    assert means["hit@1"] == pytest.approx(0.5)


def test_query_missing_from_predictions_scores_zero_and_stays_in_denominator():
    predictions = {"q1": {"a": 0.9}}
    references = {"q1": ["a"], "q2": ["a"]}
    means, per_query = evaluate_run(predictions, references, [1])
    assert per_query["q2"]["hit@1"] == 0.0
    assert means["hit@1"] == pytest.approx(0.5)


def test_queries_without_relevants_are_excluded():
    predictions = {"q1": {"a": 0.9}, "q2": {"a": 0.9}}
    references = {"q1": ["a"], "q2": []}
    means, per_query = evaluate_run(predictions, references, [1])
    assert "q2" not in per_query
    assert means["hit@1"] == pytest.approx(1.0)
