import math

import pytest

from tjm.metrics.fairness_ranking import (
    SKEW_FLOOR,
    amortized_attention,
    exposure_ratios,
    fair_rerank,
    ndkl,
    skew_at_k,
)


def test_skew_is_zero_when_the_top_k_matches_the_desired_proportion():
    ranked = ["f1", "m1", "f2", "m2"]
    group_of = {"f1": "F", "f2": "F", "m1": "M", "m2": "M"}
    assert skew_at_k(ranked, group_of, k=2)["F"] == pytest.approx(0.0)


def test_skew_is_positive_for_an_over_represented_group():
    """Pool is half F, top-2 is all F: skew is log(1.0 / 0.5)."""
    ranked = ["f1", "f2", "m1", "m2"]
    group_of = {"f1": "F", "f2": "F", "m1": "M", "m2": "M"}
    assert skew_at_k(ranked, group_of, k=2)["F"] == pytest.approx(math.log(2.0))


def test_skew_uses_an_explicit_desired_distribution_when_given():
    ranked = ["f1", "m1"]
    group_of = {"f1": "F", "m1": "M"}
    skews = skew_at_k(ranked, group_of, k=2, desired={"F": 0.25, "M": 0.75})
    assert skews["F"] == pytest.approx(math.log(0.5 / 0.25))


def test_skew_of_an_absent_group_is_floored_rather_than_negative_infinity():
    ranked = ["m1", "m2", "f1"]
    group_of = {"f1": "F", "m1": "M", "m2": "M"}
    assert math.isfinite(skew_at_k(ranked, group_of, k=2)["F"])


def test_ndkl_is_zero_only_when_every_prefix_matches_the_target():
    """A single-group pool is the one case where even the length-1 prefix is on target; with two
    groups the first prefix is unavoidably 100% one of them, so NDKL is strictly positive."""
    single = ["f1", "f2"]
    assert ndkl(single, {"f1": "F", "f2": "F"}) == pytest.approx(0.0)
    balanced = ["f1", "m1"]
    assert ndkl(balanced, {"f1": "F", "m1": "M"}) > 0.0


def test_ndkl_is_larger_for_a_more_segregated_ranking():
    group_of = {f"f{i}": "F" for i in range(1, 5)} | {f"m{i}": "M" for i in range(1, 5)}
    alternating = ["f1", "m1", "f2", "m2", "f3", "m3", "f4", "m4"]
    segregated = ["f1", "f2", "f3", "f4", "m1", "m2", "m3", "m4"]
    assert ndkl(segregated, group_of) > ndkl(alternating, group_of)


def test_ndkl_is_never_negative():
    group_of = {"f1": "F", "m1": "M", "m2": "M"}
    assert ndkl(["m1", "m2", "f1"], group_of) >= 0.0


def test_exposure_ratios_are_one_when_both_groups_are_equally_treated():
    """Symmetric ranking and symmetric relevance: neither group is favoured."""
    rankings = {"q1": ["f1", "m1"], "q2": ["m1", "f1"]}
    references = {"q1": ["f1", "m1"], "q2": ["f1", "m1"]}
    group_of = {"f1": "F", "m1": "M"}
    result = exposure_ratios(rankings, references, group_of, "F", "M")
    assert result["dtr"] == pytest.approx(1.0)
    assert result["dir"] == pytest.approx(1.0)


def test_exposure_ratios_exceed_one_when_the_first_group_gets_more_exposure_per_unit_relevance():
    rankings = {"q1": ["f1", "m1"]}
    references = {"q1": ["f1", "m1"]}
    group_of = {"f1": "F", "m1": "M"}
    assert exposure_ratios(rankings, references, group_of, "F", "M")["dtr"] > 1.0


def test_exposure_ratios_report_the_underlying_group_quantities():
    rankings = {"q1": ["f1", "m1"]}
    references = {"q1": ["f1"]}
    group_of = {"f1": "F", "m1": "M"}
    result = exposure_ratios(rankings, references, group_of, "F", "M")
    assert result["relevance_F"] == pytest.approx(1.0)
    assert result["relevance_M"] == pytest.approx(0.0)
    assert result["dtr"] is None  # undefined when a group has no relevant members


def test_amortized_attention_is_zero_when_attention_tracks_relevance():
    """Two equally relevant candidates that swap positions accrue equal attention."""
    rankings = {"q1": ["a", "b"], "q2": ["b", "a"]}
    references = {"q1": ["a", "b"], "q2": ["a", "b"]}
    group_of = {"a": "F", "b": "M"}
    result = amortized_attention(rankings, references, group_of)
    assert result["unfairness"] == pytest.approx(0.0, abs=1e-9)


def test_amortized_attention_detects_a_candidate_starved_of_attention():
    rankings = {"q1": ["a", "b"], "q2": ["a", "b"]}
    references = {"q1": ["a", "b"], "q2": ["a", "b"]}
    group_of = {"a": "F", "b": "M"}
    assert amortized_attention(rankings, references, group_of)["unfairness"] > 0.0


def test_amortized_attention_reports_each_group_attention_share_against_its_relevance_share():
    rankings = {"q1": ["a", "b"]}
    references = {"q1": ["a", "b"]}
    group_of = {"a": "F", "b": "M"}
    result = amortized_attention(rankings, references, group_of)
    assert result["attention_share_F"] > result["relevance_share_F"]


def test_fair_rerank_lifts_the_protected_group_to_the_required_proportion():
    """All protected candidates start below all others; the constraint must pull them up."""
    ranked = ["m1", "m2", "m3", "m4", "f1", "f2", "f3", "f4"]
    group_of = {f"m{i}": "M" for i in range(1, 5)} | {f"f{i}": "F" for i in range(1, 5)}
    reranked = fair_rerank(ranked, group_of, protected="F", p=0.5, k=4)
    assert sum(1 for c in reranked[:4] if group_of[c] == "F") >= 1


def test_fair_rerank_preserves_the_within_group_order():
    ranked = ["m1", "m2", "f1", "f2"]
    group_of = {"m1": "M", "m2": "M", "f1": "F", "f2": "F"}
    reranked = fair_rerank(ranked, group_of, protected="F", p=0.5, k=4)
    assert [c for c in reranked if c.startswith("f")] == ["f1", "f2"]
    assert [c for c in reranked if c.startswith("m")] == ["m1", "m2"]


def test_fair_rerank_returns_exactly_k_items():
    ranked = ["m1", "m2", "f1", "f2", "m3"]
    group_of = {"m1": "M", "m2": "M", "m3": "M", "f1": "F", "f2": "F"}
    assert len(fair_rerank(ranked, group_of, protected="F", p=0.4, k=3)) == 3


def test_fair_rerank_leaves_an_already_fair_ranking_untouched():
    ranked = ["f1", "m1", "f2", "m2"]
    group_of = {"f1": "F", "f2": "F", "m1": "M", "m2": "M"}
    assert fair_rerank(ranked, group_of, protected="F", p=0.5, k=4) == ranked


def test_exposure_ratios_can_group_by_the_querying_entity():
    """The attribute may sit on the query rather than on the ranked items."""
    rankings = {"tF": ["v1", "v2"], "tM": ["v2", "v1"]}
    references = {"tF": ["v1"], "tM": ["v1"]}
    group_of = {"tF": "F", "tM": "M"}
    result = exposure_ratios(rankings, references, group_of, "F", "M", group_side="query")
    assert result["relevance_F"] == pytest.approx(0.5)
    assert result["dir"] > 1.0  # tF sees its relevant vacancy first, tM sees it second


def test_exposure_ratios_reject_an_unknown_group_side():
    with pytest.raises(ValueError):
        exposure_ratios({"q": ["a"]}, {"q": ["a"]}, {"a": "F"}, "F", "M", group_side="nonsense")


def test_skew_floors_a_group_absent_from_a_list_shorter_than_k():
    """A top-k shorter than k forces a group to be absent, which is geometry, not unfairness: the
    caller must therefore be able to spot such lists, so the floor is reported rather than an error."""
    group_of = {"f1": "F", "m1": "M"}
    corpus_share = {"F": 0.5, "M": 0.5}
    assert skew_at_k(["f1"], group_of, k=5, desired=corpus_share)["M"] == pytest.approx(SKEW_FLOOR)
