import pytest

from tjm.metrics.fairness import (
    binary_category_split,
    median_split,
    classification_rates,
    individual_classification_rates,
    individual_rate_values,
    classification_rates_by_query_group,
    representation_gap,
    permutation_group_test,
    relevant_candidate_outcomes,
    summarise_groups,
    top_k_group_shares,
)


def test_summarise_groups_reports_size_and_mean_per_group():
    values = {"Female": [1.0, 0.0, 1.0, 1.0], "Male": [0.0, 0.0, 1.0, 1.0]}
    rows = {row["group"]: row for row in summarise_groups(values, n_resamples=200, seed=0)}
    assert rows["Female"]["n"] == 4
    assert rows["Female"]["mean"] == pytest.approx(0.75)
    assert rows["Male"]["mean"] == pytest.approx(0.5)


def test_summarise_groups_brackets_the_mean_with_a_confidence_interval():
    values = {"Female": [0.2, 0.4, 0.6, 0.8, 1.0]}
    row = summarise_groups(values, n_resamples=500, seed=0)[0]
    assert row["ci_low"] <= row["mean"] <= row["ci_high"]


def test_summarise_groups_skips_groups_with_no_observations():
    values = {"Female": [1.0], "Non-binary": []}
    groups = [row["group"] for row in summarise_groups(values, n_resamples=100, seed=0)]
    assert groups == ["Female"]


def test_permutation_group_test_reports_the_observed_gap_between_group_means():
    result = permutation_group_test([1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], n_resamples=200, seed=0)
    assert result["gap"] == pytest.approx(1.0)
    assert result["n_a"] == 4
    assert result["n_b"] == 4


def test_permutation_group_test_finds_a_clean_separation_unlikely_under_the_null():
    """Two perfectly separated groups must not look like a chance arrangement of the labels."""
    a = [1.0] * 12
    b = [0.0] * 12
    assert permutation_group_test(a, b, n_resamples=2000, seed=0)["p_value"] < 0.01


def test_permutation_group_test_finds_identical_groups_entirely_unremarkable():
    a = [0.3, 0.6, 0.9, 0.3, 0.6, 0.9]
    b = [0.3, 0.6, 0.9, 0.3, 0.6, 0.9]
    assert permutation_group_test(a, b, n_resamples=2000, seed=0)["p_value"] > 0.5


def test_permutation_group_test_is_reproducible_for_a_fixed_seed():
    a, b = [1.0, 0.0, 1.0, 0.5], [0.0, 1.0, 0.0, 0.5]
    first = permutation_group_test(a, b, n_resamples=500, seed=7)
    second = permutation_group_test(a, b, n_resamples=500, seed=7)
    assert first == second


def test_permutation_group_test_rejects_an_empty_group():
    with pytest.raises(ValueError):
        permutation_group_test([1.0], [], n_resamples=10, seed=0)


def test_relevant_candidate_outcomes_records_rank_and_retrieval_per_relevant_pair():
    rankings = {"q1": ["c3", "c1", "c2"]}
    references = {"q1": ["c1", "c2"]}
    outcomes = {o["candidate_id"]: o for o in relevant_candidate_outcomes(rankings, references, k=2)}
    assert outcomes["c1"]["rank"] == 2
    assert outcomes["c1"]["retrieved"] == 1.0
    assert outcomes["c1"]["reciprocal_rank"] == pytest.approx(0.5)
    assert outcomes["c2"]["rank"] == 3
    assert outcomes["c2"]["retrieved"] == 0.0


def test_relevant_candidate_outcomes_ignores_relevant_ids_absent_from_the_ranking():
    """A positive that never entered the candidate pool is not an outcome the ranker produced."""
    rankings = {"q1": ["c1"]}
    references = {"q1": ["c1", "missing"]}
    outcomes = relevant_candidate_outcomes(rankings, references, k=1)
    assert [o["candidate_id"] for o in outcomes] == ["c1"]


def test_relevant_candidate_outcomes_skips_queries_with_no_relevance_judgements():
    rankings = {"q1": ["c1"], "q2": ["c2"]}
    references = {"q1": ["c1"]}
    assert {o["query_id"] for o in relevant_candidate_outcomes(rankings, references, k=1)} == {"q1"}


def test_top_k_group_shares_gives_each_group_its_fraction_of_the_top_k():
    rankings = {"q1": ["a1", "a2", "b1", "b2"]}
    group_of = {"a1": "Female", "a2": "Female", "b1": "Male", "b2": "Male"}
    shares = top_k_group_shares(rankings, group_of, k=2)
    assert shares["Female"] == [pytest.approx(1.0)]
    assert shares["Male"] == [pytest.approx(0.0)]


def test_top_k_group_shares_normalises_by_labelled_candidates_not_slot_count():
    """Unlabelled candidates must not silently deflate every group's share."""
    rankings = {"q1": ["a1", "unknown", "b1", "b2"]}
    group_of = {"a1": "Female", "b1": "Male", "b2": "Male"}
    shares = top_k_group_shares(rankings, group_of, k=3)
    assert shares["Female"] == [pytest.approx(0.5)]
    assert shares["Male"] == [pytest.approx(0.5)]


def test_top_k_group_shares_skips_queries_whose_top_k_has_no_labelled_candidate():
    rankings = {"q1": ["unknown"], "q2": ["a1"]}
    group_of = {"a1": "Female"}
    shares = top_k_group_shares(rankings, group_of, k=1)
    assert shares["Female"] == [pytest.approx(1.0)]


def test_representation_gap_is_zero_when_the_top_k_mirrors_the_pool():
    rankings = {"q1": ["f1", "m1", "f2", "m2"]}
    group_of = {"f1": "Female", "f2": "Female", "m1": "Male", "m2": "Male"}
    gaps = representation_gap(rankings, group_of, k=2)
    assert gaps["Female"] == [pytest.approx(0.0)]
    assert gaps["Male"] == [pytest.approx(0.0)]


def test_representation_gap_is_positive_for_a_group_over_represented_in_the_top_k():
    """Half the pool is Female but the whole top-2 is: the shortlist over-represents them by 50 points."""
    rankings = {"q1": ["f1", "f2", "m1", "m2"]}
    group_of = {"f1": "Female", "f2": "Female", "m1": "Male", "m2": "Male"}
    gaps = representation_gap(rankings, group_of, k=2)
    assert gaps["Female"] == [pytest.approx(0.5)]
    assert gaps["Male"] == [pytest.approx(-0.5)]


def test_representation_gap_pairs_top_k_and_pool_within_the_same_query():
    """A query contributing no labelled candidate to the top-k must drop out of both sides at once."""
    rankings = {"q1": ["unknown", "unknown2", "f1"], "q2": ["f1", "m1"]}
    group_of = {"f1": "Female", "m1": "Male"}
    gaps = representation_gap(rankings, group_of, k=2)
    assert len(gaps["Female"]) == 1
    assert len(gaps["Male"]) == 1
    assert gaps["Female"] == [pytest.approx(0.0)]


def _toy():
    rankings = {"q1": ["a", "b", "c", "d"]}
    references = {"q1": ["a", "c"]}
    group_of = {"a": "Female", "b": "Female", "c": "Male", "d": "Male"}
    return rankings, references, group_of


def test_classification_rates_reports_the_positive_prediction_rate_per_group():
    """Only 'a' is predicted positive at k=1, and it is one of the two Female candidates."""
    rates = classification_rates(*_toy(), k=1)
    assert rates["Female"]["positive_rate"] == pytest.approx(0.5)
    assert rates["Male"]["positive_rate"] == pytest.approx(0.0)


def test_classification_rates_reports_the_true_positive_rate_per_group():
    rates = classification_rates(*_toy(), k=1)
    assert rates["Female"]["true_positive_rate"] == pytest.approx(1.0)
    assert rates["Male"]["true_positive_rate"] == pytest.approx(0.0)


def test_classification_rates_reports_accuracy_over_all_candidates_of_the_group():
    """Female: both decisions correct. Male: 'c' is relevant but unretrieved, 'd' correctly not."""
    rates = classification_rates(*_toy(), k=1)
    assert rates["Female"]["accuracy"] == pytest.approx(1.0)
    assert rates["Male"]["accuracy"] == pytest.approx(0.5)


def test_classification_rates_counts_the_candidates_behind_each_group():
    rates = classification_rates(*_toy(), k=1)
    assert rates["Female"]["n_candidates"] == 2
    assert rates["Female"]["n_relevant"] == 1


def test_classification_rates_leaves_the_true_positive_rate_undefined_without_relevant_members():
    rankings = {"q1": ["a", "b"]}
    references = {"q1": ["a"]}
    group_of = {"a": "Female", "b": "Male"}
    rates = classification_rates(rankings, references, group_of, k=1)
    assert rates["Male"]["n_relevant"] == 0
    assert rates["Male"]["true_positive_rate"] is None


def test_classification_rates_ignores_queries_without_relevance_judgements():
    rankings = {"q1": ["a", "b"], "q2": ["a", "b"]}
    references = {"q1": ["a"]}
    group_of = {"a": "Female", "b": "Male"}
    rates = classification_rates(rankings, references, group_of, k=1)
    assert rates["Female"]["n_candidates"] == 1


def test_query_grouped_rates_attribute_every_pair_to_the_querying_group():
    """Both candidates belong to q1's group, whatever the candidates themselves are."""
    rankings = {"q1": ["a", "b"]}
    references = {"q1": ["a"]}
    rates = classification_rates_by_query_group(rankings, references, {"q1": "Female"}, k=1)
    assert rates["Female"]["n_candidates"] == 2
    assert rates["Female"]["positive_rate"] == pytest.approx(0.5)
    assert rates["Female"]["true_positive_rate"] == pytest.approx(1.0)


def test_query_grouped_rates_separate_two_queriers_of_different_groups():
    rankings = {"q1": ["a", "b"], "q2": ["a", "b"]}
    references = {"q1": ["a"], "q2": ["b"]}
    group_of = {"q1": "Female", "q2": "Male"}
    rates = classification_rates_by_query_group(rankings, references, group_of, k=1)
    assert rates["Female"]["true_positive_rate"] == pytest.approx(1.0)
    assert rates["Male"]["true_positive_rate"] == pytest.approx(0.0)


def test_query_grouped_rates_skip_queries_whose_group_is_unknown():
    rankings = {"q1": ["a"], "q2": ["a"]}
    references = {"q1": ["a"], "q2": ["a"]}
    rates = classification_rates_by_query_group(rankings, references, {"q1": "Female"}, k=1)
    assert rates["Female"]["n_candidates"] == 1


def test_individual_rates_weight_each_person_once_not_each_pair():
    """p1 is relevant to two queries and retrieved in one; p2 to one query and retrieved in it.
    Pooling pairs would give 2/3; weighting people equally gives the mean of 0.5 and 1.0."""
    rankings = {"q1": ["p1", "x"], "q2": ["x", "p1"], "q3": ["p2", "x"]}
    references = {"q1": ["p1"], "q2": ["p1"], "q3": ["p2"]}
    group_of = {"p1": "Female", "p2": "Female"}
    rates = individual_classification_rates(rankings, references, group_of, k=1)
    assert rates["Female"]["true_positive_rate"] == pytest.approx(0.75)


def test_individual_rates_separate_groups():
    rankings = {"q1": ["f1", "m1"], "q2": ["f1", "m1"]}
    references = {"q1": ["f1", "m1"], "q2": ["f1", "m1"]}
    group_of = {"f1": "Female", "m1": "Male"}
    rates = individual_classification_rates(rankings, references, group_of, k=1)
    assert rates["Female"]["true_positive_rate"] == pytest.approx(1.0)
    assert rates["Male"]["true_positive_rate"] == pytest.approx(0.0)


def test_individual_rates_count_people_not_pairs():
    rankings = {"q1": ["f1"], "q2": ["f1"]}
    references = {"q1": ["f1"], "q2": ["f1"]}
    rates = individual_classification_rates(rankings, references, {"f1": "Female"}, k=1)
    assert rates["Female"]["n_individuals"] == 1


def test_individual_rates_exclude_people_never_relevant_from_the_true_positive_rate():
    rankings = {"q1": ["f1", "f2"]}
    references = {"q1": ["f1"]}
    group_of = {"f1": "Female", "f2": "Female"}
    rates = individual_classification_rates(rankings, references, group_of, k=1)
    assert rates["Female"]["n_relevant_individuals"] == 1
    assert rates["Female"]["true_positive_rate"] == pytest.approx(1.0)


def test_individual_rates_do_not_fall_back_to_candidate_ids_when_the_query_group_is_unknown():
    """Job ids and talent ids share one integer namespace, so inferring the protected side per query
    lets an unlabelled query silently match a candidate id against the talent attribute table."""
    rankings = {"q_unknown": ["7"]}
    references = {"q_unknown": ["7"]}
    group_of = {"7": "Female"}
    rates = individual_classification_rates(rankings, references, group_of, k=1, group_side="query")
    assert rates["Female"]["n_individuals"] == 0


def test_individual_rates_group_by_candidate_when_asked():
    rankings = {"q1": ["7"]}
    references = {"q1": ["7"]}
    rates = individual_classification_rates(rankings, references, {"7": "Female"}, k=1,
                                             group_side="candidate")
    assert rates["Female"]["n_individuals"] == 1


def test_individual_rates_reject_an_unknown_group_side():
    with pytest.raises(ValueError):
        individual_classification_rates({"q1": ["a"]}, {"q1": ["a"]}, {"a": "Female"}, k=1,
                                         group_side="nonsense")


def test_individual_rate_values_return_one_value_per_person():
    rankings = {"q1": ["f1", "f2"], "q2": ["f1", "f2"]}
    references = {"q1": ["f1"], "q2": ["f2"]}
    group_of = {"f1": "Female", "f2": "Female"}
    values = individual_rate_values(rankings, references, group_of, k=1, group_side="candidate")
    assert len(values["Female"]["positive_rate"]) == 2


def test_individual_rate_values_agree_with_the_aggregate():
    """The aggregate must be the mean of the values, or the two can silently disagree."""
    rankings = {"q1": ["f1", "m1"], "q2": ["m1", "f1"]}
    references = {"q1": ["f1", "m1"], "q2": ["f1", "m1"]}
    group_of = {"f1": "Female", "m1": "Male"}
    values = individual_rate_values(rankings, references, group_of, k=1, group_side="candidate")
    rates = individual_classification_rates(rankings, references, group_of, k=1, group_side="candidate")
    for group in ("Female", "Male"):
        observed = values[group]["true_positive_rate"]
        assert rates[group]["true_positive_rate"] == pytest.approx(sum(observed) / len(observed))


def test_median_split_labels_below_and_at_or_above_the_median():
    values = {"a": 10.0, "b": 20.0, "c": 30.0, "d": 40.0}
    groups = median_split(values, low_label="Young", high_label="Old")
    assert groups["a"] == "Young"
    assert groups["b"] == "Young"
    assert groups["c"] == "Old"
    assert groups["d"] == "Old"


def test_median_split_excludes_values_outside_the_valid_range():
    """A value of 0 for age is a data error, not a real observation, and must not silently join
    either group or shift the median computed from the rest."""
    values = {"a": 0.0, "b": 20.0, "c": 30.0}
    groups = median_split(values, low_label="Young", high_label="Old", valid_range=(15, 100))
    assert "a" not in groups
    assert groups["b"] == "Young"
    assert groups["c"] == "Old"


def test_median_split_computes_the_median_only_over_values_kept_after_filtering():
    values = {"a": 0.0, "b": 10.0, "c": 20.0, "d": 30.0}
    groups = median_split(values, low_label="Young", high_label="Old", valid_range=(1, 100))
    # median of {10, 20, 30} is 20, not skewed by the excluded 0
    assert groups["b"] == "Young"
    assert groups["c"] == "Old"
    assert groups["d"] == "Old"


def test_binary_category_split_labels_the_target_and_pools_everything_else():
    values = {"a": "IN", "b": "CN", "c": "IT", "d": "IN"}
    groups = binary_category_split(values, target="IN", other_label="Other")
    assert groups == {"a": "IN", "b": "Other", "c": "Other", "d": "IN"}


def test_binary_category_split_excludes_missing_values():
    values = {"a": "IN", "b": float("nan"), "c": None}
    groups = binary_category_split(values, target="IN", other_label="Other")
    assert groups == {"a": "IN"}
