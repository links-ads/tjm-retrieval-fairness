import pytest

from tjm.metrics.significance import bootstrap_ci, holm_bonferroni, paired_bootstrap_test


def test_bootstrap_ci_is_reproducible_for_a_fixed_seed():
    values = [0.1, 0.4, 0.2, 0.9, 0.5, 0.3]
    assert bootstrap_ci(values, n_resamples=500, seed=7) == bootstrap_ci(values, n_resamples=500, seed=7)


def test_bootstrap_ci_changes_with_the_seed():
    values = [0.1, 0.4, 0.2, 0.9, 0.5, 0.3]
    assert bootstrap_ci(values, n_resamples=500, seed=1) != bootstrap_ci(values, n_resamples=500, seed=2)


def test_bootstrap_ci_reports_the_observed_mean_and_brackets_it():
    values = [0.1, 0.4, 0.2, 0.9, 0.5, 0.3]
    mean, lo, hi = bootstrap_ci(values, n_resamples=2000, seed=0)
    assert mean == pytest.approx(sum(values) / len(values))
    assert lo <= mean <= hi


def test_bootstrap_ci_collapses_when_all_values_are_equal():
    mean, lo, hi = bootstrap_ci([0.5] * 8, n_resamples=200, seed=0)
    assert (mean, lo, hi) == (0.5, 0.5, 0.5)


def test_paired_bootstrap_reports_no_difference_for_identical_systems():
    values = [0.2, 0.7, 0.1, 0.9, 0.4]
    result = paired_bootstrap_test(values, values, n_resamples=500, seed=0)
    assert result["delta"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)


def test_paired_bootstrap_finds_a_separated_system_significant():
    a = [1.0] * 12
    b = [0.0] * 12
    result = paired_bootstrap_test(a, b, n_resamples=500, seed=0)
    assert result["delta"] == pytest.approx(1.0)
    assert result["p_value"] == pytest.approx(0.0)


def test_paired_bootstrap_reports_delta_confidence_interval():
    a = [0.9, 0.8, 0.7, 0.95, 0.85, 0.75]
    b = [0.1, 0.2, 0.3, 0.05, 0.15, 0.25]
    result = paired_bootstrap_test(a, b, n_resamples=2000, seed=0)
    assert result["ci_low"] <= result["delta"] <= result["ci_high"]
    assert result["ci_low"] > 0.0


def test_paired_bootstrap_rejects_mismatched_query_counts():
    with pytest.raises(ValueError):
        paired_bootstrap_test([0.1, 0.2], [0.1], n_resamples=10, seed=0)


def test_paired_bootstrap_is_reproducible_for_a_fixed_seed():
    a = [0.3, 0.6, 0.1, 0.8]
    b = [0.2, 0.5, 0.4, 0.3]
    assert paired_bootstrap_test(a, b, n_resamples=300, seed=3) == paired_bootstrap_test(
        a, b, n_resamples=300, seed=3
    )


def test_holm_corrects_a_known_example():
    corrected, reject = holm_bonferroni([0.01, 0.04, 0.03], alpha=0.05)
    assert corrected == pytest.approx([0.03, 0.06, 0.06])
    assert reject == [True, False, False]


def test_holm_caps_corrected_values_at_one():
    corrected, _ = holm_bonferroni([0.5, 0.6], alpha=0.05)
    assert max(corrected) <= 1.0


def test_holm_on_a_single_hypothesis_leaves_it_unchanged():
    corrected, reject = holm_bonferroni([0.02], alpha=0.05)
    assert corrected == pytest.approx([0.02])
    assert reject == [True]


def test_holm_is_monotone_in_sorted_order():
    corrected, _ = holm_bonferroni([0.001, 0.002, 0.003, 0.004], alpha=0.05)
    assert corrected == sorted(corrected)
