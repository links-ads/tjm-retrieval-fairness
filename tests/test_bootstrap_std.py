import pytest

from tjm.metrics.significance import bootstrap_std


def test_bootstrap_std_is_reproducible_for_a_fixed_seed():
    values = [0.1, 0.4, 0.2, 0.9, 0.5, 0.3]
    assert bootstrap_std(values, n_resamples=500, seed=7) == bootstrap_std(values, n_resamples=500, seed=7)


def test_bootstrap_std_changes_with_the_seed():
    values = [0.1, 0.4, 0.2, 0.9, 0.5, 0.3]
    assert bootstrap_std(values, n_resamples=500, seed=1) != bootstrap_std(values, n_resamples=500, seed=2)


def test_bootstrap_std_is_zero_when_all_values_are_equal():
    assert bootstrap_std([0.5] * 8, n_resamples=200, seed=0) == pytest.approx(0.0)


def test_bootstrap_std_is_positive_for_dispersed_values():
    assert bootstrap_std([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], n_resamples=2000, seed=0) > 0.0


def test_bootstrap_std_shrinks_with_more_queries_at_the_same_dispersion():
    """More queries at the same per-query variance narrows the estimate of the mean."""
    small = bootstrap_std([0.0, 1.0] * 5, n_resamples=5000, seed=0)
    large = bootstrap_std([0.0, 1.0] * 500, n_resamples=5000, seed=0)
    assert large < small


def test_bootstrap_std_raises_on_empty_input():
    with pytest.raises(ValueError):
        bootstrap_std([], n_resamples=100, seed=0)
