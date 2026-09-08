import pytest

from tjm.metrics.report import correct_family


def test_corrects_p_values_jointly_across_the_whole_family():
    rows = [{"p_value": 0.01}, {"p_value": 0.04}, {"p_value": 0.03}]
    corrected = correct_family(rows, alpha=0.05)
    assert [r["p_value_corrected"] for r in corrected] == pytest.approx([0.03, 0.06, 0.06])
    assert [r["significant"] for r in corrected] == [True, False, False]


def test_a_single_row_alone_is_a_separate_family_from_a_row_tested_jointly():
    """The bug this guards against: calling Holm on one hypothesis at a time is a no-op."""
    jointly = correct_family([{"p_value": 0.04}, {"p_value": 0.04}, {"p_value": 0.04}], alpha=0.05)
    alone = correct_family([{"p_value": 0.04}], alpha=0.05)
    assert jointly[0]["p_value_corrected"] > alone[0]["p_value_corrected"]


def test_preserves_the_original_row_contents():
    rows = [{"p_value": 0.02, "system": "a", "delta": 0.1}]
    corrected = correct_family(rows, alpha=0.05)
    assert corrected[0]["system"] == "a"
    assert corrected[0]["delta"] == 0.1


def test_empty_family_returns_empty():
    assert correct_family([], alpha=0.05) == []
