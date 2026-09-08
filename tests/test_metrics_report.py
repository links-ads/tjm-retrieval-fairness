import pandas as pd
import pytest

from tjm.metrics.report import align_per_query, build_references, compare_against_baseline

JOBS = pd.DataFrame({"id": [10, 11], "text": ["job ten", "job eleven"]})
RESUMES = pd.DataFrame({"id": [1, 2, 3], "cleaned_text": ["resume one", "resume two", ""]})
TRIPLETS = pd.DataFrame(
    {
        "talent_id": [1, 2, 3, 2],
        "vacancy_id": [10, 10, 11, 11],
        "label": ["Fit", "Fit", "Fit", "Unlabeled"],
    }
)


@pytest.fixture
def data_dir(tmp_path):
    JOBS.to_csv(tmp_path / "jobs.csv", index=False)
    RESUMES.to_csv(tmp_path / "resumes.csv", index=False)
    TRIPLETS.to_csv(tmp_path / "triplets.csv", index=False)
    return str(tmp_path)


def test_references_map_jobs_to_their_fit_talents(data_dir):
    assert build_references(data_dir, "triplets", "job_to_talent") == {"10": ["1", "2"]}


def test_references_invert_for_talent_to_job(data_dir):
    assert build_references(data_dir, "triplets", "talent_to_job") == {"1": ["10"], "2": ["10"]}


def test_references_drop_pairs_whose_text_is_empty(data_dir):
    """Talent 3 has no resume text, so job 11 loses its only positive and stops being a query."""
    assert "11" not in build_references(data_dir, "triplets", "job_to_talent")


def test_references_ignore_unlabeled_pairs(data_dir):
    assert "2" not in build_references(data_dir, "triplets", "talent_to_job").get("11", [])


def test_align_restricts_to_the_shared_query_set():
    systems = {
        "a": {"q1": {"recall@10": 0.5}, "q2": {"recall@10": 1.0}},
        "b": {"q1": {"recall@10": 0.25}},
    }
    query_ids, values = align_per_query(systems, "recall@10")
    assert query_ids == ["q1"]
    assert values == {"a": [0.5], "b": [0.25]}


def test_align_keeps_values_in_a_consistent_query_order():
    systems = {
        "a": {"q2": {"m": 1.0}, "q1": {"m": 0.0}},
        "b": {"q1": {"m": 0.5}, "q2": {"m": 0.75}},
    }
    query_ids, values = align_per_query(systems, "m")
    assert query_ids == ["q1", "q2"]
    assert values == {"a": [0.0, 1.0], "b": [0.5, 0.75]}


def test_align_refuses_a_disjoint_query_set():
    systems = {"a": {"q1": {"m": 1.0}}, "b": {"q2": {"m": 1.0}}}
    with pytest.raises(ValueError, match="no shared queries"):
        align_per_query(systems, "m")


def test_comparison_reports_deltas_against_the_baseline():
    systems = {
        "baseline": {f"q{i}": {"m": 0.0} for i in range(10)},
        "better": {f"q{i}": {"m": 1.0} for i in range(10)},
    }
    rows = compare_against_baseline(systems, baseline="baseline", metric="m", n_resamples=200, seed=0)
    row = next(r for r in rows if r["system"] == "better")
    assert row["delta"] == pytest.approx(1.0)
    assert row["n_queries"] == 10


def test_comparison_applies_holm_correction_across_the_family():
    systems = {
        "baseline": {f"q{i}": {"m": 0.0} for i in range(10)},
        "a": {f"q{i}": {"m": 1.0} for i in range(10)},
        "b": {f"q{i}": {"m": 1.0} for i in range(10)},
    }
    rows = compare_against_baseline(systems, baseline="baseline", metric="m", n_resamples=200, seed=0)
    for row in rows:
        assert row["p_value_corrected"] >= row["p_value"]


def test_comparison_excludes_the_baseline_from_its_own_family():
    systems = {
        "baseline": {f"q{i}": {"m": 0.0} for i in range(5)},
        "a": {f"q{i}": {"m": 1.0} for i in range(5)},
    }
    rows = compare_against_baseline(systems, baseline="baseline", metric="m", n_resamples=100, seed=0)
    assert [row["system"] for row in rows] == ["a"]


def test_comparison_records_the_metric_and_baseline_names():
    systems = {
        "baseline": {f"q{i}": {"m": 0.0} for i in range(5)},
        "a": {f"q{i}": {"m": 0.5} for i in range(5)},
    }
    row = compare_against_baseline(systems, baseline="baseline", metric="m", n_resamples=100, seed=0)[0]
    assert row["metric"] == "m"
    assert row["baseline"] == "baseline"
