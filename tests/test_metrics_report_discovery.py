import json

import pytest

from tjm.metrics.report import discover_runs, summarise_runs

RUN_A = {"q1": [{"candidate_id": "gold", "score": 0.9}, {"candidate_id": "x", "score": 0.1}]}
RUN_B = {"q1": [{"candidate_id": "x", "score": 0.9}, {"candidate_id": "gold", "score": 0.1}]}


@pytest.fixture
def runs_dir(tmp_path):
    for model, payload in [("bm25", RUN_A), ("mpnet", RUN_B)]:
        directory = tmp_path / model / "triplets"
        directory.mkdir(parents=True)
        (directory / "job_to_talent_full.json").write_text(json.dumps(payload))
        (directory / "talent_to_job_observed.json").write_text(json.dumps(payload))
    return tmp_path


def test_discovers_runs_keyed_by_model_mode_and_pool(runs_dir):
    found = discover_runs(runs_dir, "triplets")
    assert ("bm25", "job_to_talent", "full") in found
    assert ("mpnet", "talent_to_job", "observed") in found
    assert len(found) == 4


def test_discovery_can_filter_by_mode_and_pool(runs_dir):
    found = discover_runs(runs_dir, "triplets", evaluation_mode="job_to_talent", candidate_pool="full")
    assert set(found) == {("bm25", "job_to_talent", "full"), ("mpnet", "job_to_talent", "full")}


def test_discovery_of_a_missing_split_is_empty(runs_dir):
    assert discover_runs(runs_dir, "test") == {}


def test_summarise_reports_means_with_confidence_intervals(runs_dir):
    references = {"q1": ["gold"]}
    rows = summarise_runs(
        discover_runs(runs_dir, "triplets", evaluation_mode="job_to_talent"),
        references,
        k_values=[1],
        n_resamples=100,
        seed=0,
    )
    row = next(r for r in rows if r["system"] == "bm25")
    assert row["recall@1"] == pytest.approx(1.0)
    assert row["recall@1_ci_low"] == pytest.approx(1.0)
    assert row["recall@1_std"] == pytest.approx(0.0)
    assert row["n_queries"] == 1


def test_summarise_distinguishes_the_two_systems(runs_dir):
    references = {"q1": ["gold"]}
    rows = summarise_runs(
        discover_runs(runs_dir, "triplets", evaluation_mode="job_to_talent"),
        references,
        k_values=[1],
        n_resamples=100,
        seed=0,
    )
    scores = {row["system"]: row["recall@1"] for row in rows}
    assert scores == {"bm25": 1.0, "mpnet": 0.0}


def test_summarise_returns_per_query_values_for_significance_testing(runs_dir):
    references = {"q1": ["gold"]}
    runs = discover_runs(runs_dir, "triplets", evaluation_mode="job_to_talent")
    rows = summarise_runs(runs, references, k_values=[1], n_resamples=100, seed=0, per_query_out=(store := {}))
    assert rows
    assert store[("bm25", "job_to_talent", "full")]["q1"]["recall@1"] == pytest.approx(1.0)
