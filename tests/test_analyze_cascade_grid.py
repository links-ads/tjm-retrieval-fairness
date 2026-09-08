import json

import pytest

from tjm.metrics.cascade_grid import build_cascade_grid

DEPTH = 2


def write_run(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


@pytest.fixture
def runs_dir(tmp_path):
    # retA ranks gold 2nd; rerB, given the chance, would rank it 1st.
    retriever = {
        "q1": [
            {"candidate_id": "x", "score": 0.9},
            {"candidate_id": "gold", "score": 0.8},
            {"candidate_id": "y", "score": 0.1},
        ]
    }
    reranker = {
        "q1": [
            {"candidate_id": "gold", "score": 0.99},
            {"candidate_id": "x", "score": 0.2},
            {"candidate_id": "y", "score": 0.1},
        ]
    }
    write_run(tmp_path / "retA" / "triplets" / "job_to_talent_full.json", retriever)
    write_run(tmp_path / "rerB" / "triplets" / "job_to_talent_full.json", reranker)
    return tmp_path


def test_grid_reconstructs_a_cascade_for_every_pair_and_depth(runs_dir):
    references = {"q1": ["gold"]}
    rows = build_cascade_grid(
        runs_dir,
        split="triplets",
        first_stage_systems=["retA"],
        reranker_systems=["rerB"],
        depths=[DEPTH],
        evaluation_mode="job_to_talent",
        references=references,
        k_values=[1],
        n_resamples=50,
        seed=0,
    )
    row = rows[0]
    assert row["first_stage"] == "retA"
    assert row["reranker"] == "rerB"
    assert row["depth"] == DEPTH
    assert row["first_stage_recall"] == pytest.approx(1.0)
    assert row["recall@1_std"] == pytest.approx(0.0)


def test_grid_reflects_the_first_stage_ceiling(runs_dir):
    """Gold is ranked 2nd by retA, so a depth-1 shortlist drops it regardless of the reranker."""
    references = {"q1": ["gold"]}
    rows = build_cascade_grid(
        runs_dir,
        split="triplets",
        first_stage_systems=["retA"],
        reranker_systems=["rerB"],
        depths=[1],
        evaluation_mode="job_to_talent",
        references=references,
        k_values=[1],
        n_resamples=50,
        seed=0,
    )
    row = rows[0]
    assert row["first_stage_recall"] == pytest.approx(0.0)
    assert row["recall@1"] == pytest.approx(0.0)


def test_grid_skips_a_first_stage_reranker_pair_missing_from_disk(runs_dir):
    rows = build_cascade_grid(
        runs_dir,
        split="triplets",
        first_stage_systems=["retA", "missing"],
        reranker_systems=["rerB"],
        depths=[DEPTH],
        evaluation_mode="job_to_talent",
        references={"q1": ["gold"]},
        k_values=[1],
        n_resamples=50,
        seed=0,
    )
    assert {row["first_stage"] for row in rows} == {"retA"}


def test_grid_covers_every_requested_depth(runs_dir):
    rows = build_cascade_grid(
        runs_dir,
        split="triplets",
        first_stage_systems=["retA"],
        reranker_systems=["rerB"],
        depths=[1, 2, 3],
        evaluation_mode="job_to_talent",
        references={"q1": ["gold"]},
        k_values=[1],
        n_resamples=50,
        seed=0,
    )
    assert sorted(row["depth"] for row in rows) == [1, 2, 3]
