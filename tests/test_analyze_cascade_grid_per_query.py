import pytest

from tjm.metrics.cascade_grid import build_cascade_grid


def test_grid_can_report_its_per_query_values_for_significance_testing(tmp_path):
    import json

    retriever = {"q1": [{"candidate_id": "x", "score": 0.9}, {"candidate_id": "gold", "score": 0.8}]}
    reranker = {"q1": [{"candidate_id": "gold", "score": 0.99}, {"candidate_id": "x", "score": 0.2}]}
    for name, payload in [("retA", retriever), ("rerB", reranker)]:
        path = tmp_path / name / "triplets"
        path.mkdir(parents=True)
        (path / "job_to_talent_full.json").write_text(json.dumps(payload))

    store: dict = {}
    build_cascade_grid(
        tmp_path,
        split="triplets",
        first_stage_systems=["retA"],
        reranker_systems=["rerB"],
        depths=[2],
        evaluation_mode="job_to_talent",
        references={"q1": ["gold"]},
        k_values=[1],
        n_resamples=50,
        seed=0,
        per_query_out=store,
    )
    assert store[("retA", "rerB", 2)]["q1"]["recall@1"] == pytest.approx(1.0)
