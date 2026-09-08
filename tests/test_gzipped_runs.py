import gzip
import json

from tjm.metrics import discover_runs, load_run


def write_run(path, payload, compressed):
    opener = gzip.open if compressed else open
    with opener(path, "wt") as handle:
        json.dump(payload, handle)


PAYLOAD = {"q1": [{"candidate_id": "a", "score": 2.0}, {"candidate_id": "b", "score": 1.0}]}


def test_load_run_reads_gzipped_predictions(tmp_path):
    path = tmp_path / "job_to_talent_full.json.gz"
    write_run(path, PAYLOAD, compressed=True)
    assert load_run(path) == {"q1": {"a": 2.0, "b": 1.0}}


def test_load_run_still_reads_plain_json(tmp_path):
    path = tmp_path / "job_to_talent_full.json"
    write_run(path, PAYLOAD, compressed=False)
    assert load_run(path) == {"q1": {"a": 2.0, "b": 1.0}}


def test_discover_runs_finds_gzipped_predictions(tmp_path):
    split_dir = tmp_path / "bm25" / "triplets"
    split_dir.mkdir(parents=True)
    write_run(split_dir / "job_to_talent_full.json.gz", PAYLOAD, compressed=True)
    runs = discover_runs(tmp_path, "triplets", evaluation_mode="job_to_talent", candidate_pool="full")
    assert set(runs) == {("bm25", "job_to_talent", "full")}


def test_discover_runs_prefers_plain_json_when_both_exist(tmp_path):
    split_dir = tmp_path / "bm25" / "triplets"
    split_dir.mkdir(parents=True)
    write_run(split_dir / "job_to_talent_full.json", PAYLOAD, compressed=False)
    write_run(split_dir / "job_to_talent_full.json.gz", PAYLOAD, compressed=True)
    runs = discover_runs(tmp_path, "triplets", evaluation_mode="job_to_talent", candidate_pool="full")
    assert runs[("bm25", "job_to_talent", "full")].suffix == ".json"


def test_cascade_grid_reads_gzipped_runs(tmp_path):
    """The grid must see gzipped runs, or a repository storing them compressed silently
    reconstructs no cascades at all."""
    for system, scores in [("retriever", [3.0, 2.0, 1.0]), ("reranker", [1.0, 3.0, 2.0])]:
        split_dir = tmp_path / system / "triplets"
        split_dir.mkdir(parents=True)
        payload = {
            "q1": [{"candidate_id": c, "score": s} for c, s in zip("abc", scores)],
        }
        write_run(split_dir / "job_to_talent_full.json.gz", payload, compressed=True)

    from tjm.metrics import build_cascade_grid

    grid = build_cascade_grid(
        tmp_path,
        split="triplets",
        first_stage_systems=["retriever"],
        reranker_systems=["reranker"],
        depths=[2],
        evaluation_mode="job_to_talent",
        references={"q1": ["b"]},
        k_values=[1],
        n_resamples=10,
    )
    assert len(grid) == 1
    assert grid[0]["first_stage"] == "retriever" and grid[0]["reranker"] == "reranker"
