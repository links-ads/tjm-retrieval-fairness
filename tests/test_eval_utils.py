import pandas as pd

from tjm.utils.eval_utils import save_eval_results


def write(results_file, metrics, **overrides):
    kwargs = {
        "metrics": metrics,
        "results_file": results_file,
        "model_name": "bm25",
        "model_version": "v5",
        "collaborator": "links",
        "evaluation_mode": "job_to_talent",
        "candidate_pool": "full",
    }
    return save_eval_results(**{**kwargs, **overrides})


def test_writes_every_metric_it_is_given(tmp_path):
    path = tmp_path / "results.csv"
    write(path, {"recall@10": 0.25, "hit@10": 0.35, "ndcg@10": 0.3, "mrr": 0.2})
    row = pd.read_csv(path).iloc[0]
    assert row["recall@10"] == 0.25
    assert row["hit@10"] == 0.35
    assert row["ndcg@10"] == 0.3
    assert row["mrr"] == 0.2


def test_keeps_the_descriptive_columns(tmp_path):
    path = tmp_path / "results.csv"
    write(path, {"recall@10": 0.25})
    row = pd.read_csv(path).iloc[0]
    assert row["model"] == "bm25"
    assert row["model_version"] == "v5"
    assert row["collaborator"] == "links"
    assert row["evaluation_mode"] == "job_to_talent"
    assert row["candidate_pool"] == "full"


def test_appends_without_duplicating_the_header(tmp_path):
    path = tmp_path / "results.csv"
    write(path, {"recall@10": 0.1})
    write(path, {"recall@10": 0.2})
    assert list(pd.read_csv(path)["recall@10"]) == [0.1, 0.2]


def test_a_row_with_new_metrics_does_not_corrupt_earlier_rows(tmp_path):
    """Appending under a stale header would shift values into the wrong columns."""
    path = tmp_path / "results.csv"
    write(path, {"recall@10": 0.1}, model_name="old")
    write(path, {"recall@10": 0.2, "ndcg@10": 0.9}, model_name="new")

    frame = pd.read_csv(path)
    old, new = frame[frame.model == "old"].iloc[0], frame[frame.model == "new"].iloc[0]
    assert old["recall@10"] == 0.1
    assert pd.isna(old["ndcg@10"])
    assert new["recall@10"] == 0.2
    assert new["ndcg@10"] == 0.9


def test_records_provenance_when_supplied(tmp_path):
    path = tmp_path / "results.csv"
    write(path, {"recall@10": 0.1}, n_queries=38, seed=0, reference_date="2026-03-25")
    row = pd.read_csv(path).iloc[0]
    assert row["n_queries"] == 38
    assert row["seed"] == 0
    assert row["reference_date"] == "2026-03-25"
