import pytest

from tjm.evaluate import build_ir_payload
from tjm.models.load_model import load_model
from tjm.models.retrievers.bm25 import BM25Retriever

QUERY = "machine learning engineer wanted"
POOL = {
    "10": "machine learning engineer with pytorch experience",
    "11": "pastry chef specialising in laminated dough",
    "12": "marine biologist studying coral reefs",
}
DATASET = [
    {"query_id": 1, "query": QUERY, "candidate_id": 10, "candidate": POOL["10"], "label": 1},
    {"query_id": 1, "query": QUERY, "candidate_id": 11, "candidate": POOL["11"], "label": 0},
]


class RecordingModel:
    model_name = "recording"
    model_type = "retriever"

    def __init__(self):
        self.indexed_corpus = None

    def index_corpus(self, corpus):
        self.indexed_corpus = corpus

    def score(self, query_text, candidate_texts):
        return [1.0] * len(candidate_texts)


class PlainModel:
    model_name = "plain"
    model_type = "retriever"

    def score(self, query_text, candidate_texts):
        return [1.0] * len(candidate_texts)


def test_load_model_resolves_bm25():
    assert isinstance(load_model("bm25"), BM25Retriever)


def test_build_ir_payload_indexes_the_full_pool_for_indexable_models():
    model = RecordingModel()
    build_ir_payload(DATASET, model, candidates_pool=POOL, candidate_pool="observed")
    assert model.indexed_corpus == POOL


def test_build_ir_payload_still_works_for_models_without_indexing():
    predictions, references = build_ir_payload(DATASET, PlainModel(), candidates_pool=POOL, candidate_pool="observed")
    assert references == {"1": ["10"]}
    assert set(predictions["1"]) == {"10", "11"}


def test_bm25_runs_end_to_end_through_the_harness():
    predictions, references = build_ir_payload(
        DATASET, load_model("bm25"), candidates_pool=POOL, candidate_pool="full"
    )
    assert set(predictions["1"]) == {"10", "11", "12"}
    assert predictions["1"]["10"] > predictions["1"]["11"]


def test_build_ir_payload_returns_metric_ready_payload_for_the_shared_module():
    from tjm.metrics.ranking import evaluate_run

    predictions, references = build_ir_payload(
        DATASET, load_model("bm25"), candidates_pool=POOL, candidate_pool="full"
    )
    means, per_query = evaluate_run(predictions, references, [1])
    assert means["hit@1"] == pytest.approx(1.0)
    assert per_query["1"]["recall@1"] == pytest.approx(1.0)
