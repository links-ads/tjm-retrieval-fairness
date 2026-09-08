import pytest

from tjm.models.retrievers.bm25 import BM25Retriever

CORPUS = {
    "1": "machine learning engineer with deep neural network experience",
    "2": "pastry chef specialising in laminated dough and viennoiserie",
    "3": "machine operator for industrial packaging lines",
    "4": "deep sea marine biologist studying coral reef ecosystems",
}


@pytest.fixture
def retriever():
    model = BM25Retriever()
    model.index_corpus(CORPUS)
    return model


def test_declares_itself_a_retriever():
    assert BM25Retriever().model_type == "retriever"


def test_uses_anserini_default_parameters():
    model = BM25Retriever()
    assert (model.k1, model.b) == (0.9, 0.4)


def test_scoring_before_indexing_is_refused():
    with pytest.raises(RuntimeError, match="index_corpus"):
        BM25Retriever().score("machine learning", ["some text"])


def test_returns_one_score_per_candidate_in_the_given_order(retriever):
    scores = retriever.score("machine learning", [CORPUS["1"], CORPUS["2"], CORPUS["3"]])
    assert len(scores) == 3
    assert scores[0] > scores[2] > scores[1]


def test_unmatched_query_scores_zero(retriever):
    assert retriever.score("helicopter aerodynamics", [CORPUS["2"]]) == [0.0]


def test_stopwords_are_removed_so_a_stopword_query_scores_nothing(retriever):
    assert retriever.score("the and with for", [CORPUS["1"]]) == [0.0]


def test_stemming_matches_morphological_variants(retriever):
    assert retriever.score("ecosystem", [CORPUS["4"]])[0] > 0.0


def test_idf_is_collection_level_not_batch_level(retriever):
    """Scoring a single candidate must equal scoring it inside the full pool.

    This is the property that a per-batch BM25 would violate: with only one document in view its
    IDF and average document length differ from the collection's.
    """
    alone = retriever.score("machine learning", [CORPUS["1"]])[0]
    in_pool = retriever.score("machine learning", [CORPUS[cid] for cid in ["1", "2", "3", "4"]])[0]
    assert alone == pytest.approx(in_pool)


def test_rare_query_terms_outrank_common_ones(retriever):
    """'learning' occurs in one document, 'machine' in two, so it must carry more weight."""
    rare = retriever.score("learning", [CORPUS["1"]])[0]
    common = retriever.score("machine", [CORPUS["1"]])[0]
    assert rare > common


def test_scores_by_explicit_candidate_ids(retriever):
    by_id = retriever.score("machine learning", [""], candidate_ids=["1"])
    by_text = retriever.score("machine learning", [CORPUS["1"]])
    assert by_id == pytest.approx(by_text)


def test_unknown_candidate_text_is_refused(retriever):
    with pytest.raises(KeyError):
        retriever.score("machine learning", ["a resume that was never indexed"])
