import bm25s
import Stemmer

ANSERINI_K1 = 0.9
ANSERINI_B = 0.4


class BM25Retriever:
    """Lexical first-stage retriever over a fixed candidate collection.

    Uses Anserini's default parameters (k1=0.9, b=0.4) with the Lucene BM25 variant, lowercasing,
    English stopwords and Porter stemming, so results are comparable to a stock Pyserini run without
    requiring a JVM.

    Term statistics must come from the whole collection, so ``index_corpus`` has to be called before
    scoring. The evaluation harness scores candidates in small batches; deriving IDF and average
    document length from a batch would silently produce different numbers.
    """

    def __init__(
        self,
        model_name: str = "bm25",
        k1: float = ANSERINI_K1,
        b: float = ANSERINI_B,
        stemmer_language: str = "english",
        stopwords: str = "en",
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.model_type = "retriever"
        self.k1 = k1
        self.b = b
        self.stopwords = stopwords
        self.stemmer = Stemmer.Stemmer(stemmer_language)

        self._retriever: bm25s.BM25 | None = None
        self._id_to_row: dict[str, int] = {}
        self._text_to_id: dict[str, str] = {}

    def _tokenize(self, texts: list[str]) -> list[list[str]]:
        return bm25s.tokenize(
            texts,
            lower=True,
            stopwords=self.stopwords,
            stemmer=self.stemmer,
            return_ids=False,
            show_progress=False,
        )

    def index_corpus(self, corpus: dict[str, str]) -> None:
        """Build the BM25 index over the full candidate pool.

        Args:
            corpus: Maps candidate_id -> candidate text. Term statistics are derived from all of it.
        """
        candidate_ids = list(corpus.keys())
        texts = [corpus[cid] for cid in candidate_ids]

        self._retriever = bm25s.BM25(k1=self.k1, b=self.b, method="lucene")
        self._retriever.index(self._tokenize(texts), show_progress=False)
        self._id_to_row = {cid: row for row, cid in enumerate(candidate_ids)}
        self._text_to_id = {text: cid for cid, text in zip(candidate_ids, texts)}

    def score(
        self,
        query_text: str,
        candidate_texts: list[str],
        candidate_ids: list[str] | None = None,
    ) -> list[float]:
        """Score candidates against a query using collection-level term statistics.

        Args:
            query_text: The query document text.
            candidate_texts: Candidate texts, used to resolve ids when ``candidate_ids`` is omitted.
            candidate_ids: Optional explicit candidate ids, preferred when available.

        Returns:
            One BM25 score per candidate, in the order given.

        Raises:
            RuntimeError: If called before ``index_corpus``.
            KeyError: If a candidate cannot be resolved to an indexed document.
        """
        if self._retriever is None:
            raise RuntimeError("BM25Retriever.score requires index_corpus to be called first")

        resolved = candidate_ids or [self._text_to_id[text] for text in candidate_texts]
        query_tokens = self._tokenize([query_text])[0]
        if not query_tokens:
            return [0.0] * len(resolved)

        all_scores = self._retriever.get_scores(query_tokens)
        return [float(all_scores[self._id_to_row[cid]]) for cid in resolved]
