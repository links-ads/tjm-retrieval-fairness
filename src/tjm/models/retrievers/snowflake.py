import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class SnowflakeModel:
    def __init__(
        self,
        model_name: str = "Snowflake/snowflake-arctic-embed-l",
        device: str | None = None,
        max_length: int = 512,
        query_prefix: str = DEFAULT_QUERY_PREFIX,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.query_prefix = query_prefix

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        self.model.to(self.device)
        self.model.eval()

        self._candidate_cache: dict[str, torch.Tensor] = {}

    def _encode(self, texts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.no_grad():
            embeddings = self.model(**tokens)[0][:, 0]

        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    @torch.no_grad()
    def score(
        self,
        query: str,
        candidates: list[str],
        candidate_ids: list[str] | None = None,
    ) -> list[float]:
        query_emb = self._encode([f"{self.query_prefix}{query}"])

        if candidate_ids is None:
            candidate_emb = self._encode(candidates)
        else:
            missing_texts = []
            missing_ids = []
            for cid, ctext in zip(candidate_ids, candidates):
                if cid not in self._candidate_cache:
                    missing_ids.append(cid)
                    missing_texts.append(ctext)

            if missing_texts:
                new_embs = self._encode(missing_texts)
                for cid, emb in zip(missing_ids, new_embs):
                    self._candidate_cache[cid] = emb

            candidate_emb = torch.stack([self._candidate_cache[cid] for cid in candidate_ids])

        scores = torch.matmul(query_emb, candidate_emb.T).squeeze(0).cpu().tolist()
        return scores