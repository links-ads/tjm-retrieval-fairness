import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


DEFAULT_TASK_DESCRIPTION = (
    "Given a talent-job matching query, retrieve relevant candidate texts that match the query intent."
)


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


class Qwen3EmbeddingRetriever:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str | None = None,
        max_length: int = 8192,
        batch_size: int = 16,
        task_description: str = DEFAULT_TASK_DESCRIPTION,
        use_query_instruction: bool = True,
        dtype: torch.dtype | None = None,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.model_type = "retriever"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size
        self.task_description = task_description
        self.use_query_instruction = use_query_instruction

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            trust_remote_code=True,
        )

        model_kwargs: dict = {"trust_remote_code": True}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()

        self._candidate_cache: dict[str, Tensor] = {}

    def _format_query(self, query_text: str) -> str:
        if not self.use_query_instruction:
            return query_text
        return get_detailed_instruct(self.task_description, query_text)

    @torch.no_grad()
    def _encode(self, texts: list[str]) -> Tensor:
        batch_dict = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

        outputs = self.model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        return F.normalize(embeddings, p=2, dim=1)

    @torch.no_grad()
    def score(
        self,
        query_text: str,
        candidate_texts: list[str],
        candidate_ids: list[str] | None = None,
    ) -> list[float]:
        query_emb = self._encode([self._format_query(query_text)])

        if candidate_ids is None:
            scores: list[float] = []
            for start in range(0, len(candidate_texts), self.batch_size):
                batch_candidates = candidate_texts[start : start + self.batch_size]
                candidate_emb = self._encode(batch_candidates)
                batch_scores = torch.matmul(query_emb, candidate_emb.T).squeeze(0).cpu().tolist()
                scores.extend(batch_scores)
            return scores

        missing_texts: list[str] = []
        missing_ids: list[str] = []
        for cid, ctext in zip(candidate_ids, candidate_texts):
            if cid not in self._candidate_cache:
                missing_ids.append(cid)
                missing_texts.append(ctext)

        for start in range(0, len(missing_texts), self.batch_size):
            batch_missing_texts = missing_texts[start : start + self.batch_size]
            batch_missing_ids = missing_ids[start : start + self.batch_size]
            new_embs = self._encode(batch_missing_texts)
            for cid, emb in zip(batch_missing_ids, new_embs):
                self._candidate_cache[cid] = emb

        candidate_emb = torch.stack([self._candidate_cache[cid] for cid in candidate_ids])
        return torch.matmul(query_emb, candidate_emb.T).squeeze(0).cpu().tolist()
