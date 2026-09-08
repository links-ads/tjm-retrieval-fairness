from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModel


class JinaReranker:
	def __init__(
		self,
		model_name: str = "jinaai/jina-reranker-v3",
		device: Optional[str] = None,
		batch_size: int = 1,
		show_progress: bool = False,
		**kwargs,
	) -> None:
		self.model_name = model_name
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		self.batch_size = batch_size
		if self.batch_size < 1:
			raise ValueError("batch_size must be >= 1")
		config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
		# Avoid tying embeddings that can reference Identity modules without weights.
		if hasattr(config, "tie_word_embeddings"):
			config.tie_word_embeddings = False
		self.model = AutoModel.from_pretrained(
			model_name,
			config=config,
			trust_remote_code=True,
			torch_dtype="auto",
		)
		self.model.to(self.device)
		self.show_progress = show_progress

	def score(
		self,
		query_text: str,
		candidate_texts: List[str]
	) -> List[float]:
		"""score candidates in batches to avoid GPU out-of-memory spikes."""
		if not candidate_texts:
			return []

		scores = [0.0] * len(candidate_texts)
		progress = tqdm(
			total=len(candidate_texts),
			desc="Jina scoring",
			disable=(not self.show_progress),
		)

		chunk_size = self.batch_size
		start = 0
		while start < len(candidate_texts):
			end = min(start + chunk_size, len(candidate_texts))
			batch_candidates = candidate_texts[start:end]
			results = self.model.rerank(query_text, batch_candidates)
			for res in results:
				score = float(res.get("relevance_score", 0.0))
				idx = None
				if "index" in res:
					idx = int(res["index"])
				elif "corpus_id" in res:
					idx = int(res["corpus_id"])

				if idx is not None and 0 <= idx < len(batch_candidates):
					scores[start + idx] = score
				elif "document" in res:
					doc = res["document"]
					for i, cand in enumerate(batch_candidates):
						global_idx = start + i
						if cand == doc and scores[global_idx] == 0.0:
							scores[global_idx] = score
							break

			progress.update(end - start)
			start = end

		progress.close()
		return scores
