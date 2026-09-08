from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer


class MpnetRetriever:
	def __init__(
		self,
		model_name: str = "sentence-transformers/all-mpnet-base-v2",
		device: Optional[str] = None,
		**kwargs,
	) -> None:
		self.model_type = "retriever"
		self.model_name = model_name
		self.model = SentenceTransformer(model_name, device=device)
		self._candidate_cache: dict[str, torch.Tensor] = {}

	def score(
		self,
		query_text: str,
		candidate_texts: List[str],
		candidate_ids: Optional[List[str]] = None,
	) -> List[float]:

		query_emb = self.model.encode(
			[query_text], convert_to_tensor=True, normalize_embeddings=True
		)

		if candidate_ids is None:
			candidate_emb = self.model.encode(
				candidate_texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True
			)
		else:
			missing_texts = []
			missing_ids = []
			cached = []
			for cid, ctext in zip(candidate_ids, candidate_texts):
				if cid in self._candidate_cache:
					cached.append(self._candidate_cache[cid])
				else:
					missing_ids.append(cid)
					missing_texts.append(ctext)

			if missing_texts:
				new_embs = self.model.encode(
					missing_texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True
				)
				for cid, emb in zip(missing_ids, new_embs):
					self._candidate_cache[cid] = emb

			all_embs = []
			cached_iter = iter(cached)
			new_iter = iter(new_embs) if missing_texts else iter([])
			for cid in candidate_ids:
				if cid in self._candidate_cache:
					all_embs.append(self._candidate_cache[cid])
				else:
					all_embs.append(next(new_iter))
			candidate_emb = torch.stack(all_embs)

		scores = torch.matmul(query_emb, candidate_emb.T).squeeze(0).cpu().tolist()

		return scores
