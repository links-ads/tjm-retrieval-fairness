from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer


class MiniLmRetriever:
	def __init__(
		self,
		model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
		device: Optional[str] = None,
		**kwargs,
	) -> None:
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
			[query_text], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True
		)

		if candidate_ids is None:
			candidate_emb = self.model.encode(
					candidate_texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True
				)
		else:
			missing_texts = []
			missing_ids = []
			for cid, ctext in zip(candidate_ids, candidate_texts):
				if cid not in self._candidate_cache:
					missing_ids.append(cid)
					missing_texts.append(ctext)

			if missing_texts:
				new_embs = self.model.encode(
					missing_texts, convert_to_tensor=True, normalize_embeddings=True
				)
				for cid, emb in zip(missing_ids, new_embs):
					self._candidate_cache[cid] = emb

			candidate_emb_list = [self._candidate_cache[cid] for cid in candidate_ids]
			candidate_emb = torch.stack(candidate_emb_list)
		scores = torch.matmul(query_emb, candidate_emb.T).squeeze(0).cpu().tolist()
		return scores
