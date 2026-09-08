from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BgeReranker:
	def __init__(
		self,
		model_name: str = "BAAI/bge-reranker-v2-m3",
		device: Optional[str] = None,
		max_length: int = 512,
		batch_size: int = 1,
		verbose: bool = True,
		**kwargs,
	) -> None:
		self.model_name = model_name
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		self.model.to(self.device)
		self.max_length = max_length
		self.batch_size = batch_size
		self.verbose = verbose

	def score(
		self,
		query_text: str,
		candidate_texts: List[str],
	) -> List[float]:
		scores: List[float] = []
		total = len(candidate_texts)
		batch_iter = range(0, total, self.batch_size)
		if self.verbose:
			batch_iter = tqdm(batch_iter, desc="BGE scoring batches")
		for start in batch_iter:
			end = min(start + self.batch_size, total)
			batch_pairs = [
				(query_text, candidate_text)
				for candidate_text in candidate_texts[start:end]
			]
			if self.verbose and not isinstance(batch_iter, tqdm):
				print(f"BGE scoring batch {start}-{end} of {total}")
			encoded = self.tokenizer(
				batch_pairs,
				padding=True,
				truncation=True,
				max_length=self.max_length,
				return_tensors="pt",
			)
			encoded = {k: v.to(self.device) for k, v in encoded.items()}
			with torch.no_grad():
				logits = self.model(**encoded, return_dict=True).logits.view(-1,).float()
			scores.extend(logits.detach().cpu().tolist())
		return scores
