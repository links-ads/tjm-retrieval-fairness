from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_INSTRUCTION = "Given a talent resume or a job description, judge whether the candidate text is relevant to the query."


# Torch versions before 2.5 do not accept a device_type argument in torch.is_autocast_enabled.
# Qwen3 models call torch.is_autocast_enabled(device_type), which raises TypeError on older torch.
# Monkey-patch a backward-compatible wrapper when needed.
try:
	# Probe with a dummy device_type; if it fails, patch.
	torch.is_autocast_enabled("cuda")
except TypeError:
	orig_is_autocast_enabled = torch.is_autocast_enabled

	def _is_autocast_enabled_compat(device_type=None):  # type: ignore[override]
		return orig_is_autocast_enabled()

	torch.is_autocast_enabled = _is_autocast_enabled_compat  # type: ignore[assignment]


class QwenReranker:
	def __init__(
		self,
		model_name: str = "Qwen/Qwen3-Reranker-0.6B",
		device: Optional[str] = None,
		max_length: int = 8192,
		batch_size: int = 1,
		instruction: str | None = None,
		show_progress: bool = False,
		dtype: torch.dtype = torch.bfloat16,
		**kwargs,
	) -> None:
		self.model_name = model_name
		self.model_type = "reranker"
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		self.dtype = dtype
		self.tokenizer = AutoTokenizer.from_pretrained(
			model_name,
			padding_side="left",
			use_fast=False,
			trust_remote_code=True,
		)
		self.model = AutoModelForCausalLM.from_pretrained(
			model_name,
			trust_remote_code=True,
			torch_dtype=self.dtype,
		)
		self.model.to(self.device)
		self.max_length = max_length
		self.batch_size = batch_size
		self.instruction = instruction or DEFAULT_INSTRUCTION
		self.show_progress = show_progress

		self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
		self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
		self.prefix = (
			"<|im_start|>system\n"
			"You are a relevance judge for talent-job matching. "
			"Given a Query (talent resume or job description) and a Document (candidate resume or job description), respond only with yes if the Document is relevant to the Query, otherwise no.<|im_end|>\n<|im_start|>user\n"
		)
		self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
		self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
		self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

	def collate_fn(self, batch: list):
		pairs = [self._format_pair(b["query_text"], b["candidate_text"]) for b in batch]
		enc = self._prepare_inputs(pairs)  # already moved to self.device
		enc["labels"] = torch.tensor(
			[int(b["label"]) for b in batch],
			dtype=torch.float,
			# device=self.device,
		)
		return enc

	def _format_pair(self, query: str, doc: str) -> str:
		return (
			f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"
		)

	def _prepare_inputs(self, pairs: List[str]):
		"""
		For each pair, we construct the input as:
		<prefix><pair><suffix>
		Then we tokenize and pad the batch of inputs.
		We do manual prefix/suffix token addition to ensure they are not truncated.

		args:
			pairs: List of strings, each is a formatted "instruction + query + document"
		returns:
			inputs: Dict of tokenized inputs, ready to be fed into the model
		"""
		inputs = self.tokenizer(
			pairs,
			padding=False,
			truncation="longest_first",
			return_attention_mask=False,
			max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
		)
		for i, ids in enumerate(inputs["input_ids"]):
			inputs["input_ids"][i] = self.prefix_tokens + ids + self.suffix_tokens
		inputs = self.tokenizer.pad(
			inputs,
			padding=True,
			return_tensors="pt",
			max_length=self.max_length,
		)
		# inputs = {k: v.to(self.device) for k, v in inputs.items()}
		return inputs

	def compute_probs(self, outputs) -> List[float]:
		logits = outputs.logits[:, -1, :]
		true_vec = logits[:, self.token_true_id]
		false_vec = logits[:, self.token_false_id]
		stacked = torch.stack([false_vec, true_vec], dim=1)
		log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
		return log_probs[:, 1].exp() # return the "yes" probability as the relevance score
	
	@torch.no_grad()
	def score(
		self,
		query_text: str,
		candidate_texts: List[str],
	) -> List[float]:
		"""
		example from: https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
		
		"""
		self.model.eval()
		scores: List[float] = []
		total = len(candidate_texts)
		batch_iter = range(0, total, self.batch_size)
		for start in tqdm(batch_iter, desc="Qwen scoring batches", disable=not self.show_progress):
			batch_docs = candidate_texts[start : start + self.batch_size]
			pairs = [self._format_pair(query_text, doc) for doc in batch_docs]
			inputs = self._prepare_inputs(pairs)
			inputs = {k: v.to(self.device) for k, v in inputs.items()}
			outputs = self.model(**inputs)
			prob = self.compute_probs(outputs).cpu().tolist()  # Move to CPU immediately
			scores.extend(prob)	
			del inputs, outputs, prob		
			if self.device.startswith("cuda") and torch.cuda.is_available():
				torch.cuda.empty_cache()
		return scores
