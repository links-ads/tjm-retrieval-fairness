import torch

from typing import List, Optional, Literal
from transformers import AutoTokenizer, AutoModel, AutoConfig

try:
    from peft import PeftModel
    from llm2vec import LLM2Vec
except ImportError:
    pass


class ShearedLLamaRetriever:
    def __init__(
        self,
        model_name: str = "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp",
        device: Optional[str] = None,
        evaluation_mode: Literal["job_to_talent", "talent_to_job"] = "job_to_talent",
    ) -> None:
        self.model_type = "retriever"
        self.model_name = model_name
        self.device = device
        self.evaluation_mode = evaluation_mode
    
        self._candidate_cache: dict[str, torch.Tensor] = {}
        
        self._init_model()
        self._init_instructions()


    def _init_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            config=self.config,
            torch_dtype=torch.bfloat16,
            device_map=self.device if self.device else "auto",
        )
        self.model = PeftModel.from_pretrained(
            self.model,
            f"{self.model_name}-supervised",
        )
        self.model = self.model.merge_and_unload()
        
        self.l2v = LLM2Vec(self.model, self.tokenizer, pooling_mode = "mean", max_length = 1024)

    def _init_instructions(self):
        if self.evaluation_mode == "job_to_talent":
            self.query_instruction = "Given a job description, retrieve relevant talent resumes. Job description: {query}"
            self.candidate_instruction = "Given a talent resume, judge whether it is relevant to the job description. Talent resume: {candidate}"
        else:
            self.query_instruction = "Given a talent resume, retrieve relevant job descriptions. Talent resume: {query}"
            self.candidate_instruction = "Given a job description, judge whether it is relevant to the talent resume. Job description: {candidate}"


    def score(
        self,
        query_text: str,
        candidate_texts: List[str],
        candidate_ids: Optional[List[str]] = None,
    ) -> List[float]:

        query_text = self.query_instruction.format(query=query_text)
        candidate_texts = [self.candidate_instruction.format(candidate=ctext) for ctext in candidate_texts]

        query_emb = self.l2v.encode([query_text])
        query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)

        if candidate_ids is None:
            candidate_emb = self.l2v.encode(candidate_texts)
            candidate_emb = torch.nn.functional.normalize(candidate_emb, p=2, dim=1)
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
                new_embs = self.l2v.encode(missing_texts)
                new_embs = torch.nn.functional.normalize(new_embs, p=2, dim=1)
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
