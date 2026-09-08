from tjm.models.rerankers import (
	BgeReranker,
	JinaReranker,
	QwenReranker,
)
from tjm.models.retrievers import (
	BM25Retriever,
	MpnetRetriever,
	MiniLmRetriever,
	Qwen3EmbeddingRetriever,
	ShearedLLamaRetriever,
	SnowflakeModel,
)


def load_model(
	model_name: str,
	device: str | None = None,
	pretrained: str | None = None,
	show_progress: bool = False,
	**kwargs,
):
	retrievers = {
		"bm25": (BM25Retriever, "bm25"),
		"mpnet": (MpnetRetriever, "sentence-transformers/all-mpnet-base-v2"),
		"minilm": (MiniLmRetriever, "sentence-transformers/all-MiniLM-L6-v2"),
		"sheared_llama": (ShearedLLamaRetriever, "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp"),
		"snowflake": (SnowflakeModel, "Snowflake/snowflake-arctic-embed-l"),
		"qwen3_embedding": (Qwen3EmbeddingRetriever, "Qwen/Qwen3-Embedding-0.6B"),
		"qwen3_embedding4": (Qwen3EmbeddingRetriever, "Qwen/Qwen3-Embedding-4B"),
		"qwen3_embedding8": (Qwen3EmbeddingRetriever, "Qwen/Qwen3-Embedding-8B"),
	}
	rerankers = {
		"bge": (BgeReranker, "BAAI/bge-reranker-v2-m3"),
		"bge-base": (BgeReranker, "BAAI/bge-reranker-base"),
		"bge-large": (BgeReranker, "BAAI/bge-reranker-large"),
		"qwen": (QwenReranker, "Qwen/Qwen3-Reranker-0.6B"),
		"qwen4": (QwenReranker, "Qwen/Qwen3-Reranker-4B"),
		"qwen8": (QwenReranker, "Qwen/Qwen3-Reranker-8B"),
		"jina": (JinaReranker, "jinaai/jina-reranker-v3"),
	}

	if model_name in retrievers:
		cls, default_ckpt = retrievers[model_name]
		return cls(model_name=pretrained or default_ckpt, device=device, **kwargs)
	if model_name in rerankers:
		cls, default_ckpt = rerankers[model_name]
		try:
			return cls(model_name=pretrained or default_ckpt, device=device, show_progress=show_progress, **kwargs)
		except TypeError:
			return cls(model_name=pretrained or default_ckpt, device=device, **kwargs)
	raise ValueError(f"Unknown model name: {model_name}")