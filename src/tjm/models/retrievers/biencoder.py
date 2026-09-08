from dataclasses import dataclass
from torch import Tensor, tensor
from torch.nn import (
    Module,
    Parameter,
    functional as F
)


@dataclass
class BiEncoderOutput:
    logits: Tensor
    query_embeddings: Tensor
    candidate_embeddings: Tensor

class BiEncoderModel(Module):
    """Wraps a SentenceTransformer for bi-encoder training with HF Trainer.

    Adds learnable SigLIP temperature (exp of log_temperature) and bias
    parameters that scale the similarity matrix before the sigmoid loss.
    """

    def __init__(self, sentence_transformer, temperature_init=10.0, bias_init=-10.0):
        super().__init__()
        self.encoder = sentence_transformer
        self.log_temperature = Parameter(
            tensor(float(temperature_init)).log()
        )
        self.bias = Parameter(tensor(float(bias_init)))

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def _encode(self, input_ids, attention_mask):
        features = {"input_ids": input_ids, "attention_mask": attention_mask}
        output = self.encoder(features)
        return F.normalize(output["sentence_embedding"], dim=-1)

    def forward(
        self,
        query_input_ids,
        query_attention_mask,
        candidate_input_ids,
        candidate_attention_mask,
        **kwargs,
    ):
        query_emb = self._encode(query_input_ids, query_attention_mask)
        candidate_emb = self._encode(candidate_input_ids, candidate_attention_mask)
        logits = self.temperature * (query_emb * candidate_emb).sum(dim=-1) - self.bias
        return BiEncoderOutput(
            logits=logits,
            query_embeddings=query_emb,
            candidate_embeddings=candidate_emb,
        )