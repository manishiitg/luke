import torch
import torch.nn.functional as F
from torch import nn

from transformers import PreTrainedModel

MLDOC_NUM_LABELS = 4


class DocumentClassificationModel(nn.Module):
    def __init__(self, encoder: PreTrainedModel):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(self.encoder.config.hidden_size, MLDOC_NUM_LABELS)

    def forward(
        self,
        word_ids: torch.LongTensor,
        word_segment_ids: torch.LongTensor,
        word_attention_mask: torch.LongTensor,
        label: torch.LongTensor = None,
    ):
        _, pooled_output = self.encoder.forward(word_ids, word_segment_ids, word_attention_mask)

        logits = self.classifier(pooled_output)

        if label is not None:
            loss = F.cross_entropy(logits, label, ignore_index=-1)
            return loss, logits

        return logits
