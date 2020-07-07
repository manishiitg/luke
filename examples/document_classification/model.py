import torch
import torch.nn.functional as F
from torch import nn

from luke.model import LukeModel

MLDOC_NUM_LABEL = 4


class LukeForDocumentClassification(LukeModel):
    def __init__(self, config):
        super(LukeForDocumentClassification, self).__init__(config)

        self.classifier = nn.Linear(config.hidden_size, MLDOC_NUM_LABEL)

        self.apply(self.init_weights)

    def forward(
        self,
        word_ids: torch.LongTensor,
        word_segment_ids: torch.LongTensor,
        word_attention_mask: torch.LongTensor,
        label: torch.LongTensor = None,
    ):
        _, pooled_output = super(LukeForDocumentClassification, self).forward(
            word_ids, word_segment_ids, word_attention_mask
        )

        logits = self.classifier(pooled_output)

        if label is not None:
            loss = F.cross_entropy(logits, label, ignore_index=-1)
            return loss, logits

        return logits
