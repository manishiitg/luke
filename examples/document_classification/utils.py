from typing import List, Tuple, Dict
import re
import numpy as np
import torch
from torch.utils.data import DataLoader

from transformers import PreTrainedTokenizer


LABEL_INDEX_MAPPING = {"CCAT": 0, "ECAT": 1, "GCAT": 2, "MCAT": 3}


class MldocReader:
    def __init__(self, tokenizer: PreTrainedTokenizer = None, max_seq_length: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def parse_file(self, file_path: str) -> List[Tuple[str, int]]:
        instances = []
        with open(file_path, "r") as f:
            for line in f:
                label, sentence = line.strip().split("\t")

                # clean the sentence
                sentence = re.sub(r"\u3000+", "\u3000", sentence)
                sentence = re.sub(r" +", " ", sentence)
                sentence = re.sub(r"\(c\) Reuters Limited \d\d\d\d", "", sentence)
                instances.append((sentence, label))
        return instances

    def _instance2feature(self, sentence: str, label: str) -> Dict[str, np.ndarray]:
        tokens = self.tokenizer.tokenize(sentence)
        word_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        label_id = LABEL_INDEX_MAPPING[label]

        return {
            "word_ids": np.array(word_ids),
            "word_segment_ids": np.zeros(len(word_ids), dtype=np.int),
            "word_attention_mask": np.ones(len(word_ids), dtype=np.int),
            "label": label_id,
        }

    def read(self, filepath: str) -> Dict[str, np.ndarray]:

        features = []
        for sentence, label in self.parse_file(filepath):
            feature_dict = self._instance2feature(sentence, label)
            if len(feature_dict["word_ids"]) < self.max_seq_length:
                features.append(feature_dict)

        return features


def features2data_loader(
    features: Dict[str, np.ndarray], pad_token_id: int, batch_size: int, shuffle: bool = False
) -> DataLoader:
    def collate_fn(batch: List[Dict[str, np.ndarray]]):
        def create_padded_sequence(attr_name: str, padding_value: int):
            tensors = [torch.tensor(o[attr_name], dtype=torch.long) for o in batch]
            return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)

        ret = dict(
            word_ids=create_padded_sequence("word_ids", pad_token_id),
            word_segment_ids=create_padded_sequence("word_segment_ids", 0),
            word_attention_mask=create_padded_sequence("word_attention_mask", 0),
        )

        ret["label"] = torch.LongTensor([o["label"] for o in batch])
        return ret

    return DataLoader(features, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
