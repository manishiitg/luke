from typing import List, Tuple
import re
import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


LABEL_INDEX_MAPPING = {"CCAT": 0, "ECAT": 1, "GCAT": 2, "MCAT": 3}


def parse_mldoc(file_path: str):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            label, sentence = line.strip().split("\t")

            # clean the sentence
            sentence = re.sub(r"\u3000+", "\u3000", sentence)
            sentence = re.sub(r" +", " ", sentence)
            sentence = re.sub(r"\(c\) Reuters Limited \d\d\d\d", "", sentence)
            data.append((sentence, label))
    return data


class MLDocDataset(Dataset):
    def __init__(self, train_data_path: str = None, dev_data_path: str = None, test_data_path: str = None):
        if train_data_path:
            self.train = parse_mldoc(train_data_path)
        if dev_data_path:
            self.dev = parse_mldoc(dev_data_path)
        if test_data_path:
            self.test = parse_mldoc(test_data_path)


class InputFeatures(object):
    def __init__(self, word_ids: List[int], label: int):
        self.word_ids = np.array(word_ids)
        self.word_segment_ids = np.zeros(len(word_ids), dtype=np.int)
        self.word_attention_mask = np.ones(len(word_ids), dtype=np.int)
        self.label = label


def convert_documents_to_features(
    documents: List[Tuple[str, str]], tokenizer: PreTrainedTokenizer, max_sequence_length: int
):

    data = []
    for sentence, label in documents:
        tokens = tokenizer.tokenize(sentence)
        word_ids = tokenizer.convert_tokens_to_ids(tokens)
        label_id = LABEL_INDEX_MAPPING[label]

        if len(word_ids) < max_sequence_length:
            data.append(InputFeatures(word_ids, label_id))
    return data
