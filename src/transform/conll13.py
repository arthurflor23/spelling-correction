"""Transform CoNLL13 dataset"""

import os
from data import preproc as pp, m2


class Transform():

    def __init__(self, source, charset, max_text_length):
        self.m2_file = os.path.join(source, "revised", "data", "official-preprocessed.m2")
        self.charset = charset
        self.max_text_length = max_text_length
        self.partitions = dict()

    def build(self, only=True):
        lines = list(set(m2.read_raw(self.m2_file)))
        lines = [pp.padding_punctuation(x) for x in lines]
        lines = [y for x in lines for y in pp.split_by_max_length(x, self.charset, self.max_text_length)]
        lines = pp.shuffle(lines)

        total = len(lines)
        train_i = int(total * 0.8)
        valid_i = train_i + int((total - train_i) / 2)

        self.partitions["train"] = lines[:train_i]
        self.partitions["valid"] = lines[train_i:valid_i]
        self.partitions["test"] = lines[valid_i:]
