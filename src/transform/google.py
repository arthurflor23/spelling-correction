"""Transform 1-Billion Google dataset (subset)"""

import os
from data import preproc as pp


class Transform():

    def __init__(self, source, charset, max_text_length):
        self.source = source
        self.charset = charset
        self.max_text_length = max_text_length
        self.partitions = dict()

    def build(self):
        m2_list = next(os.walk(self.source))[2]
        lines = []

        for m2_file in m2_list:
            if ("news-commentary-v6" in m2_file) and (".en" in m2_file or ".fr" in m2_file):
                lines += open(os.path.join(self.source, m2_file)).read().splitlines()

        lines = list(set(lines))
        lines = pp.normalize_text(lines, charset=self.charset, max_text_length=self.max_text_length)

        total = len(lines)
        train_i = int(total * 0.8)
        valid_i = train_i + int((total - train_i) / 2)

        self.partitions["train"] = lines[:train_i]
        self.partitions["valid"] = lines[train_i:valid_i]
        self.partitions["test"] = lines[valid_i:]
