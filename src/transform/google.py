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
        lines_en, lines_fr = [], []

        for m2_file in m2_list:
            if "2011" in m2_file:
                if ".en" in m2_file:
                    lines_en.extend(open(os.path.join(self.source, m2_file)).read().splitlines())
                elif ".fr" in m2_file:
                    lines_fr.extend(open(os.path.join(self.source, m2_file)).read().splitlines())

        lines_en = list(set(lines_en))
        lines_fr = list(set(lines_fr))

        min_value = min(len(lines_en), len(lines_fr))
        lines_en = lines_en[:min_value]
        lines_fr = lines_fr[:min_value]

        lines = lines_en + lines_fr
        del lines_en, lines_fr

        lines = pp.standardize(lines, charset=self.charset, max_text_length=self.max_text_length)
        lines = pp.shuffle(lines)

        train_i = int(len(lines) * 0.8)
        valid_i = train_i + int((len(lines) - train_i) / 2)

        self.partitions["train"] = lines[:train_i]
        self.partitions["valid"] = lines[train_i:valid_i]
        self.partitions["test"] = lines[valid_i:]
