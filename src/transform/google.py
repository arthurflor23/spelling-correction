"""Transform 1-Billion Google dataset (subset)"""

import os
from data import preproc as pp


class Transform():

    def __init__(self, source, charset, max_text_length):
        self.source = source
        self.charset = charset
        self.max_text_length = max_text_length
        self.partitions = dict()

    def build(self, balance=True):
        m2_list = next(os.walk(self.source))[2]
        lines_en, lines_fr = [], []

        for m2_file in m2_list:
            if "2011" in m2_file:
                if ".en" in m2_file:
                    lines_en.extend(open(os.path.join(self.source, m2_file)).read().splitlines())
                elif ".fr" in m2_file:
                    lines_fr.extend(open(os.path.join(self.source, m2_file)).read().splitlines())

        lines_en = list(set(lines_en[::-1]))
        lines_fr = list(set(lines_fr[::-1]))

        # if dataset only 'google', english and french will be 42K samples.
        # if dataset is 'all', english will be 4.2K and french 42K samples.
        # this make a balance samples with the other datasets (english and french).
        en_split = 4.2e4 if balance else 4.2e3
        fr_split = 4.2e4

        lines_en = lines_en[:int(en_split)]
        lines_fr = lines_fr[:int(fr_split)]

        lines = lines_en + lines_fr
        del lines_en, lines_fr

        lines = pp.padding_punctuation(lines)
        lines = pp.split_by_max_length(lines, charset=self.charset, max_text_length=self.max_text_length)
        lines = pp.shuffle(lines)

        train_i = int(len(lines) * 0.8)
        valid_i = train_i + int((len(lines) - train_i) / 2)

        self.partitions["train"] = lines[:train_i]
        self.partitions["valid"] = lines[train_i:valid_i]
        self.partitions["test"] = lines[valid_i:]
