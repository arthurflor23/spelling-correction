"""
Transform Washington dataset
"""

import os
from data import preproc


class Transform():

    def __init__(self, source, charset, max_text_length):
        self.source = source
        self.charset = charset
        self.max_text_length = max_text_length
        self.lines = dict()
        self.partitions = dict()

    def build(self):
        lines = open(os.path.join(self.source, "ground_truth", "transcription.txt")).read().splitlines()

        for line in lines:
            splitted = line.split()
            splitted[1] = splitted[1].replace("-", "").replace("|", " ")
            splitted[1] = splitted[1].replace("s_pt", ".").replace("s_cm", ",")
            splitted[1] = splitted[1].replace("s_mi", "-").replace("s_qo", ":")
            splitted[1] = splitted[1].replace("s_sq", ";").replace("s_et", "V")
            splitted[1] = splitted[1].replace("s_bl", "(").replace("s_br", ")")
            splitted[1] = splitted[1].replace("s_qt", "'").replace("s_", "")
            self.lines[splitted[0]] = " ".join(splitted[1].split())

        self.partitions["train"] = self._build_partition("train.txt")
        self.partitions["valid"] = self._build_partition("valid.txt")
        self.partitions["test"] = self._build_partition("test.txt")

    def _build_partition(self, partition):
        partition_list = open(os.path.join(self.source, "sets", "cv1", partition)).read().splitlines()
        lines = []

        for partition in partition_list:
            lines.append(self.lines[partition])

        lines = list(set(lines))
        lines = preproc.text_normalization(lines, charset=self.charset, limit=self.max_text_length)

        return lines
