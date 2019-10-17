"""Transform IAM dataset"""

import os
import numpy as np
from data import preproc as pp


class Transform():

    def __init__(self, source, charset, max_text_length):
        self.source = source
        self.charset = charset
        self.max_text_length = max_text_length
        self.lines = dict()
        self.partitions = dict()

    def build(self, only=True):
        lines = open(os.path.join(self.source, "ascii", "lines.txt")).read().splitlines()

        for line in lines:
            if (not line or line[0] == "#"):
                continue

            splitted = line.split()

            if splitted[1] == "ok":
                self.lines[splitted[0]] = " ".join(splitted[8::]).replace("|", " ")

        self.partitions["train"] = self._build_partition("trainset.txt")
        self.partitions["valid"] = self._build_partition("validationset1.txt")
        self.partitions["valid"] += self._build_partition("validationset2.txt")
        self.partitions["test"] = self._build_partition("testset.txt")

    def _build_partition(self, partition):
        sub = "largeWriterIndependentTextLineRecognitionTask"
        partition_list = open(os.path.join(self.source, sub, partition)).read().splitlines()
        lines = []

        for partition in partition_list:
            lines.append(self.lines[partition])

        lines = list(set(lines))
        lines = [y for x in lines for y in pp.split_by_max_length(x, self.max_text_length)]
        lines = [pp.text_standardize(x) for x in lines]
        np.random.shuffle(lines)

        return lines
