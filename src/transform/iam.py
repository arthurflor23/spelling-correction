"""Transform IAM dataset"""

import os
from data import preproc as pp


class Transform():

    def __init__(self, source, charset, max_text_length):
        self.source = source
        self.charset = charset
        self.max_text_length = max_text_length
        self.lines = dict()
        self.partitions = dict()

    def build(self):
        lines = open(os.path.join(self.source, "ascii", "lines.txt")).read().splitlines()

        for line in lines:
            if (not line or line[0] == "#"):
                continue

            splitted = line.split()
            self.lines[splitted[0]] = " ".join(splitted[-1].replace("|", " ").split())

        self.partitions["train"] = self._build_partition("trainset.txt")
        self.partitions["valid"] = self._build_partition("validationset1.txt")
        self.partitions["test"] = self._build_partition("testset.txt")

    def _build_partition(self, partition):
        sub = "largeWriterIndependentTextLineRecognitionTask"
        partition_list = open(os.path.join(self.source, sub, partition)).read().splitlines()
        lines = []

        for partition in partition_list:
            lines.append(self.lines[partition])

        lines = list(set(lines))
        lines = pp.standardize(lines, charset=self.charset, max_text_length=self.max_text_length)

        return lines
