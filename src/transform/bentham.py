"""Transform Bentham dataset"""

import os
import html
from data import preproc as pp


class Transform():

    def __init__(self, source, charset, max_text_length):
        self.source = os.path.join(source, "BenthamDatasetR0-GT")
        self.charset = charset
        self.max_text_length = max_text_length
        self.partitions = dict()

    def build(self):
        self.partitions["train"] = self._build_partition("TrainLines.lst")
        self.partitions["valid"] = self._build_partition("ValidationLines.lst")
        self.partitions["test"] = self._build_partition("TestLines.lst")

    def _build_partition(self, partition):
        partition_list = open(os.path.join(self.source, "Partitions", partition)).read().splitlines()
        lines = []

        for item in partition_list:
            line = " ".join(open(os.path.join(self.source, "Transcriptions", f"{item}.txt")).read().splitlines())
            line = html.unescape(" ".join(line.split())).replace("<gap/>", "")

            if len(line) > 2:
                lines.append(line)

        lines = list(set(lines))
        lines = pp.standardize(lines, charset=self.charset, max_text_length=self.max_text_length)

        return lines
