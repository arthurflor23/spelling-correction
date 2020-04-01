"""Dataset reader and process"""

import os
import html
import string
import numpy as np
import xml.etree.ElementTree as ET

from glob import glob
from data import preproc as pp


class Dataset():

    def __init__(self, source):
        self.source = os.path.splitext(source)[0]
        self.partitions = ['train', 'valid', 'test']
        self.dataset = dict()
        self.size = {'total': 0}

        for pt in self.partitions:
            self.size[pt] = 0

    def read_lines(self, maxlen):
        """Read sentences from dataset and preprocess"""

        name = os.path.basename(self.source)
        print(f"The {name} dataset will be transformed...")

        self.dataset = getattr(self, f"_{name}")()
        multigrams = dict()

        if isinstance(self.dataset, list):
            index = int(len(self.dataset) * 0.1)
            _dataset = dict()

            _dataset['train'] = self.dataset[index:-(index // 2)]
            _dataset['valid'] = self.dataset[index:]
            _dataset['test'] = self.dataset[:-(index // 2)]

            self.dataset = _dataset
            del _dataset

        for pt in self.partitions:
            # split sentences by max length and standardize it
            self.dataset[pt] = [y for x in self.dataset[pt] for y in pp.split_by_max_length(x, maxlen)]

            self.dataset[pt] = [pp.text_standardize(x) for x in self.dataset[pt]]
            self.dataset[pt] = [x for x in self.dataset[pt] if self.check_text(x)]

            # generate multigrams and standardize it
            _multigrams = [pp.generate_multigrams(x) for x in self.dataset[pt]]

            multigrams[pt] = list(set([pp.text_standardize(y) for x in _multigrams for y in x]))
            multigrams[pt] = [x for x in multigrams[pt] if self.check_text(x)]

            self.size[pt] += len(self.dataset[pt])
            self.size['total'] += self.size[pt] + len(multigrams[pt])

        # balance validation set (up to 10% of the dataset size)
        for pt in self.partitions[:-1]:
            missing_items = int(self.size['total'] * 0.1) - self.size['valid']

            if missing_items <= 2:
                break

            i = np.random.choice(np.arange(0, len(multigrams[pt]) - 1), missing_items // 2)
            self.dataset['valid'] += [multigrams[pt].pop(i) for i in sorted(i, reverse=True)]

        # multigrams into train set
        self.dataset['train'] += multigrams['train'] + multigrams['valid'] + multigrams['test']
        self.size['total'] = 0

        # update dataset partition sizes (shuffle is also applied)
        for pt in self.partitions:
            np.random.shuffle(self.dataset[pt])

            self.size[pt] = len(self.dataset[pt])
            self.size['total'] += self.size[pt]

    def _bea2019(self):
        """BEA2019 dataset reader"""

        basedir = os.path.join(self.source, "m2")
        m2_list = next(os.walk(basedir))[2]
        lines = []

        for m2_file in m2_list:
            lines.extend(read_from_m2(os.path.join(basedir, m2_file)))

        return lines

    def _bentham(self):
        """Bentham dataset reader"""

        source = os.path.join(self.source, "BenthamDatasetR0-GT")
        pt_path = os.path.join(source, "Partitions")

        paths = {"train": open(os.path.join(pt_path, "TrainLines.lst")).read().splitlines(),
                 "valid": open(os.path.join(pt_path, "ValidationLines.lst")).read().splitlines(),
                 "test": open(os.path.join(pt_path, "TestLines.lst")).read().splitlines()}

        transcriptions = os.path.join(source, "Transcriptions")
        gt = os.listdir(transcriptions)
        gt_dict, dataset = dict(), dict()

        for index, x in enumerate(gt):
            text = " ".join(open(os.path.join(transcriptions, x)).read().splitlines())
            text = html.unescape(text).replace("<gap/>", "")
            gt_dict[os.path.splitext(x)[0]] = " ".join(text.split())

        for i in self.partitions:
            dataset[i] = [gt_dict[x] for x in paths[i]]

        return dataset

    def _conll13(self):
        """CONLL13 dataset reader"""

        m2_file = os.path.join(self.source, "revised", "data", "official-preprocessed.m2")

        return read_from_m2(m2_file)

    def _conll14(self):
        """CONLL14 dataset reader"""

        m2_file = os.path.join(self.source, "alt", "official-2014.combined-withalt.m2")

        return read_from_m2(m2_file)

    def _google(self):
        """
        Google 1-Billion dataset reader.
        In this project, the google dataset only get 1M data from English and French partitions.
        """

        basedir = os.path.join(self.source)
        m2_list = next(os.walk(basedir))[2]
        lines_en, lines_fr = [], []

        for m2_file in m2_list:
            if "2010" in m2_file and ".en" in m2_file:
                with open(os.path.join(basedir, m2_file)) as f:
                    lines_en = list(set([line for line in f]))[::-1]

            elif "2009" in m2_file and ".fr" in m2_file:
                with open(os.path.join(basedir, m2_file)) as f:
                    lines_fr = list(set([line for line in f]))[::-1]

        # English and french will be 1% samples.
        lines_en = lines_en[:int(len(lines_en) * 0.01)]
        lines_fr = lines_fr[:int(len(lines_fr) * 0.01)]

        return (lines_en + lines_fr)

    def _iam(self):
        """IAM dataset reader"""

        pt_path = os.path.join(self.source, "largeWriterIndependentTextLineRecognitionTask")

        paths = {"train": open(os.path.join(pt_path, "trainset.txt")).read().splitlines(),
                 "valid": open(os.path.join(pt_path, "validationset1.txt")).read().splitlines(),
                 "test": open(os.path.join(pt_path, "testset.txt")).read().splitlines()}

        lines = open(os.path.join(self.source, "ascii", "lines.txt")).read().splitlines()
        gt_dict, dataset = dict(), dict()

        for line in lines:
            if (not line or line[0] == "#"):
                continue

            split = line.split()

            if split[1] == "ok":
                gt_dict[split[0]] = " ".join(split[8::]).replace("|", " ")

        for i in self.partitions:
            dataset[i] = [gt_dict[x] for x in paths[i] if x in gt_dict.keys()]

        return dataset

    def _rimes(self):
        """Rimes dataset reader"""

        def generate(xml, paths, validation=False):
            xml = ET.parse(os.path.join(self.source, xml)).getroot()
            dt = []

            for page_tag in xml:
                for i, line_tag in enumerate(page_tag.iter("Line")):
                    text = html.unescape(line_tag.attrib['Value'])
                    dt.append(" ".join(text.split()))

            if validation:
                index = int(len(dt) * 0.9)
                paths['valid'] = dt[index:]
                paths['train'] = dt[:index]
            else:
                paths['test'] = dt

        dataset, paths = dict(), dict()
        generate("training_2011.xml", paths, validation=True)
        generate("eval_2011_annotated.xml", paths, validation=False)

        for i in self.partitions:
            dataset[i] = [x for x in paths[i]]

        return dataset

    def _saintgall(self):
        """Saint Gall dataset reader"""

        pt_path = os.path.join(self.source, "sets")

        paths = {"train": open(os.path.join(pt_path, "train.txt")).read().splitlines(),
                 "valid": open(os.path.join(pt_path, "valid.txt")).read().splitlines(),
                 "test": open(os.path.join(pt_path, "test.txt")).read().splitlines()}

        lines = open(os.path.join(self.source, "ground_truth", "transcription.txt")).read().splitlines()
        gt_dict = dict()

        for line in lines:
            split = line.split()
            split[1] = split[1].replace("-", "").replace("|", " ")
            gt_dict[split[0]] = split[1]

        img_path = os.path.join(self.source, "data", "line_images_normalized")
        dataset = dict()

        for i in self.partitions:
            dataset[i] = []

            for line in paths[i]:
                glob_filter = os.path.join(img_path, f"{line}*")
                img_list = [x for x in glob(glob_filter, recursive=True)]

                for line in img_list:
                    line = os.path.splitext(os.path.basename(line))[0]
                    dataset[i].append(gt_dict[line])

        return dataset

    def _washington(self):
        """Washington dataset reader"""

        pt_path = os.path.join(self.source, "sets", "cv1")

        paths = {"train": open(os.path.join(pt_path, "train.txt")).read().splitlines(),
                 "valid": open(os.path.join(pt_path, "valid.txt")).read().splitlines(),
                 "test": open(os.path.join(pt_path, "test.txt")).read().splitlines()}

        lines = open(os.path.join(self.source, "ground_truth", "transcription.txt")).read().splitlines()
        gt_dict, dataset = dict(), dict()

        for line in lines:
            split = line.split()
            split[1] = split[1].replace("-", "").replace("|", " ")
            split[1] = split[1].replace("s_pt", ".").replace("s_cm", ",")
            split[1] = split[1].replace("s_mi", "-").replace("s_qo", ":")
            split[1] = split[1].replace("s_sq", ";").replace("s_et", "V")
            split[1] = split[1].replace("s_bl", "(").replace("s_br", ")")
            split[1] = split[1].replace("s_qt", "'").replace("s_GW", "G.W.")
            split[1] = split[1].replace("s_", "")
            gt_dict[split[0]] = split[1]

        for i in self.partitions:
            dataset[i] = [gt_dict[i] for i in paths[i]]

        return dataset

    @staticmethod
    def check_text(text):
        """Make sure text has more characters instead of punctuation marks"""

        strip_punc = text.strip(string.punctuation).strip()
        no_punc = text.translate(str.maketrans("", "", string.punctuation)).strip()

        if len(text) == 0 or len(strip_punc) == 0 or len(no_punc) == 0:
            return False

        punc_percent = (len(strip_punc) - len(no_punc)) / len(strip_punc)

        return len(no_punc) >= 2 and punc_percent <= 0.1


def read_from_txt(file_name):
    """Read the M2 file and return labels and sentences (ground truth and data)"""

    train = {"dt": [], "gt": []}
    valid = {"dt": [], "gt": []}
    test = {"dt": [], "gt": []}

    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")

    for item in lines:
        arr = item.split()

        if len(arr) == 0:
            continue

        x = " ".join(arr[1::])

        if arr[0] == "TR_L":
            train['gt'].append(x)
            train['dt'].append("")
        elif arr[0] == "TR_P":
            train['dt'][-1] = x

        if arr[0] == "VA_L":
            valid['gt'].append(x)
            valid['dt'].append("")
        elif arr[0] == "VA_P":
            valid['dt'][-1] = x

        if arr[0] == "TE_L":
            test['gt'].append(x)
            test['dt'].append("")
        elif arr[0] == "TE_P":
            test['dt'][-1] = x

    dt = {"train": train, "valid": valid, "test": test}

    return dt


def read_from_m2(file_name):
    """
    Read the M2 file and return the sentences with the corrections.

    Tool to apply text error correction annotations in m2 format, available:
    URL: https://github.com/samueljamesbell/m2-correct
    """

    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")

    sentences, corrections = parse(lines)
    corrected = []

    for s, c in zip(sentences, corrections):
        coor = apply_corrections(s, c[0])

        if len(coor) > 0:
            corrected.append(coor)

    return corrected


def apply_corrections(sentence, corrections):
    """
    Return a new sentence with corrections applied.
    Sentence should be a whitespace-separated tokenised string. Corrections
    should be a list of corrections.
    """

    tokens = sentence.split()
    offset = 0

    for c in corrections:
        tokens, offset = _apply_correction(tokens, c, offset)

    return " ".join(tokens)


def _apply_correction(tokens, correction, offset):
    """Apply a single correction to a list of tokens"""

    start_token_offset, end_token_offset, _, insertion = correction
    to_insert = insertion[0].split(" ")
    end_token_offset += (len(to_insert) - 1)

    to_insert_filtered = [t for t in to_insert if t != ""]

    head = tokens[:start_token_offset + offset]
    tail = tokens[end_token_offset + offset:]

    new_tokens = head + to_insert_filtered + tail

    new_offset = len(to_insert_filtered) - (end_token_offset - start_token_offset) + offset

    return new_tokens, new_offset


"""
The `parse` and `paragraphs` functions are modifications
of code from the NUS M2 scorer (GNU GPL v2.0 license), available here:
https://github.com/nusnlp/m2scorer/

Below is the original preamble:

This file is part of the NUS M2 scorer.
The NUS M2 scorer is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The NUS M2 scorer is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""


def parse(lines):
    source_sentences = []
    gold_edits = []

    for item in paragraphs(lines):
        sentence = [line[2:].strip() for line in item if line.startswith("S ")]
        assert sentence != []
        annotations = {}

        for line in item[1:]:
            if line.startswith("I ") or line.startswith("S "):
                continue
            assert line.startswith("A ")

            line = line[2:]
            fields = line.split("|||")
            start_offset = int(fields[0].split()[0])
            end_offset = int(fields[0].split()[1])
            etype = fields[1]

            if etype == "noop":
                start_offset = -1
                end_offset = -1

            corrections = [c.strip() if c != "-NONE-" else "" for c in fields[2].split("||")]
            original = " ".join(" ".join(sentence).split()[start_offset:end_offset])
            annotator = int(fields[5])

            if annotator not in annotations.keys():
                annotations[annotator] = []

            annotations[annotator].append((start_offset, end_offset, original, corrections))

        tok_offset = 0

        for this_sentence in sentence:
            tok_offset += len(this_sentence.split())
            source_sentences.append(this_sentence)
            this_edits = {}

            for annotator, annotation in annotations.items():
                this_edits[annotator] = [edit for edit in annotation if edit[0] <=
                                         tok_offset and edit[1] <= tok_offset and edit[0] >= 0 and edit[1] >= 0]

            if len(this_edits) == 0:
                this_edits[0] = []

            gold_edits.append(this_edits)

    return (source_sentences, gold_edits)


def paragraphs(lines):
    paragraph = []

    for line in lines:
        if line == "":
            if paragraph:
                yield paragraph
                paragraph = []
        else:
            paragraph.append(line)
