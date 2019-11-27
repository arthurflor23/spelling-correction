"""Dataset reader and process"""

import os
import re
import html
import string
import numpy as np
import xml.etree.ElementTree as ET

from glob import glob
from data import preproc as pp


class Dataset():

    def __init__(self, source, names, min_text_len=5):
        self.source = source
        self.names = names
        self.min_text_len = min_text_len

        self.partitions = {"train": [], "valid": [], "test": []}
        self.size = {"train": 0, "valid": 0, "test": 0, "total": 0}

    def read_lines(self, maxlen):
        """Read sentences from dataset and preprocess"""

        for dataset in self.names:
            print(f"The {dataset} dataset will be transformed...")
            lines = getattr(self, f"_{dataset}")()

            # split sentences by max length
            lines = [y for x in lines for y in pp.split_by_max_length(x, maxlen)]

            # standardize sentences
            lines = [pp.text_standardize(x) for x in lines]

            # generate ngrams from senteces
            ngrams = [pp.generate_ngram_sentences(x) for x in lines]

            # add random ngrams into valid data and test data.
            # finally, add last ngrams + original lines into train data
            for i, ngram in enumerate(ngrams):
                y = sorted([x for x in ngram if self.check_text(x)], key=len)

                if len(y) > 2:
                    factor = int(len(y) * 0.1)
                    arange = np.random.choice(np.arange(int(len(y) * 0.25), int(len(y) * 0.75)), factor)
                    self.partitions['valid'].extend([y.pop(i) for i in sorted(arange, reverse=True)])

                if len(y) > 1:
                    arange = np.random.choice(np.arange(int(len(y) * 0.25), int(len(y) * 0.75)), 1)
                    self.partitions['test'].extend([y.pop(i) for i in sorted(arange, reverse=True)])

                self.partitions['train'].extend(y + [lines[i]])

            for pt in self.partitions.keys():
                self.partitions[pt] = list(set(self.partitions[pt]))
                self.size[pt] = len(self.partitions[pt])
                self.size['total'] += self.size[pt]

            # # generate ngrams from senteces and shuffle
            # ngrams = [y for x in lines for y in pp.generate_ngram_sentences(x)]
            # np.random.shuffle(ngrams)

            # # ignore duplicate items and ngrams with more punctuation percent
            # ngrams = list(set([x for x in ngrams if self.check_text(x)]))

            # # get 10% random ngrams into valid data
            # arange = np.random.choice(np.arange(0, len(ngrams) - 1), int(len(ngrams) * 0.1))
            # self.partitions['valid'] = [ngrams.pop(i) for i in sorted(arange, reverse=True)]

            # # get 1% random ngrams into valid data
            # arange = np.random.choice(np.arange(0, len(ngrams) - 1), int(len(ngrams) * 0.01))
            # self.partitions['test'] = [ngrams.pop(i) for i in sorted(arange, reverse=True)]

            # # get original lines and last ngrams into train data
            # self.partitions['train'] = lines + ngrams
            # np.random.shuffle(self.partitions['train'])

        # self.size['train'] = len(self.partitions['train'])
        # self.size['valid'] = len(self.partitions['valid'])
        # self.size['test'] = len(self.partitions['test'])
        # self.size['total'] = self.size['train'] + self.size['valid'] + self.size['test']

    def check_text(self, text):
        """Make sure text has more characters instead of punctuation marks"""

        x = text.translate(str.maketrans("", "", string.punctuation))
        x = re.compile(r'[^\S\n]+', re.UNICODE).sub(" ", x.strip())

        return len(x) > 1 and len(x) > len(text) * 0.5

    def _bea2019(self):
        """BEA2019 dataset reader"""

        basedir = os.path.join(self.source, "bea2019", "m2")
        m2_list = next(os.walk(basedir))[2]
        lines = []

        for m2_file in m2_list:
            lines.extend(read_from_m2(os.path.join(basedir, m2_file)))

        return list(set(lines))

    def _bentham(self):
        """Bentham dataset reader"""

        basedir = os.path.join(self.source, "bentham", "BenthamDatasetR0-GT")
        transdir = os.path.join(basedir, "Transcriptions")
        files = os.listdir(transdir)

        ptdir = os.path.join(basedir, "Partitions")
        lines, images = [], []

        for x in ['TrainLines.lst', 'ValidationLines.lst', 'TestLines.lst']:
            images.extend(open(os.path.join(ptdir, x)).read().splitlines())

        for item in files:
            text = open(os.path.join(transdir, item)).read().splitlines()[0]
            text = html.unescape(text).replace("<gap/>", "")

            if len(text) > self.min_text_len and os.path.splitext(item)[0] in images:
                lines.append(text)

        return list(set(lines))

    def _conll13(self):
        """CONLL13 dataset reader"""

        m2_file = os.path.join(self.source, "conll13", "revised", "data", "official-preprocessed.m2")

        return list(set(read_from_m2(m2_file)))

    def _conll14(self):
        """CONLL14 dataset reader"""

        m2_file = os.path.join(self.source, "conll14", "alt", "official-2014.combined-withalt.m2")

        return list(set(read_from_m2(m2_file)))

    def _google(self):
        """
        Google 1-Billion dataset reader.
        In this project, the google dataset only get 1M data from English and French partitions.
        """

        basedir = os.path.join(self.source, "google")
        m2_list = next(os.walk(basedir))[2]
        lines_en, lines_fr = [], []

        for m2_file in m2_list:
            if "2010" in m2_file and ".en" in m2_file:
                with open(os.path.join(basedir, m2_file)) as f:
                    lines_en = [line for line in f if len(line) > self.min_text_len]
                    lines_en = list(set(lines_en))[::-1]

            elif "2009" in m2_file and ".fr" in m2_file:
                with open(os.path.join(basedir, m2_file)) as f:
                    lines_fr = [line for line in f if len(line) > self.min_text_len]
                    lines_fr = list(set(lines_fr))[::-1]

        # English and french will be 1% samples.
        lines_en = lines_en[:int(len(lines_en) * 0.01)]
        lines_fr = lines_fr[:int(len(lines_fr) * 0.01)]
        lines = lines_en + lines_fr

        del lines_en, lines_fr
        np.random.shuffle(lines)

        return lines

    def _iam(self):
        """IAM dataset reader"""

        basedir = os.path.join(self.source, "iam")
        files = open(os.path.join(basedir, "ascii", "lines.txt")).read().splitlines()

        ptdir = os.path.join(basedir, "largeWriterIndependentTextLineRecognitionTask")
        lines, images = [], []

        for x in ['trainset.txt', 'validationset1.txt', 'validationset2.txt', 'testset.txt']:
            images.extend(open(os.path.join(ptdir, x)).read().splitlines())

        for item in files:
            if (not item or item[0] == "#"):
                continue

            splitted = item.split()

            if splitted[1] == "ok" and splitted[0] in images:
                text = " ".join(splitted[8::]).replace("|", " ")

                if len(text) > self.min_text_len:
                    lines.append(text)

        return list(set(lines))

    def _rimes(self):
        """Rimes dataset reader"""

        def read_from_xml(xml_file):
            xml_file = ET.parse(xml_file).getroot()
            lines = []

            for page_tag in xml_file:
                for _, line_tag in enumerate(page_tag.iter("Line")):
                    text = " ".join(html.unescape(line_tag.attrib['Value']).split())

                    if len(text) > self.min_text_len:
                        lines.append(text)

            return lines

        basedir = os.path.join(self.source, "rimes")
        lines = []

        for f in ['training_2011.xml', 'eval_2011_annotated.xml']:
            lines.extend(read_from_xml(os.path.join(basedir, f)))

        return list(set(lines))

    def _saintgall(self):
        """Saint Gall dataset reader"""

        basedir = os.path.join(self.source, "saintgall")
        files = open(os.path.join(basedir, "ground_truth", "transcription.txt")).read().splitlines()

        ptdir = os.path.join(basedir, "sets")
        lines, pages, images = [], [], []

        for x in ['train.txt', 'valid.txt', 'test.txt']:
            pages.extend(open(os.path.join(ptdir, x)).read().splitlines())

        for x in pages:
            glob_filter = os.path.join(basedir, "data", "line_images_normalized", f"{x}*")
            files_list = [x for x in glob(glob_filter, recursive=True)]

            for y in files_list:
                images.append(os.path.splitext(os.path.basename(y))[0])

        for item in files:
            splitted = item.split()
            splitted[1] = splitted[1].replace("-", "").replace("|", " ")

            if len(splitted[1]) > self.min_text_len and splitted[0] in images:
                lines.append(splitted[1])

        return list(set(lines))

    def _washington(self):
        """Washington dataset reader"""

        basedir = os.path.join(self.source, "washington")
        files = open(os.path.join(basedir, "ground_truth", "transcription.txt")).read().splitlines()

        ptdir = os.path.join(basedir, "sets", "cv1")
        lines, images = [], []

        for x in ['train.txt', 'valid.txt', 'test.txt']:
            images.extend(open(os.path.join(ptdir, x)).read().splitlines())

        for item in files:
            splitted = item.split()
            splitted[1] = splitted[1].replace("-", "").replace("|", " ")
            splitted[1] = splitted[1].replace("s_pt", ".").replace("s_cm", ",")
            splitted[1] = splitted[1].replace("s_mi", "-").replace("s_qo", ":")
            splitted[1] = splitted[1].replace("s_sq", ";").replace("s_et", "V")
            splitted[1] = splitted[1].replace("s_bl", "(").replace("s_br", ")")
            splitted[1] = splitted[1].replace("s_qt", "'").replace("s_", "")

            if len(splitted[1]) > self.min_text_len and splitted[0] in images:
                lines.append(splitted[1])

        return list(set(lines))


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
