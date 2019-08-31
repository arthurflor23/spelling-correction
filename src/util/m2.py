"""
Tool to apply text error correction annotations in m2 format, available here:
https://github.com/samueljamesbell/m2-correct
"""


def read_dataset(file_name):
    """
    Read the M2 file and return labels and sentences (load dataset transformed).
    """

    train, valid, test = dict(), dict(), dict()
    train["lab"], train["sen"] = [], []
    valid["lab"], valid["sen"] = [], []
    test["lab"], test["sen"] = [], []

    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")

    for item in lines:
        if item.startswith("TR_L "):
            train["lab"].append(item[5:].strip())
            train["sen"].append(None)
        elif item.startswith("VA_L "):
            valid["lab"].append(item[5:].strip())
        elif item.startswith("VA_S "):
            valid["sen"].append(item[5:].strip())
        elif item.startswith("TE_L "):
            test["lab"].append(item[5:].strip())
        elif item.startswith("TE_S "):
            test["sen"].append(item[5:].strip())

    dt = dict()
    dt["train"], dt["valid"], dt["test"] = train, valid, test

    return dt


def read_raw(file_name):
    """
    Read the M2 file and return the sentences with the corrections.
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
    """Apply a single correction to a list of tokens."""

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
