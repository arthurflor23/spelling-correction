"""Manager data preprocess"""

import re
import html
import string
import unicodedata
import numpy as np

"""
DeepSpell based text cleaning process.
    Tal Weiss.
    Deep Spelling.
    Medium: https://machinelearnings.co/deep-spelling-9ffef96a24f6#.2c9pu8nlm
    Github: https://github.com/MajorTal/DeepSpell
"""
NORMALIZE_WHITESPACE_REGEX = re.compile(r'[^\S\n]+', re.UNICODE)
RE_DASH_FILTER = re.compile(r'[\-\˗\֊\‐\‑\‒\–\—\⁻\₋\−\﹣\－]', re.UNICODE)
RE_APOSTROPHE_FILTER = re.compile(r'&#39;|[ʼ՚＇‘’‛❛❜ߴߵ`‵´ˊˋ{}{}{}{}{}{}{}{}{}]'.format(chr(768), chr(769),
                                                                                      chr(832), chr(833),
                                                                                      chr(2387), chr(5151),
                                                                                      chr(5152), chr(65344),
                                                                                      chr(8242)), re.UNICODE)
RE_RESERVED_CHAR_FILTER = re.compile(r'[¶¤«œ»]', re.UNICODE)
RE_LEFT_PARENTH_FILTER = re.compile(r'[\(\[\{\⁽\₍\❨\❪\﹙\（]', re.UNICODE)
RE_RIGHT_PARENTH_FILTER = re.compile(r'[\)\]\}\⁾\₎\❩\❫\﹚\）]', re.UNICODE)
RE_BASIC_CLEANER = re.compile(r'[^\w\s{}]'.format(re.escape(string.punctuation)), re.UNICODE)


def text_standardize(txt):
    """Organize/add spaces around punctuation marks"""

    if txt is None:
        return ""

    txt = html.unescape(txt).replace("\\n", "").replace("\\t", "")

    txt = RE_RESERVED_CHAR_FILTER.sub("", txt)
    txt = RE_DASH_FILTER.sub("-", txt)
    txt = RE_APOSTROPHE_FILTER.sub("'", txt)
    txt = RE_LEFT_PARENTH_FILTER.sub("(", txt)
    txt = RE_RIGHT_PARENTH_FILTER.sub(")", txt)
    txt = RE_BASIC_CLEANER.sub("", txt)

    txt = txt.translate(str.maketrans({c: f" {c} " for c in string.punctuation}))
    txt = NORMALIZE_WHITESPACE_REGEX.sub(" ", txt.strip())

    return txt


def split_by_max_length(sentence, max_text_length=128):
    """Standardize n_sentences: split long n_sentences into max_text_length"""

    tolerance = 5
    new_n_sentences = []

    if len(sentence) < max_text_length - tolerance:
        new_n_sentences.append(sentence)
    else:
        splitted = sentence.split()
        text = []

        for x in splitted:
            text_temp = " ".join(text)

            if len(text_temp) + len(x) < max_text_length:
                text.append(x)
            else:
                new_n_sentences.append(text_temp)
                text = [x]

        text_temp = " ".join(text)

        if len(text_temp) > tolerance:
            new_n_sentences.append(text_temp)

    return new_n_sentences


def generate_ngram_sentences(sentence):
    """
    Generate sentences combinations (like ngrams).
    i.e.:
    original sentence: I like code .
        > sentence 1 : I like
        > sentence 2 : I like code .
        > sentence 3 : like
        > sentence 4 : like code .
        > sentence 5 : code .
    """

    tokens = sentence.split()
    ngrams = []

    for y in range(len(tokens)):
        new_sentence = True
        support_text = ""

        for x in range(y, len(tokens)):
            if len(tokens[x]) < 3 and not sentence.endswith(tokens[x]):
                support_text += f" {tokens[x]}"
                continue

            last = ""
            if x > y and len(ngrams) > 0 and not new_sentence:
                last = ngrams[-1]

            ngrams.append(f"{last}{support_text} {tokens[x]}".strip())
            new_sentence = False
            support_text = ""

    return ngrams


def add_noise(x, max_text_length, max_prob=0.9, iterations=9):
    """Generate some artificial spelling mistakes (or not) in the sentences"""

    assert(1 <= iterations)
    assert(0.0 <= max_prob <= 1.0)

    chars = list(f"{string.ascii_letters}{string.digits} .")
    np.random.shuffle(chars)
    sentences = x.copy()

    for i, s in enumerate(sentences):
        prob = len(s) * (max_prob / max_text_length)

        for _ in range(iterations):
            if len(s) <= 2:
                continue

            if np.random.rand() <= prob:
                # Replace characters...
                sentence = s

                if np.random.rand() <= 0.5:
                    # by accentuation
                    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")

                if sentence == s:
                    # by random characters
                    random_index = np.random.randint(len(s))
                    s = s[:random_index] + np.random.choice(chars) + s[random_index + 1:]

            if np.random.rand() <= prob:
                # Delete characters...
                sentence = s

                if np.random.rand() <= 0.5:
                    # by repeat characters
                    s = re.compile(r'(.)\1{1,}', re.IGNORECASE).sub(r'\1', s)

                if sentence == s:
                    # by random characters
                    random_index = np.random.randint(len(s))
                    s = s[:random_index] + s[random_index + 1:]

            if np.random.rand() <= prob:
                # Add a random character
                random_index = np.random.randint(len(s))
                s = s[:random_index] + np.random.choice(chars) + s[random_index:]

            if np.random.rand() <= prob:
                # Transpose 2 random characters
                random_index = np.random.randint(len(s) - 1)
                s = s[:random_index] + s[random_index + 1] + s[random_index] + s[random_index + 2:]

        sentences[i] = text_standardize(s)

    return sentences
