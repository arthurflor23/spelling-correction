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

RE_DASH_FILTER = re.compile(r'[\-\˗\֊\‐\‑\‒\–\—\⁻\₋\−\﹣\－]', re.UNICODE)
RE_APOSTROPHE_FILTER = re.compile(r'&#39;|[ʼ՚＇‘’‛❛❜ߴߵ`‵´ˊˋ{}{}{}{}{}{}{}{}{}]'.format(
    chr(768), chr(769), chr(832), chr(833), chr(2387),
    chr(5151), chr(5152), chr(65344), chr(8242)), re.UNICODE)
RE_RESERVED_CHAR_FILTER = re.compile(r'[¶¤«»]', re.UNICODE)
RE_LEFT_PARENTH_FILTER = re.compile(r'[\(\[\{\⁽\₍\❨\❪\﹙\（]', re.UNICODE)
RE_RIGHT_PARENTH_FILTER = re.compile(r'[\)\]\}\⁾\₎\❩\❫\﹚\）]', re.UNICODE)
RE_BASIC_CLEANER = re.compile(r'[^\w\s{}]'.format(re.escape(string.punctuation)), re.UNICODE)

LEFT_PUNCTUATION_FILTER = """!%&),.:;<=>?@\\]^_`|}~"""
RIGHT_PUNCTUATION_FILTER = """"(/<=>@[\\^_`{|~"""
NORMALIZE_WHITESPACE_REGEX = re.compile(r'[^\S\n]+', re.UNICODE)


def text_standardize(text):
    """Organize/add spaces around punctuation marks"""

    if text is None:
        return ""

    text = html.unescape(text).replace("\\n", "").replace("\\t", "")

    text = RE_RESERVED_CHAR_FILTER.sub("", text)
    text = RE_DASH_FILTER.sub("-", text)
    text = RE_APOSTROPHE_FILTER.sub("'", text)
    text = RE_LEFT_PARENTH_FILTER.sub("(", text)
    text = RE_RIGHT_PARENTH_FILTER.sub(")", text)
    text = RE_BASIC_CLEANER.sub("", text)

    text = text.lstrip(LEFT_PUNCTUATION_FILTER)
    text = text.rstrip(RIGHT_PUNCTUATION_FILTER)
    text = text.translate(str.maketrans({c: f" {c} " for c in string.punctuation}))
    text = NORMALIZE_WHITESPACE_REGEX.sub(" ", text.strip())

    return text


def split_by_max_length(sentence, max_text_length=128):
    """Standardize n_sentences: split long n_sentences into max_text_length"""

    if len(sentence) < max_text_length:
        return [sentence]

    splitted = sentence.split()
    new_n_sentences, text = [], []

    for x in splitted:
        support_text = " ".join(text)

        if len(support_text) + len(x) < max_text_length:
            text.append(x)
        else:
            new_n_sentences.append(support_text)
            text = [x]

    text = " ".join(text)

    if len(text) > 2:
        new_n_sentences.append(text)

    return new_n_sentences


def generate_multigrams(sentence):
    """
    Generate n-grams of the sentence.
    i.e.:
    original sentence: I like code .
        > sentence 1 : I like
        > sentence 2 : I like code .
        > sentence 3 : like
        > sentence 4 : like code .
        > sentence 5 : code .
    """

    tokens = sentence.split()
    tk_length = len(tokens)
    multigrams = []

    for y in range(tk_length):
        new_sentence = True
        support_text = ""

        for x in range(y, tk_length):
            if y == 0 and tk_length > 2 and x == (tk_length - 1):
                continue

            if len(tokens[x]) <= 2 and tokens[x] != tokens[-1]:
                support_text += f" {tokens[x]}"
                continue

            last = ""
            if x > y and len(multigrams) > 0 and not new_sentence:
                last = multigrams[-1]

            multigrams.append(f"{last}{support_text} {tokens[x]}".strip())
            new_sentence = False
            support_text = ""

    return multigrams


def add_noise(x, max_text_length, ratio=0.8, iterations=4):
    """Generate some artificial spelling mistakes in the sentences"""

    assert(0 < ratio <= 1)
    assert(iterations > 0)

    chars = list(" ." + string.digits + string.ascii_letters)
    sentences = x.copy()

    for i, s in enumerate(sentences):
        prob = len(s) * (ratio / max_text_length)

        for _ in range(iterations):
            if len(s) <= 4:
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
                # Add a random character
                random_index = np.random.randint(len(s))
                s = s[:random_index] + np.random.choice(chars) + s[random_index:]

            if np.random.rand() <= prob:
                # Transpose 2 random characters
                random_index = np.random.randint(len(s) - 1)
                s = s[:random_index] + s[random_index + 1] + s[random_index] + s[random_index + 2:]

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

        sentences[i] = text_standardize(s)

    return sentences
