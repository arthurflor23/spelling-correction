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
RE_RESERVED_CHAR_FILTER = re.compile(r'[·¶«œ»]', re.UNICODE)
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


def add_noise(sentences, max_text_length, amount_noise=0.5):
    """Generate some artificial spelling mistakes (or not) in the sentences"""

    chars = list(" " + string.ascii_letters + string.digits)
    n_sentences = []

    assert(0.0 <= amount_noise <= 1.0)

    for x in sentences:
        if len(x) >= 3:
            prob = 0.1 if len(x) <= 5 else amount_noise

            if np.random.rand() <= prob:
                # Replace characters...
                sentence = x

                if np.random.rand() <= 0.5:
                    # by accentuation
                    x = unicodedata.normalize("NFKD", x).encode("ASCII", "ignore").decode("ASCII")

                if sentence == x:
                    # by random characters
                    random_index = np.random.randint(len(x))
                    x = x[:random_index] + np.random.choice(chars) + x[random_index + 1:]

            if np.random.rand() <= prob:
                # Delete characters...
                sentence = x

                if np.random.rand() <= 0.5:
                    # by repeat characters
                    x = re.compile(r'(.)\1{1,}', re.IGNORECASE).sub(r'\1', x)

                if sentence == x:
                    # by random characters
                    random_index = np.random.randint(len(x))
                    x = x[:random_index] + x[random_index + 1:]

            if np.random.rand() <= prob:
                # Add a random character
                random_index = np.random.randint(len(x))
                x = x[:random_index] + np.random.choice(chars) + x[random_index:]

            if np.random.rand() <= prob:
                # Transpose 2 random characters
                random_index = np.random.randint(len(x) - 1)
                x = x[:random_index] + x[random_index + 1] + x[random_index] + x[random_index + 2:]

        x = text_standardize(x)
        n_sentences.append(x)

    return n_sentences
