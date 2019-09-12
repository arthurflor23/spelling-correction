"""Methods to help manager data development."""

import re
import string
import numpy as np


def encode(text, charset, max_text_length, matrix=False):
    """Encode text array (sparse)."""

    if not isinstance(text, list):
        text = [text]

    pad_encoded = np.zeros(shape=(max_text_length, max_text_length))

    for index, item in enumerate(text):
        encoded = [float(charset.find(x)) for x in item if charset.find(x) > -1]
        encoded = [float(charset.find("&"))] if len(encoded) == 0 else encoded
        pad_encoded[index][0:len(encoded)] = encoded

    if matrix:
        return pad_encoded

    return pad_encoded[0]


def parse_sentence(text, splitted=False):
    """Remove punctuation marks."""

    matches = re.findall(r"(([^\W_]|['’])+)", text)
    matches = [match[0] for match in matches]

    if splitted:
        return matches

    return " ".join(matches)


def normalize_text(texts, charset, limit):
    """Normalize a batch of texts: replace some stuffs and split sentence when necessary."""

    min_text_length = 3
    limit -= min_text_length
    text_list = []

    if not isinstance(texts, list):
        texts = [texts]

    for i in range(len(texts)):
        texts[i] = "".join([y for y in texts[i] if y in charset])
        texts[i] = organize_space(texts[i])

        if len(texts[i]) < min_text_length:
            continue

        if len(texts[i]) < limit:
            text_list.append(texts[i])
            texts[i] = None
        else:
            splitted = texts[i].split()
            texts[i] = None
            text = []

            for x in splitted:
                if len(" ".join(text)) + len(x) < limit:
                    text.append(x)
                else:
                    text_list.append(" ".join(text))
                    text = [x]

            if len(text) >= min_text_length:
                text_list.append(" ".join(text))

    return text_list


def organize_space(text):
    """Organize/add spaces around punctuation marks."""

    text = text.replace("«", "").replace("»", "").replace("“", "\"")
    text = text.replace(" '", "").replace("'s", "s").replace("'", "")

    for y in text:
        if y in string.punctuation:
            text = text.replace(y, f" {y} ")

    return " ".join(text.split())


"""
Tool to apply text random noise error, available here:
https://gist.github.com/MajorTal/67d54887a729b5e5aa85
"""


def add_noise(batch, max_text_length):
    """Add some artificial spelling mistakes to the string"""

    amount_of_noise = 0.1 * max_text_length
    charset = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

    if not isinstance(batch, list):
        batch = [batch]

    for i in range(len(batch)):
        for _ in range(2):
            # Replace a character with a random character
            if np.random.rand() < amount_of_noise * len(batch[i]):
                random_char_position = np.random.randint(len(batch[i]))
                batch[i] = batch[i][:random_char_position] + np.random.choice(charset[:-1]) + batch[i][random_char_position + 1:]

            # Transpose 2 characters
            if np.random.rand() < amount_of_noise * len(batch[i]):
                random_char_position = np.random.randint(len(batch[i]) - 1)
                batch[i] = (batch[i][:random_char_position] + batch[i][random_char_position + 1] +
                            batch[i][random_char_position] + batch[i][random_char_position + 2:])

        # Add a random character
        if len(batch[i]) < max_text_length and np.random.rand() < amount_of_noise * len(batch[i]):
            random_char_position = np.random.randint(len(batch[i]))
            batch[i] = batch[i][:random_char_position] + np.random.choice(charset[:-1]) + batch[i][random_char_position:]

        # Delete a character
        if np.random.rand() < amount_of_noise * len(batch[i]):
            random_char_position = np.random.randint(len(batch[i]))
            batch[i] = batch[i][:random_char_position] + batch[i][random_char_position + 1:]

        batch[i] = organize_space(batch[i])

    return batch
