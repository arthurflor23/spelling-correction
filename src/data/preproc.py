"""Methods to help manager data development."""

import re
import string
import numpy as np


def encode_ctc(text, charset, max_text_length):
    """Encode text array (sparse)."""

    if not isinstance(text, list):
        text = [text]

    pad_encoded = np.zeros(shape=(max_text_length, max_text_length))

    for index, item in enumerate(text):
        encoded = [float(charset.find(x)) for x in item if charset.find(x) > -1]
        encoded = [float(charset.find("&"))] if len(encoded) == 0 else encoded
        pad_encoded[index][0:len(encoded)] = encoded

    return pad_encoded


def decode_onehot(text, charset, reverse=False):
    """Decode from one-hot."""

    text_encoded = np.array(text).argmax(axis=-1)
    decoded = []

    for index in text_encoded:
        try:
            decoded.append(charset[index])
        except KeyError:
            pass

    if reverse:
        decoded = decoded[::-1]

    return "".join(decoded)


def encode_onehot(text, charset, max_text_length, reverse=False):
    """Encode to one-hot."""

    encoded = np.zeros((max_text_length, len(charset)), dtype=np.bool)

    for i, char in enumerate(text):
        try:
            encoded[i, charset.find(char)] = 1
        except KeyError:
            pass

    if reverse:
        encoded = encoded[::-1]

    return encoded


def shuffle(array):
    """Modify a sequence by shuffling its contents."""

    arange = np.arange(0, len(array))
    np.random.shuffle(arange)
    new_array = []

    for i in range(len(arange)):
        new_array.append(array[i])
        array[i] = None

    return new_array


def parse_sentence(text, splitted=False):
    """Remove punctuation marks."""

    matches = re.findall(r"(([^\W_]|['’])+)", text)
    matches = [match[0] for match in matches]

    if splitted:
        return matches

    return " ".join(matches)


def normalize_text(sentences, charset, max_text_length):
    """Normalize sentences: replace some stuffs and split if necessary."""

    text_list = []
    min_text_length = 3
    max_text_length -= (min_text_length * 2)

    for i in range(len(sentences)):
        sentences[i] = "".join([y for y in sentences[i] if y in charset])
        sentences[i] = padding_space(sentences[i])

        if len(sentences[i]) < min_text_length:
            continue

        if len(sentences[i]) < max_text_length:
            text_list.append(sentences[i])
            sentences[i] = None
        else:
            splitted = sentences[i].split()
            sentences[i] = None
            text = []

            for x in splitted:
                if len(" ".join(text)) + len(x) < max_text_length:
                    text.append(x)
                else:
                    text_list.append(" ".join(text))
                    text = [x]

            if len(text) >= min_text_length:
                text_list.append(" ".join(text))

    return text_list


def padding_space(text):
    """Organize/add spaces around punctuation marks."""

    text = text.replace("«", "").replace("»", "").replace("“", "\"")
    text = text.replace("æ", "").replace("ø", "").replace("ß", "")
    text = text.replace(" '", "").replace("'s", "s").replace("'", "")

    for y in text:
        if y in string.punctuation:
            text = text.replace(y, f" {y} ")

    return " ".join(text.split())


"""
Method to apply text random noise error (adapted):
    Author: Tal Weiss
    Title: Deep Spelling, 2016
    Article: https://machinelearnings.co/deep-spelling-9ffef96a24f6
    Repository URL: https://github.com/MajorTal/DeepSpell
"""


def add_noise(batch, max_text_length):
    """Add some artificial spelling mistakes to the string"""

    amount_of_noise = 0.2 * max_text_length
    charset = list(string.whitespace[0] + string.digits + string.ascii_letters)

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

        batch[i] = padding_space(batch[i])

    return batch
