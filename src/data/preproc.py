"""Methods to help manager data development"""

import string
import numpy as np


def shuffle(array):
    """Modify a sequence by shuffling its contents"""

    arange = np.arange(0, len(array))
    np.random.shuffle(arange)
    new_array = []

    for i in range(len(arange)):
        new_array.append(array[i])
        array[i] = None

    return new_array


def padding_space(text):
    """Organize/add spaces around punctuation marks"""

    text = " ".join(text.split()).replace(" '", "'").replace("' ", "'")
    text = text.replace("«", "").replace("»", "")

    for y in text:
        if y in string.punctuation.replace("'", ""):
            text = text.replace(y, f" {y} ")

    return " ".join(text.split())


def standardize(sentences, charset, max_text_length):
    """Standardize sentences: replace some stuffs and split if necessary"""

    text_list = []
    min_text_length = 3

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


"""
Method to apply text random noise error (adapted):
    Author: Tal Weiss
    Title: Deep Spelling, 2016
    Article: https://machinelearnings.co/deep-spelling-9ffef96a24f6
    Repository URL: https://github.com/MajorTal/DeepSpell
"""


def add_noise(batch, charset=None, max_text_length=128, level=3):
    """Add some artificial spelling mistakes to the string"""

    charset = charset if charset else string.printable[:-5]
    charset = list(charset)

    for i in range(len(batch)):
        for _ in range(level):
            # Replace a character with a random character
            random_char_position = np.random.randint(len(batch[i]))
            batch[i] = batch[i][:random_char_position] + np.random.choice(charset[:-1]) + batch[i][random_char_position + 1:]

            # Transpose 2 characters
            random_char_position = np.random.randint(len(batch[i]) - 1)
            batch[i] = (batch[i][:random_char_position] + batch[i][random_char_position + 1] +
                        batch[i][random_char_position] + batch[i][random_char_position + 2:])

            # Delete a character
            random_char_position = np.random.randint(len(batch[i]))
            batch[i] = batch[i][:random_char_position] + batch[i][random_char_position + 1:]

            # Add a random character
            if len(batch[i]) < max_text_length:
                random_char_position = np.random.randint(len(batch[i]))
                batch[i] = batch[i][:random_char_position] + np.random.choice(charset[:-1]) + batch[i][random_char_position:]

        batch[i] = padding_space(batch[i])
        batch[i] = batch[i][:max_text_length]

    return batch
