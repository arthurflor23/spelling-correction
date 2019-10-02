"""Methods to help manager data development"""

import string
import numpy as np


def padding_punctuation(sentences):
    """Organize/add spaces around punctuation marks"""

    for i in range(len(sentences)):
        sentences[i] = " ".join(sentences[i].split()).replace(" '", "'").replace("' ", "'")
        sentences[i] = sentences[i].replace("«", "").replace("»", "")

        for y in sentences[i]:
            if y in string.punctuation.replace("'", ""):
                sentences[i] = sentences[i].replace(y, f" {y} ")

        sentences[i] = " ".join(sentences[i].split())

    return sentences


def split_by_max_length(sentences, charset=None, max_text_length=100):
    """Standardize sentences: split long sentences into max_text_length"""

    tolerance = 3
    min_text_length = 3
    new_setences = []

    for i in range(len(sentences)):
        x = sentences[i]

        if charset is not None:
            x = "".join([c for c in sentences[i] if c in charset])

        if len(x) < min_text_length:
            continue

        if len(x) < max_text_length - tolerance:
            new_setences.append(x)
        else:
            splitted = x.split()
            text = []

            for y in splitted:
                te = " ".join(text)

                if len(te) + len(y) < max_text_length - tolerance:
                    text.append(y)
                else:
                    new_setences.append(te)
                    text = [y]

            te = " ".join(text)

            if len(te) >= min_text_length:
                new_setences.append(te)

    return new_setences


def shuffle(array):
    """Modify a sequence by shuffling its contents"""

    arange = np.arange(0, len(array))
    np.random.shuffle(arange)
    new_array = []

    for i in range(len(arange)):
        new_array.append(array[i])
        array[i] = None

    return new_array


"""
Method to apply text random noise error (adapted):
    Author: Tal Weiss
    Title: Deep Spelling, 2016
    Article: https://machinelearnings.co/deep-spelling-9ffef96a24f6
    Repository URL: https://github.com/MajorTal/DeepSpell
"""


def add_noise(batch, max_text_length=128, level=1):
    """Add some artificial spelling mistakes to the string"""

    charset = list(string.ascii_letters + string.digits)

    for i in range(len(batch)):
        if len(batch[i]) > 4:
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

        batch[i] = padding_punctuation([batch[i]])[0]
        batch[i] = batch[i][:max_text_length - 1]

    return batch
