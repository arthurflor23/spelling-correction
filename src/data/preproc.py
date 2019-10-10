"""Methods to help manager data development"""

import re
import string
import numpy as np


def padding_punctuation(sentence):
    """Organize/add spaces around punctuation marks"""

    sentence = sentence.replace(" '", "'").replace("' ", "'")
    sentence = sentence.replace("«", "").replace("»", "")

    for y in sentence:
        if y in string.punctuation.replace("'", ""):
            sentence = sentence.replace(y, f" {y} ")

    sentence = " ".join(sentence.split())

    return sentence


def split_by_max_length(sentence, charset=None, max_text_length=128):
    """Standardize n_sentences: split long n_sentences into max_text_length"""

    tolerance = 5
    max_text_length -= tolerance
    new_n_sentences = []

    if charset is not None:
        sentence = "".join([c for c in sentence if c in charset])

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


def add_noise(sentences, max_text_length=128, amount_noise=0.5, level=2):
    """Add some artificial spelling mistakes to the string"""

    charset = list(set(" ()[].,\"'" + string.ascii_letters + string.digits))
    n_sentences = sentences.copy()

    for i in range(len(n_sentences)):
        for _ in range(level):

            if len(n_sentences[i]) > 4:
                # Replace a character with a random character
                if np.random.rand() < amount_noise:
                    position = np.random.randint(len(n_sentences[i]))
                    n_sentences[i] = (n_sentences[i][:position] + np.random.choice(charset[:-1]) +
                                      n_sentences[i][position + 1:])

                # Transpose 2 characters
                if np.random.rand() < amount_noise:
                    position = np.random.randint(len(n_sentences[i]) - 1)
                    n_sentences[i] = (n_sentences[i][:position] + n_sentences[i][position + 1] +
                                      n_sentences[i][position] + n_sentences[i][position + 2:])

                # Delete a character
                if np.random.rand() < amount_noise:
                    position = np.random.randint(len(n_sentences[i]))
                    n_sentences[i] = n_sentences[i][:position] + n_sentences[i][position + 1:]

                # Delete repeated characters
                if np.random.rand() < amount_noise:
                    n_sentences[i] = re.compile(r'(.)\1{1,}', re.IGNORECASE).sub(r'\1', n_sentences[i])

                # Add a random character
                if np.random.rand() < amount_noise and len(n_sentences[i]) < max_text_length:
                    position = np.random.randint(len(n_sentences[i]))
                    n_sentences[i] = (n_sentences[i][:position] + np.random.choice(charset[:-1]) +
                                      n_sentences[i][position:])

        n_sentences[i] = padding_punctuation(n_sentences[i])
        n_sentences[i] = n_sentences[i][:max_text_length - 1]

    return n_sentences
