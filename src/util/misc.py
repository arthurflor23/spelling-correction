"""
Methods to help manage data development.
"""

from numpy.random import choice as random_choice, randint as random_randint, rand
import string


def text_normalization(strings, charset, limit):
    """
    Normalize a batch of strings: replace some stuffs, add spaces around punctuation marks.
    """

    limit -= 1
    min_text_length = 3
    text_list = []

    if not isinstance(strings, list):
        strings = [strings]

    for i in range(len(strings)):
        strings[i] = strings[i].replace("«", "").replace("»", "").replace("“", "\"")

        for y in strings[i]:
            if y not in charset:
                strings[i] = strings[i].replace(y, "")

            if y in string.punctuation.replace("'", ""):
                strings[i] = strings[i].replace(y, f" {y} ")

        strings[i] = " ".join(strings[i].split())

        if len(strings[i]) < min_text_length:
            continue

        if len(strings[i]) < limit:
            text_list.append(strings[i])
            strings[i] = None
        else:
            splitted = strings[i].split()
            strings[i] = None
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


"""
Tool to apply text random noise error, available here:
https://gist.github.com/MajorTal/67d54887a729b5e5aa85
"""


def add_noise(batch):
    """Add some artificial spelling mistakes to the string"""

    max_text_length = 128
    amount_of_noise = 0.1 * max_text_length
    charset = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    if not isinstance(charset, list):
        charset = list(charset)

    for i in range(len(batch)):
        for _ in range(2):
            if rand() < amount_of_noise * len(batch[i]):
                # Replace a character with a random character
                random_char_position = random_randint(len(batch[i]))
                batch[i] = batch[i][:random_char_position] + random_choice(charset[:-1]) + batch[i][random_char_position + 1:]

            if rand() < amount_of_noise * len(batch[i]):
                # Delete a character
                random_char_position = random_randint(len(batch[i]))
                batch[i] = batch[i][:random_char_position] + batch[i][random_char_position + 1:]

            if len(batch[i]) < max_text_length and rand() < amount_of_noise * len(batch[i]):
                # Add a random character
                random_char_position = random_randint(len(batch[i]))
                batch[i] = batch[i][:random_char_position] + random_choice(charset[:-1]) + batch[i][random_char_position:]

            if rand() < amount_of_noise * len(batch[i]):
                # Transpose 2 characters
                random_char_position = random_randint(len(batch[i]) - 1)
                batch[i] = (batch[i][:random_char_position] + batch[i][random_char_position + 1] +
                            batch[i][random_char_position] + batch[i][random_char_position + 2:])

    return batch
