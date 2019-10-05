"""
SymSpell tool to text correction through Symmetric Delete spelling correction algorithm.

Original implementation:
    Version: 6.4
    Author: Wolf Garbe <wolf.garbe@faroo.com>
    URL: https://github.com/wolfgarbe/symspell
    Description: https://medium.com/@wolfgarbe/1000x-faster-spelling-correction-algorithm-2012-8701fcd87a5f

Python port:
    Version: 6.3
    Author: mammothb
    URL: https://github.com/mammothb/symspellpy
"""

import os
import re
import time
import string
from data import preproc as pp
from symspellpy.symspellpy import SymSpell


class Symspell():
    """Symspell class presents creation of a dictionary and autorrect texts through DataGenerator support"""

    def __init__(self, output_path, max_edit_distance=2, prefix_length=5):
        self.corpus_path = os.path.join(output_path, "corpus.data")
        self.dictionary_path = os.path.join(output_path, "dictionary.data")
        self.max_edit_distance = max_edit_distance
        self.symspell = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=prefix_length)

    def load(self, corpus):
        """Create and load corpus/dictionary"""

        start_time = time.time()

        with open(self.corpus_path, "w") as f:
            matches = re.findall(r"(([^\W_]|['’])+)", " ".join(corpus))
            matches = [match[0] for match in matches]
            f.write(" ".join(matches))

        self.symspell.create_dictionary(self.corpus_path)

        with open(self.dictionary_path, "w") as f:
            for key, count in self.symspell.words.items():
                f.write(f"{key} {count}\n")

        total_time = time.time() - start_time

        train_corpus = "\n".join([
            f"Total train sentences: {len(corpus)}",
            f"Total tokens:          {len(self.symspell.words.items())}\n",
            f"Total time:            {total_time:.8f} sec",
            f"Time per sentence:     {(total_time / len(corpus)):.8f} sec\n",
        ])

        return train_corpus

    def autocorrect(self, batch):
        """Text correction through word level"""

        self.symspell.load_dictionary(self.dictionary_path, term_index=0, count_index=1)
        corrected = []

        if not isinstance(batch, list):
            batch = [batch]

        for i in range(len(batch)):
            splitted = []

            for x in batch[i].split():
                sugg = self.symspell.lookup(x, verbosity=0, max_edit_distance=self.max_edit_distance,
                                            transfer_casing=True) if x not in string.punctuation else None
                splitted.append(pp.padding_punctuation(sugg[0].term if sugg else x))

            corrected.append(" ".join(splitted))
        return corrected
