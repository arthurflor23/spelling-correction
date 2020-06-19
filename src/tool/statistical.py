"""
Language Model class.
Create and read the corpus with the language model file.

Statistical techniques for spelling correction:
    * N-gram with similarity.
    * Peter Norvig's method.
    * Symspell method.
"""

import os
import re
import string

from ngram import NGram
from spellchecker import SpellChecker
from symspellpy.symspellpy import SymSpell


class LanguageModel():

    def __init__(self, mode, output, N=2):
        self.autocorrect = getattr(self, f"_{mode}")
        self.output_path = output
        self.N = N

    def create_corpus(self, sentences):
        """Create corpus file"""

        matches = " ¶ ".join(sentences).translate(str.maketrans("", "", string.punctuation))
        matches = re.compile(r'[^\S\n]+', re.UNICODE).sub(" ", matches.strip())
        matches = "\n".join(matches.strip().split(" ¶ ")).lower()

        return matches

    def read_corpus(self, corpus_path):
        """Read corpus file to the autocorrect tool"""

        self.corpus_path = corpus_path
        self.dictionary_path = os.path.join(os.path.dirname(corpus_path), "dictionary.txt")
        self.corpus = " ".join(open(corpus_path).read().splitlines()).lower()

    def _kaldi(self, sentences, predict=True):
        """
        Kaldi Speech Recognition Toolkit with SRI Language Modeling Toolkit.
        ** Important Note **
        You'll need to do all by yourself:
        1. Compile Kaldi with SRILM and OpenBLAS.
        2. Create and add kaldi folder in the project `lib` folder (``src/lib/kaldi/``)
        3. Generate files (search `--kaldi_assets` in https://github.com/arthurflor23/handwritten-text-recognition):
            a. `chars.lst`
            b. `conf_mats.ark`
            c. `ground_truth.lst`
            d. `ID_test.lst`
            e. `ID_train.lst`
        4. Add files (item 3) in the project `output` folder: ``output/<DATASET>/kaldi/``
        More information (maybe help) in ``src/lib/kaldi-decode-script.sh`` comments.
        References:
            D. Povey, A. Ghoshal, G. Boulianne, L. Burget, O. Glembek, N. Goel, M. Hannemann,
            P. Motlicek, Y. Qian, P. Schwarz, J. Silovsky, G. Stem- mer and K. Vesely.
            The Kaldi speech recognition toolkit, 2011.
            Workshop on Automatic Speech Recognition and Understanding.
            URL: http://github.com/kaldi-asr/kaldi
            Andreas Stolcke.
            SRILM - An Extensible Language Modeling Toolkit, 2002.
            Proceedings of the 7th International Conference on Spoken Language Processing (ICSLP).
            URL: http://www.speech.sri.com/projects/srilm/
        """

        option = "TEST" if predict else "TRAIN"

        if os.system(f"./lib/kaldi-decode-script.sh {self.output_path} {option} {self.N}") != 0:
            print("\n##########################################\n")
            print("You'll have to work hard for this option.\n")
            print("See some instructions in the ``src/tool/statistical.py`` file (kaldi function section)")
            print("and also in the ``src/lib/kaldi-decode-script.sh`` file. \n☘️ ☘️ ☘️")
            print("\n##########################################\n")

        if predict:
            predicts = open(os.path.join(self.output_path, "data", "predicts_t")).read().splitlines()

            for i, line in enumerate(predicts):
                tokens = line.split()
                predicts[i] = "".join(tokens[1:]).replace("<space>", " ").strip()

            return predicts

    def _similarity(self, sentences):
        """
        N-gram with similarity.

        The NGram class extends the Python ‘set’ class with efficient fuzzy search for members by
        means of an N-gram similarity measure.

        Reference:
            Vacláv Chvátal and David Sankoff.
            Longest common subsequences of two random sequences, 1975.
            Journal of Applied Probability,

            Python module: ngram (https://pypi.org/project/ngram/)
        """

        ngram = NGram(self.corpus.split(), key=lambda x: x.lower(), N=self.N)
        predicts = []

        if not isinstance(sentences, list):
            sentences = [sentences]

        for i in range(len(sentences)):
            split = []

            for x in sentences[i].split():
                sugg = ngram.find(x.lower()) if x not in string.punctuation else None
                split.append(sugg if sugg else x)

            predicts.append(" ".join(split))

        return predicts

    def _norvig(self, sentences):
        """
        It uses a Levenshtein Distance algorithm to find permutations within an edit distance of 2
        from the original word. It then compares all permutations (insertions, deletions, replacements,
        and transpositions) to known words in a word frequency list.
        Those words that are found more often in the frequency list are more likely the correct results.

        Reference:
            Stuart J. Russell and Peter Norvig.
            Artificial intelligence - a modern approach: the intelligent agent book, 1995.
            Prentice Hall series in artificial intelligence.
            URL: http://norvig.com/spell-correct.html

            Python module: pyspellchecker (https://pypi.org/project/pyspellchecker/)
        """

        norvig = SpellChecker(distance=self.N)
        norvig.word_frequency.load_words(self.corpus.split())
        predicts = []

        if not isinstance(sentences, list):
            sentences = [sentences]

        for i in range(len(sentences)):
            split = []

            for x in sentences[i].split():
                sugg = norvig.correction(x.lower()) if x not in string.punctuation else None
                split.append(sugg if sugg else x)

            predicts.append(" ".join(split))

        return predicts

    def _symspell(self, sentences):
        """
        SymSpell tool to spelling correction through Symmetric Delete spelling algorithm.

        Reference:
            Author: Wolf Garbe <wolf.garbe@faroo.com>
            Description: https://medium.com/@wolfgarbe/1000x-faster-spelling-correction-algorithm-2012-8701fcd87a5f
            URL: https://github.com/wolfgarbe/symspell

            Python module: symspellpy (https://github.com/mammothb/symspellpy)
        """

        symspell = SymSpell(max_dictionary_edit_distance=self.N)
        symspell.create_dictionary(self.corpus_path)

        with open(self.dictionary_path, "w") as f:
            for key, count in symspell.words.items():
                f.write(f"{key} {count}\n")

        symspell.load_dictionary(self.dictionary_path, term_index=0, count_index=1)
        predicts = []

        if not isinstance(sentences, list):
            sentences = [sentences]

        for i in range(len(sentences)):
            split = []

            for x in sentences[i].split():
                sugg = symspell.lookup(x.lower(), verbosity=0, max_edit_distance=self.N,
                                       transfer_casing=True) if x not in string.punctuation else None
                split.append(sugg[0].term if sugg else x)

            predicts.append(" ".join(split))

        return predicts
