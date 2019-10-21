"""Generator function to supply train/test with text data"""

import numpy as np
from data import preproc as pp, m2
from tensorflow.keras.preprocessing.sequence import pad_sequences


class DataGenerator():
    """Generator class with data streaming"""

    def __init__(self, m2_src, batch_size, charset, max_text_length=128):
        self.tokenizer = Tokenizer(charset, max_text_length)
        self.batch_size = batch_size

        self.dataset = m2.read_dataset(m2_src)
        self._prepare_dataset()

        self.total_train = len(self.dataset["train"]["gt"])
        self.total_valid = len(self.dataset["valid"]["gt"])
        self.total_test = len(self.dataset["test"]["gt"])

        self.train_steps = np.maximum(self.total_train // batch_size, 1)
        self.valid_steps = np.maximum(self.total_valid // batch_size, 1)
        self.test_steps = np.maximum(self.total_test // batch_size, 1)

        self.train_index, self.valid_index, self.test_index = 0, 0, 0

        self.noise_process = (len(max(self.dataset["train"]["dt"])) == 0)
        self.one_hot_process = True

    def _prepare_dataset(self):
        """Prepare (text standardize and full fill) dataset up"""

        for pt in ["train", "valid", "test"]:
            # text standardize to avoid erros
            self.dataset[pt]["dt"] = [pp.text_standardize(x) for x in self.dataset[pt]["dt"]]
            self.dataset[pt]["gt"] = [pp.text_standardize(x) for x in self.dataset[pt]["gt"]]

            # full fill process to make up batch_size and steps
            while len(self.dataset[pt]["gt"]) % self.batch_size:
                i = np.random.choice(np.arange(0, len(self.dataset[pt]["gt"])), 1)[0]

                self.dataset[pt]["dt"].append(self.dataset[pt]["dt"][i])
                self.dataset[pt]["gt"].append(self.dataset[pt]["gt"][i])

    def prepare_sequence(self, sentences, sos=False, eos=False, add_noise=False, reverse=False):
        """Prepare inputs to feed the model"""

        n_sen = sentences.copy()
        sos = self.tokenizer.SOS_TK if sos else ""
        eos = self.tokenizer.EOS_TK if eos else ""

        for i in range(len(n_sen)):
            if add_noise:
                n_sen[i] = pp.add_noise([n_sen[i]], self.tokenizer.maxlen)[0]

            n_sen[i] = self.tokenizer.encode(sos + n_sen[i] + eos)
            n_sen[i] = pad_sequences([n_sen[i]], maxlen=self.tokenizer.maxlen, padding="post")[0]

            if self.one_hot_process:
                n_sen[i] = self.tokenizer.encode_one_hot(n_sen[i])

            if reverse:
                n_sen[i] = n_sen[i][::-1]

        return np.array(n_sen)

    def next_train_batch(self):
        """Get the next batch from train partition (yield)"""

        while True:
            if self.train_index >= self.total_train:
                self.train_index = 0

            index = self.train_index
            until = self.train_index + self.batch_size
            self.train_index += self.batch_size

            targets = self.dataset["train"]["gt"][index:until]

            if self.noise_process:
                inputs = self.prepare_sequence(targets, eos=True, add_noise=True)
            else:
                inputs = self.dataset["train"]["dt"][index:until]
                inputs = self.prepare_sequence(inputs, eos=True, add_noise=False)

            decoder_inputs = self.prepare_sequence(targets, sos=True)
            targets = self.prepare_sequence(targets, eos=True)

            yield [inputs, decoder_inputs], targets

    def next_valid_batch(self):
        """Get the next batch from valid partition (yield)"""

        while True:
            if self.valid_index >= self.total_valid:
                self.valid_index = 0

            index = self.valid_index
            until = self.valid_index + self.batch_size
            self.valid_index += self.batch_size

            inputs = self.dataset["valid"]["dt"][index:until]
            targets = self.dataset["valid"]["gt"][index:until]

            inputs = self.prepare_sequence(inputs, eos=True)
            decoder_inputs = self.prepare_sequence(targets, sos=True)
            targets = self.prepare_sequence(targets, eos=True)

            yield [inputs, decoder_inputs], targets

    def next_test_batch(self):
        """Get the next batch from test partition (yield)"""

        while True:
            if self.test_index >= self.total_test:
                self.test_index = 0

            index = self.test_index
            until = self.test_index + self.batch_size
            self.test_index += self.batch_size

            inputs = self.dataset["test"]["dt"][index:until]

            inputs = self.prepare_sequence(inputs, eos=True)

            yield inputs


class Tokenizer():
    """Manager tokens functions and charset/dictionary properties"""

    def __init__(self, chars, max_text_length=128):
        self.PAD_TK, self.UNK_TK, self.SOS_TK, self.EOS_TK = "¶", "¥", "«", "»"
        self.chars = (self.PAD_TK + self.UNK_TK + chars + self.SOS_TK + self.EOS_TK)

        self.PAD = self.chars.find(self.PAD_TK)
        self.UNK = self.chars.find(self.UNK_TK)
        self.SOS = self.chars.find(self.SOS_TK)
        self.EOS = self.chars.find(self.EOS_TK)

        self.vocab_size = len(self.chars)
        self.maxlen = max_text_length

    def encode(self, text):
        """Encode text to vector"""

        encoded = []

        for item in text:
            index = self.chars.find(item)
            index = self.UNK if index == -1 else index
            encoded.append(index)

        return np.array(encoded)

    def decode(self, text):
        """Decode vector to text"""

        return "".join([self.chars[int(x)] for x in text])

    def encode_one_hot(self, vector):
        """Encode vector to one-hot"""

        encoded = np.zeros((len(vector), self.vocab_size))

        for i in range(len(vector)):
            try:
                encoded[i][int(vector[i])] = 1.0
            except KeyError:
                encoded[i][int(self.UNK)] = 1.0

        return np.array(encoded)

    def decode_one_hot(self, one_hot):
        """Decode one-hot to vector"""

        return np.argmax(one_hot, axis=1)

    def remove_tokens(self, text):
        """Remove tokens (PAD, SOS, EOS) from text"""

        return text.replace(self.PAD_TK, "").replace(self.SOS_TK, "").replace(self.EOS_TK, "")
