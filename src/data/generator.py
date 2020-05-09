"""Generator function to supply train/test with text data"""

import numpy as np
from data import preproc as pp, reader


class DataGenerator():
    """Generator class with data streaming"""

    def __init__(self, source, batch_size, charset, max_text_length=128, predict=False):
        self.tokenizer = Tokenizer(charset, max_text_length)
        self.batch_size = batch_size
        self.partitions = ['test'] if predict else ['train', 'valid', 'test']

        self.size = dict()
        self.steps = dict()
        self.index = dict()
        self.dataset = reader.read_from_txt(source)

        for pt in self.partitions:
            randomize = np.arange(len(self.dataset[pt]['gt']))
            np.random.seed(42)
            np.random.shuffle(randomize)

            self.dataset[pt]['dt'] = np.asarray(self.dataset[pt]['dt'])[randomize]
            self.dataset[pt]['gt'] = np.asarray(self.dataset[pt]['gt'])[randomize]

            # text standardize to avoid erros
            self.dataset[pt]['dt'] = [pp.text_standardize(x) for x in self.dataset[pt]['dt']]
            self.dataset[pt]['gt'] = [pp.text_standardize(x) for x in self.dataset[pt]['gt']]

            # set size and setps
            self.size[pt] = len(self.dataset[pt]['gt'])
            self.steps[pt] = int(np.ceil(self.size[pt] / self.batch_size))
            self.index[pt] = 0

        self.one_hot_process = True
        self.noise_process = not bool(max(self.dataset['train']['dt'], default=['']))

        # increase `iterations` parameter if there is noise process in the train data
        if self.noise_process:
            ratio, iterations = pp.add_noise.__defaults__
            pp.add_noise.__defaults__ = (ratio, iterations + 2)

    def prepare_sequence(self, sentences, sos=False, eos=False, add_noise=False):
        """Prepare inputs to feed the model"""

        n_sen = list(sentences).copy()

        sos = self.tokenizer.SOS_TK if sos else ""
        eos = self.tokenizer.EOS_TK if eos else ""

        for i in range(len(n_sen)):
            if add_noise:
                n_sen[i] = pp.add_noise([n_sen[i]], self.tokenizer.maxlen)[0]

            n_sen[i] = self.tokenizer.encode(sos + n_sen[i] + eos)
            n_sen[i] = np.pad(n_sen[i], (0, self.tokenizer.maxlen - len(n_sen[i])))

            if self.one_hot_process:
                n_sen[i] = self.tokenizer.encode_one_hot(n_sen[i])

        return np.asarray(n_sen, dtype=np.int16)

    def next_train_batch(self):
        """Get the next batch from train partition (yield)"""

        while True:
            if self.index['train'] >= self.size['train']:
                self.index['train'] = 0

            index = self.index['train']
            until = self.index['train'] + self.batch_size
            self.index['train'] = until

            targets = self.dataset['train']['gt'][index:until]
            inputs = targets if self.noise_process else self.dataset['train']['dt'][index:until]

            inputs = self.prepare_sequence(inputs, sos=True, eos=True, add_noise=self.noise_process)
            decoder_inputs = self.prepare_sequence(targets, sos=True)
            targets = self.prepare_sequence(targets, eos=True)

            yield ([inputs, decoder_inputs], targets)

    def next_valid_batch(self):
        """Get the next batch from valid partition (yield)"""

        while True:
            if self.index['valid'] >= self.size['valid']:
                self.index['valid'] = 0

            index = self.index['valid']
            until = self.index['valid'] + self.batch_size
            self.index['valid'] = until

            inputs = self.dataset['valid']['dt'][index:until]
            targets = self.dataset['valid']['gt'][index:until]

            inputs = self.prepare_sequence(inputs, sos=True, eos=True)
            decoder_inputs = self.prepare_sequence(targets, sos=True)
            targets = self.prepare_sequence(targets, eos=True)

            yield ([inputs, decoder_inputs], targets)

    def next_test_batch(self):
        """Get the next batch from test partition (yield)"""

        while True:
            if self.index['test'] >= self.size['test']:
                self.index['test'] = 0
                break

            index = self.index['test']
            until = self.index['test'] + self.batch_size
            self.index['test'] = until

            inputs = self.dataset['test']['dt'][index:until]

            inputs = self.prepare_sequence(inputs, sos=True, eos=True)

            yield [inputs]


class Tokenizer():
    """Manager tokens functions and charset/dictionary properties"""

    def __init__(self, chars, max_text_length=128):
        self.PAD_TK, self.UNK_TK, self.SOS_TK, self.EOS_TK = "¶", "¤", "«", "»"
        self.chars = (self.PAD_TK + self.UNK_TK + self.SOS_TK + self.EOS_TK + chars)

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

        return np.asarray(encoded)

    def decode(self, text):
        """Decode vector to text"""

        return "".join([self.chars[int(x)] for x in text])

    def encode_one_hot(self, vector):
        """Encode vector to one-hot"""

        encoded = np.zeros((len(vector), self.vocab_size), dtype=bool)

        for i in range(len(vector)):
            try:
                encoded[i][int(vector[i])] = 1
            except KeyError:
                encoded[i][int(self.UNK)] = 1

        return encoded

    def decode_one_hot(self, one_hot):
        """Decode one-hot to vector"""

        return np.argmax(one_hot, axis=1, dtype=np.int16)

    def remove_tokens(self, text):
        """Remove tokens (PAD, SOS, EOS) from text"""

        return text.split(self.EOS_TK)[0].replace(self.SOS_TK, "").replace(self.PAD_TK, "")
