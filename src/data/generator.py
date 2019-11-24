"""Generator function to supply train/test with text data"""

import numpy as np
from data import preproc as pp, reader
from tensorflow.keras.preprocessing.sequence import pad_sequences


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
        self._prepare_dataset()

        self.one_hot_process = True
        self.noise_process = len(max(self.dataset['train']['dt'], default=[''])) == 0

        # increase `iterations` parameter by 2 if there is noise process in the train data
        if self.noise_process:
            max_prob, iterations = pp.add_noise.__defaults__
            pp.add_noise.__defaults__ = (max_prob, iterations + 2)

    def _prepare_dataset(self):
        """Prepare (text standardize and full fill) dataset up"""

        for pt in self.partitions:
            # text standardize to avoid erros
            self.dataset[pt]['dt'] = [pp.text_standardize(x) for x in self.dataset[pt]['dt']]
            self.dataset[pt]['gt'] = [pp.text_standardize(x) for x in self.dataset[pt]['gt']]

            # full fill process to make up batch_size and steps
            while len(self.dataset[pt]['gt']) % self.batch_size:
                i = np.random.choice(np.arange(0, len(self.dataset[pt]['gt'])), 1)[0]

                self.dataset[pt]['dt'].append(self.dataset[pt]['dt'][i])
                self.dataset[pt]['gt'].append(self.dataset[pt]['gt'][i])

            self.size[pt] = len(self.dataset[pt]['gt'])
            self.steps[pt] = max(1, self.size[pt] // self.batch_size)
            self.index[pt] = 0

    def prepare_sequence(self, sentences, sos=False, eos=False, add_noise=False):
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

        return np.array(n_sen)

    def next_train_batch(self):
        """Get the next batch from train partition (yield)"""

        while True:
            if self.index['train'] >= self.size['train']:
                self.index['train'] = 0

            index = self.index['train']
            until = self.index['train'] + self.batch_size
            self.index['train'] += self.batch_size

            targets = self.dataset['train']['gt'][index:until]
            inputs = targets if self.noise_process else self.dataset['train']['dt'][index:until]

            inputs = self.prepare_sequence(inputs, sos=True, eos=True, add_noise=self.noise_process)
            decoder_inputs = self.prepare_sequence(targets, sos=True)
            targets = self.prepare_sequence(targets, eos=True)

            # x, y and sample_weight
            yield ([inputs, decoder_inputs], targets, [])

    def next_valid_batch(self):
        """Get the next batch from valid partition (yield)"""

        while True:
            if self.index['valid'] >= self.size['valid']:
                self.index['valid'] = 0

            index = self.index['valid']
            until = self.index['valid'] + self.batch_size
            self.index['valid'] += self.batch_size

            inputs = self.dataset['valid']['dt'][index:until]
            targets = self.dataset['valid']['gt'][index:until]

            inputs = self.prepare_sequence(inputs, sos=True, eos=True)
            decoder_inputs = self.prepare_sequence(targets, sos=True)
            targets = self.prepare_sequence(targets, eos=True)

            # x, y and sample_weight
            yield ([inputs, decoder_inputs], targets, [])

    def next_test_batch(self):
        """Get the next batch from test partition (yield)"""

        while True:
            if self.index['test'] >= self.size['test']:
                break

            index = self.index['test']
            until = self.index['test'] + self.batch_size
            self.index['test'] += self.batch_size

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

        return text.split(self.EOS_TK)[0].replace(self.SOS_TK, "").replace(self.PAD_TK, "")
