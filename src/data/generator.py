"""Generator function to supply train/test with text data"""

import numpy as np
from data import preproc as pp, m2


class DataGenerator():
    """Generator class with data streaming"""

    def __init__(self, m2_src, batch_size, charset, max_text_length):
        self.dataset = m2.read_dataset(m2_src)
        self.batch_size = batch_size

        self.SOS, self.EOS = "«", "»"
        self.charset = self.SOS + charset + self.EOS
        self.padding_length = len(self.SOS) + len(self.EOS)
        self.max_text_length = max_text_length + self.padding_length

        self.full_fill_partition("train")
        self.full_fill_partition("valid")
        self.full_fill_partition("test")

        self.total_train = len(self.dataset["train"]["gt"])
        self.total_valid = len(self.dataset["valid"]["gt"])
        self.total_test = len(self.dataset["test"]["gt"])

        self.train_steps = np.maximum(self.total_train // self.batch_size, 1)
        self.valid_steps = np.maximum(self.total_valid // self.batch_size, 1)
        self.test_steps = np.maximum(self.total_test // self.batch_size, 1)

        self.train_index, self.valid_index, self.test_index = 0, 0, 0

    def full_fill_partition(self, pt):
        """Make full fill partition up to batch size and steps"""

        while len(self.dataset[pt]["gt"]) % self.batch_size:
            i = np.random.choice(np.arange(0, len(self.dataset[pt]["gt"])), 1)[0]

            self.dataset[pt]["dt"].append(self.dataset[pt]["dt"][i])
            self.dataset[pt]["gt"].append(self.dataset[pt]["gt"][i])

    def padding(self, batch, max_text_length, pre=None, post=None):
        """Add SOS and EOS chars to the sentences (padding)"""

        batch = batch.copy()

        for i in range(len(batch)):
            batch[i] = self.SOS + batch[i] if pre else batch[i]
            batch[i] += self.EOS * (max_text_length - len(batch[i]))

        return batch

    def encode_onehot(self, batch, charset, max_text_length, reverse=False):
        """Encode to one-hot"""

        encoded = np.zeros((len(batch), max_text_length, len(charset)))

        for y in range(len(batch)):
            for i, char in enumerate(batch[y]):
                try:
                    encoded[y, i, charset.find(char)] = 1
                except KeyError:
                    pass

            if reverse:
                encoded[y] = encoded[y][::-1]

        return encoded

    def next_train_batch(self):
        """Get the next batch from train partition (yield)"""

        while True:
            if self.train_index >= self.total_train:
                self.train_index = 0

            index = self.train_index
            until = self.train_index + self.batch_size
            self.train_index += self.batch_size

            y_train = self.dataset["train"]["gt"][index:until]
            x_train = pp.add_noise(y_train, max_text_length=(self.max_text_length - self.padding_length))

            x_train = self.padding(x_train, self.max_text_length, post=self.EOS)
            x_train_decoder = self.padding(y_train, self.max_text_length, pre=self.SOS, post=self.EOS)
            y_train = self.padding(y_train, self.max_text_length, post=self.EOS)

            x_train = self.encode_onehot(x_train, self.charset, self.max_text_length, reverse=True)
            x_train_decoder = self.encode_onehot(x_train_decoder, self.charset, self.max_text_length)
            y_train = self.encode_onehot(y_train, self.charset, self.max_text_length)

            yield [x_train, x_train_decoder], [y_train]

    def next_valid_batch(self):
        """Get the next batch from validation partition (yield)"""

        while True:
            if self.valid_index >= self.total_valid:
                self.valid_index = 0

            index = self.valid_index
            until = self.valid_index + self.batch_size
            self.valid_index += self.batch_size

            x_valid = self.dataset["valid"]["dt"][index:until]
            y_valid = self.dataset["valid"]["gt"][index:until]

            x_valid = self.padding(x_valid, self.max_text_length, post=self.EOS)
            x_valid_decoder = self.padding(y_valid, self.max_text_length, pre=self.SOS, post=self.EOS)
            y_valid = self.padding(y_valid, self.max_text_length, post=self.EOS)

            x_valid = self.encode_onehot(x_valid, self.charset, self.max_text_length, reverse=True)
            x_valid_decoder = self.encode_onehot(x_valid_decoder, self.charset, self.max_text_length)
            y_valid = self.encode_onehot(y_valid, self.charset, self.max_text_length)

            yield [x_valid, x_valid_decoder], [y_valid]

    def next_test_batch(self):
        """Return model predict parameters"""

        while True:
            if self.test_index >= self.total_test:
                self.test_index = 0

            index = self.test_index
            until = self.test_index + self.batch_size
            self.test_index += self.batch_size

            x_test = self.dataset["test"]["dt"][index:until]
            x_test = self.padding(x_test, self.max_text_length, post=self.EOS)
            x_test = self.encode_onehot(x_test, self.charset, self.max_text_length, reverse=True)

            yield [x_test]
