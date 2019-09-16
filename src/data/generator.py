"""Generator function to supply train/test with text data."""

import numpy as np
from data import preproc as pp, m2


class DataGenerator():
    """Generator class with data streaming"""

    def __init__(self, m2_src, batch_size, charset, max_text_length, ctc=False):
        self.dataset = m2.read_dataset(m2_src)
        self.batch_size = batch_size
        self.charset = charset
        self.max_text_length = max_text_length
        self.ctc = ctc

        # Normalize sentences of the test partition (only to custom predicts)
        # self.dataset["test"]["dt"] = pp.normalize_text(self.dataset["test"]["dt"], charset, max_text_length)
        # self.dataset["test"]["gt"] = pp.normalize_text(self.dataset["test"]["gt"], charset, max_text_length)

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

    def next_train_batch(self):
        """Get the next batch from train partition (yield)"""

        while True:
            if self.train_index >= self.total_train:
                self.train_index = 0

            index = self.train_index
            until = self.train_index + self.batch_size
            self.train_index += self.batch_size

            y_train = self.dataset["train"]["gt"][index:until]
            x_train = pp.add_noise(y_train, self.max_text_length)

            x_train = [pp.encode_onehot(x, self.charset, self.max_text_length, reverse=True) for x in x_train]

            if self.ctc:
                y_train = [pp.encode_onehot(x, self.charset, self.max_text_length) for x in y_train]

                x_train_len = np.asarray([self.max_text_length for _ in range(self.batch_size)])
                y_train_len = np.asarray([len(np.trim_zeros(y_train[i])) for i in range(self.batch_size)])

                inputs = {
                    "input": x_train,
                    "labels": y_train,
                    "input_length": x_train_len,
                    "label_length": y_train_len
                }
                output = {"CTCloss": np.zeros(self.batch_size)}
                yield (inputs, output)

            else:
                y_train = [pp.encode_onehot(x, self.charset, self.max_text_length) for x in y_train]
                yield [x_train], [y_train]

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

            x_valid = [pp.encode_onehot(x, self.charset, self.max_text_length, reverse=True) for x in x_valid]

            if self.ctc:
                y_valid = [pp.encode_onehot(x, self.charset, self.max_text_length) for x in y_valid]

                x_valid_len = np.asarray([self.max_text_length for _ in range(self.batch_size)])
                y_valid_len = np.asarray([len(np.trim_zeros(y_valid[i])) for i in range(self.batch_size)])

                inputs = {
                    "input": x_valid,
                    "labels": y_valid,
                    "input_length": x_valid_len,
                    "label_length": y_valid_len
                }
                output = {"CTCloss": np.zeros(self.batch_size)}
                yield (inputs, output)

            else:
                y_valid = [pp.encode_onehot(x, self.charset, self.max_text_length) for x in y_valid]
                yield [x_valid], [y_valid]

    def next_test_batch(self):
        """Return model predict parameters"""

        while True:
            if self.test_index >= self.total_test:
                self.test_index = 0

            index = self.test_index
            until = self.test_index + self.batch_size
            self.test_index += self.batch_size

            x_test = self.dataset["test"]["dt"][index:until]
            y_test = self.dataset["test"]["gt"][index:until]

            x_test = [pp.encode_onehot(x, self.charset, self.max_text_length, reverse=True) for x in x_test]

            if self.ctc:
                w_test = [x.encode_onehot() for x in y_test]
                y_test = [pp.encode_onehot(x, self.charset, self.max_text_length) for x in y_test]

                x_test_len = np.asarray([self.max_text_length for _ in range(self.batch_size)])
                y_test_len = np.asarray([len(np.trim_zeros(y_test[i])) for i in range(self.batch_size)])

                yield [x_test, y_test, x_test_len, y_test_len, w_test]
            else:
                yield [x_test]
