"""Generator function to supply train/test with text data."""

import numpy as np
from data import preproc, m2


class DataGenerator():
    """Generator class with data streaming"""

    def __init__(self, m2_src, batch_size, max_text_length, charset):
        self.batch_size = batch_size
        self.max_text_length = max_text_length
        self.dataset = m2.read_dataset(m2_src)
        self.charset = charset

        self.train_index, self.valid_index, self.test_index = 0, 0, 0

        self.total_train = len(self.dataset["train"]["gt"])
        self.total_valid = len(self.dataset["valid"]["gt"])
        self.total_test = len(self.dataset["test"]["gt"])

        self.train_steps = self.total_train // self.batch_size
        self.valid_steps = self.total_valid // self.batch_size
        self.test_steps = self.total_test // self.batch_size

    def fill_batch(self, partition, maximum, x, y):
        """Fill batch array (x, y) if required (batch_size)"""

        if len(x) < self.batch_size:
            fill = self.batch_size - len(x)
            i = np.random.choice(np.arange(0, maximum - fill), 1)[0]

            if x is not None:
                x = np.append(x, self.dataset[partition]["dt"][i:i + fill], axis=0)
            if y is not None:
                y = np.append(y, self.dataset[partition]["gt"][i:i + fill], axis=0)

        return (x, y)

    def next_train_batch(self):
        """Get the next batch from train partition (yield)"""

        while True:
            if self.train_index >= self.total_train:
                self.train_index = 0

            index = self.train_index
            until = self.train_index + self.batch_size
            self.train_index += self.batch_size

            y_train = self.dataset["train"]["gt"][index:until]
            _, y_train = self.fill_batch("train", self.total_train, None, y_train)

            x_train = preproc.add_noise(y_train, self.max_text_length)

            x_train = [preproc.encode(x, self.charset, self.max_text_length) for x in x_train]
            y_train = [preproc.encode(x, self.charset, self.max_text_length) for x in y_train]

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

            x_valid, y_valid = self.fill_batch("valid", self.total_valid, x_valid, y_valid)

            x_valid = [preproc.encode(x, self.charset, self.max_text_length) for x in x_valid]
            y_valid = [preproc.encode(x, self.charset, self.max_text_length) for x in y_valid]

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

            x_test, y_test = self.fill_batch("test", self.total_test, x_test, y_test)
            w_test = [x.encode() for x in y_test]

            x_test = [preproc.encode(x, self.charset, self.max_text_length) for x in x_test]
            y_test = [preproc.encode(x, self.charset, self.max_text_length) for x in y_test]

            x_test_len = np.asarray([self.max_text_length for _ in range(self.batch_size)])
            y_test_len = np.asarray([len(np.trim_zeros(y_test[i])) for i in range(self.batch_size)])

            yield [x_test, y_test, x_test_len, y_test_len, w_test]
