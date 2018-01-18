from data import data_loader
import numpy as np


class Dataset:

    def __init__(self, data_fn: str, num_tests: int):
        self.num_tests = num_tests
        self.sen1, self.sen2, self.labels, self.max_doc_len, self.vocabulary_size = data_loader.load_snli(data_fn)
        self.__shuffle_idxs = range(len(self.labels) - num_tests)

    def train_instances(self, shuffle=False):
        train_sen1 = self.sen1[:-self.num_tests]
        train_sen2 = self.sen2[:-self.num_tests]
        if shuffle:
            self.__shuffle_idxs = np.random.permutation(range(len(train_sen1)))
            train_sen1 = train_sen1[self.__shuffle_idxs]
            train_sen2 = train_sen2[self.__shuffle_idxs]
        return train_sen1, train_sen2

    def train_labels(self):
        train_labels = self.labels[:-self.num_tests]
        train_labels = train_labels[self.__shuffle_idxs]
        return train_labels

    def test_instances(self):
        test_sen1 = self.sen1[-self.num_tests:]
        test_sen2 = self.sen2[-self.num_tests:]
        return test_sen1, test_sen2

    def test_labels(self):
        return self.labels[-self.num_tests:]
