from enum import Enum

import numpy as np
import pandas as pd


class DatasetType(Enum):
    SNLI = 0,
    QQP = 1


class DatasetExperiment:

    def __init__(self, data_dir, num_tests, batch_size):
        self.data_dir = data_dir
        self.num_tests = num_tests
        self.batch_size = batch_size

    def train(self):
        raise NotImplementedError

    def dev(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError


class QQPDataset(DatasetExperiment):

    def __init__(self, *args):
        super().__init__(args)
        self.train = pd.read_csv('{}{}'.format(self.data_dir, 'train.csv'),
                                 sep=',',
                                 usecols=['question1', 'question2', 'is_duplicate'])
        self.test = pd.read_csv('{}{}'.format(self.data_dir, 'test.csv'), sep=',')

    def train(self):
        pass

    def dev(self):
        pass

    def test(self):
        pass


class SNLIDataset(DatasetExperiment):

    def __init__(self, *args):
        super().__init__(args)
        self.train = pd.read_csv('{}{}'.format(self.data_dir, 'train_snli.csv'), sep=',')

    def train(self):
        pass

    def dev(self):
        pass

    def test(self):
        pass


DATASETS = {
    DatasetType.QQP: QQPDataset,
    DatasetType.SNLI: SNLIDataset
}


class Dataset:

    def __init__(self, paraphrase_data, num_tests, batch_size):
        self.sen1 = paraphrase_data.sen1
        self.sen2 = paraphrase_data.sen2
        self.labels = paraphrase_data.labels

        self.num_tests = num_tests
        self.__shuffle_train_idxs = range(len(self.labels) - num_tests)
        self.train_sen1 = self.sen1[:-self.num_tests]
        self.train_sen2 = self.sen2[:-self.num_tests]
        self._train_labels = self.labels[:-self.num_tests]
        self.num_batches = (len(self.labels) - self.num_tests) // batch_size

    def train_instances(self, shuffle=False):
        if shuffle:
            self.__shuffle_train_idxs = np.random.permutation(range(len(self.__shuffle_train_idxs)))
            self.train_sen1 = self.train_sen1[self.__shuffle_train_idxs]
            self.train_sen2 = self.train_sen2[self.__shuffle_train_idxs]
            self._train_labels = self._train_labels[self.__shuffle_train_idxs]
        return self.train_sen1, self.train_sen2

    def train_labels(self):
        return self._train_labels

    def test_instances(self):
        test_sen1 = self.sen1[-self.num_tests:]
        test_sen2 = self.sen2[-self.num_tests:]
        return test_sen1, test_sen2

    def test_labels(self):
        return self.labels[-self.num_tests:]

    def validation_instances(self, num_instances=None):
        if num_instances is None:
            num_instances = len(self.__shuffle_train_idxs)  # get all training instances for validation

        val_idxs = np.random.permutation(range(len(self.__shuffle_train_idxs)))
        train_sen1, train_sen2 = self.train_instances(shuffle=False)
        train_labels = self.train_labels()

        val_sen1, val_sen2 = train_sen1[val_idxs][:num_instances], train_sen2[val_idxs][:num_instances]
        val_labels = train_labels[val_idxs][:num_instances]
        return val_sen1, val_sen2, val_labels

    def __str__(self):
        test_train_ratio = self.num_tests / len(self.labels)
        return 'Dataset properties:\n ' \
               'Number of training instances: {}\n ' \
               'Number of test instances: {}\n' \
               'Test/Train ratio: {}' \
            .format(len(self.labels), self.num_tests, test_train_ratio)
