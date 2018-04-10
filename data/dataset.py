from enum import Enum

import numpy as np
import pandas as pd


class DatasetType(Enum):
    SNLI = 0,
    QQP = 1


class ColumnType(Enum):
    sentence1 = 0,
    sentence2 = 1,
    labels = 2


columns = [ColumnType.sentence1.name,
           ColumnType.sentence2.name,
           ColumnType.labels.name]


class DatasetExperiment:

    def __init__(self, dev_ratio=0.01, test_ratio=0.1):
        self.data_dir = self._data_path()
        self.dev_ratio = dev_ratio
        self.test_ratio = test_ratio

    def train_set(self):
        raise NotImplementedError

    def train_set_pairs(self):
        raise NotImplementedError

    def train_labels(self):
        raise NotImplementedError

    def dev_set(self):
        raise NotImplementedError

    def dev_set_pairs(self):
        raise NotImplementedError

    def dev_labels(self):
        raise NotImplementedError

    def test_set(self):
        raise NotImplementedError

    def test_set_pairs(self):
        raise NotImplementedError

    def test_labels(self):
        raise NotImplementedError

    def _data_path(self):
        raise NotImplementedError


class QQPDataset(DatasetExperiment):

    def __init__(self, *args):
        super().__init__(*args)
        dataset = pd.read_csv('{}{}'.format(self.data_dir, 'train.csv'),
                              sep=',',
                              usecols=['question1', 'question2', 'is_duplicate'])
        dataset.dropna(inplace=True)
        dataset = dataset.sample(frac=1, random_state=1).reset_index(drop=True)
        num_instances = len(dataset)
        self.num_train = num_instances * (1 - self.dev_ratio - self.test_ratio)
        self.num_dev = num_instances * self.dev_ratio
        self.num_test = num_instances * self.test_ratio
        self.train = dataset.loc[:self.num_train]
        self.dev = dataset.loc[self.num_train:self.num_train + self.num_dev]
        self.test = dataset.loc[self.num_train + self.num_dev:self.num_train + self.num_dev + self.num_test]

    def train_set(self):
        return self.train

    def train_set_pairs(self):
        return self.train[['question1', 'question2']].as_matrix()

    def train_labels(self):
        return self.train['is_duplicate'].as_matrix()

    def dev_set(self):
        return self.dev

    def dev_set_pairs(self):
        return self.dev[['question1', 'question2']].as_matrix()

    def dev_labels(self):
        return self.dev['is_duplicate'].as_matrix()

    def test_set(self):
        return self.test

    def test_set_pairs(self):
        return self.test[['question1', 'question2']].as_matrix()

    def test_labels(self):
        return self.test['is_duplicate'].as_matrix()

    def _data_path(self):
        return 'corpora/QQP/'


class SNLIDataset(DatasetExperiment):

    def __init__(self, *args):
        super().__init__(*args)
        dataset = pd.read_csv('{}{}'.format(self.data_dir, 'train_snli.txt'),
                              delimiter='\t', header=None, names=columns, na_values='')
        dataset.dropna(inplace=True)
        dataset = dataset.sample(frac=1, random_state=1).reset_index(drop=True)
        num_instances = len(dataset)
        self.num_train = num_instances * (1 - self.dev_ratio - self.test_ratio)
        self.num_dev = num_instances * self.dev_ratio
        self.num_test = num_instances * self.test_ratio
        self.train = dataset.loc[:self.num_train]
        self.dev = dataset.loc[self.num_train:self.num_train + self.num_dev]
        self.test = dataset.loc[self.num_train + self.num_dev:self.num_train + self.num_dev + self.num_test]

    def train_set(self):
        return self.train

    def train_set_pairs(self):
        return self.train[[ColumnType.sentence1.name, ColumnType.sentence2.name]].as_matrix()

    def train_labels(self):
        return self.train[ColumnType.labels.name].as_matrix()

    def dev_set(self):
        return self.dev

    def dev_set_pairs(self):
        return self.dev[[ColumnType.sentence1.name, ColumnType.sentence2.name]].as_matrix()

    def dev_labels(self):
        return self.dev[ColumnType.labels.name].as_matrix()

    def test_set(self):
        return self.test

    def test_set_pairs(self):
        return self.test[[ColumnType.sentence1.name, ColumnType.sentence2.name]].as_matrix()

    def test_labels(self):
        return self.test[ColumnType.labels.name].as_matrix()

    def _data_path(self):
        return 'corpora/SNLI/'


DATASETS = {
    DatasetType.QQP.name: QQPDataset,
    DatasetType.SNLI.name: SNLIDataset
}


class Dataset:

    def __init__(self, vectorizer, dataset, batch_size):

        self.train_sen1, self.train_sen2 = vectorizer.vectorize_2d(dataset.train_set_pairs())
        self.dev_sen1, self.dev_sen2 = vectorizer.vectorize_2d(dataset.dev_set_pairs())
        self.test_sen1, self.test_sen2 = vectorizer.vectorize_2d(dataset.test_set_pairs())
        self.num_tests = len(dataset.test_set())
        self._train_labels = dataset.train_labels()
        self._dev_labels = dataset.dev_labels()
        self._test_labels = dataset.test_labels()
        self.__shuffle_train_idxs = range(len(self._train_labels))
        self.num_batches = len(self._train_labels) // batch_size

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
        return self.test_sen1, self.test_sen2

    def test_labels(self):
        return self._test_labels

    def dev_instances(self):
        return self.dev_sen1, self.dev_sen2, self._dev_labels

    def num_dev_instances(self):
        return len(self._dev_labels)

    def pick_train_mini_batch(self):
        train_idxs = np.arange(len(self._train_labels))
        np.random.shuffle(train_idxs)
        train_idxs = train_idxs[:self.num_dev_instances()]
        return self.train_sen1[train_idxs], self.train_sen2[train_idxs], self._train_labels[train_idxs]

    def __str__(self):
        return 'Dataset properties:\n ' \
               'Number of training instances: {}\n ' \
               'Number of dev instances: {}\n ' \
               'Number of test instances: {}\n' \
            .format(len(self._train_labels), len(self._dev_labels), len(self._test_labels))
