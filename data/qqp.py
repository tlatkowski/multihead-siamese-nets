import pandas as pd

from data import dataset


class QQPDataset(dataset.DatasetExperiment):
    
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
        self.test = dataset.loc[
                    self.num_train + self.num_dev:self.num_train + self.num_dev + self.num_test]
    
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
