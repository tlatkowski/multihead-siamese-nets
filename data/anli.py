import os

import jsonlines
import pandas as pd

from data import dataset


class ANLIDataset(dataset.DatasetExperiment):
    
    def __init__(self, *args):
        super().__init__(*args)
        self.hypothesis = []
        self.reason = []
        self.label = []
        with jsonlines.open(os.path.join(self.data_dir, 'train.jsonl')) as jsonl_reader:
            for instance in jsonl_reader:
                if instance['label'] is 'n':
                    continue
                self.hypothesis.append(instance['hypothesis'])
                self.reason.append(instance['reason'])
                if instance['label'] is 'e':
                    self.label.append(0)
                else:
                    self.label.append(1)
        
        dataset = pd.DataFrame(
            list(
                zip(
                    self.hypothesis,
                    self.reason,
                    self.label,
                )
            ),
            columns=['hypothesis', 'reason', 'label']
        )
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
        return self.train[['hypothesis', 'reason']].as_matrix()
    
    def train_labels(self):
        return self.train['label'].as_matrix()
    
    def dev_set(self):
        return self.dev
    
    def dev_set_pairs(self):
        return self.dev[['hypothesis', 'reason']].as_matrix()
    
    def dev_labels(self):
        return self.dev['label'].as_matrix()
    
    def test_set(self):
        return self.test
    
    def test_set_pairs(self):
        return self.test[['hypothesis', 'reason']].as_matrix()
    
    def test_labels(self):
        return self.test['label'].as_matrix()
    
    def _data_path(self):
        return 'corpora/ANLI/anli_v0.1/R3'
