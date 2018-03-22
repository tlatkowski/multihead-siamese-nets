import logging
import os
from enum import Enum

import numpy as np
import pandas as pd

from data.data_utils import vectorize_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ColumnType(Enum):
    sentence1 = 0,
    sentence2 = 1,
    labels = 2


columns = [ColumnType.sentence1.name,
           ColumnType.sentence2.name,
           ColumnType.labels.name]


def load_snli(data_fn, model_dir, save_vocab=True):
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    logger.info('Loading corpus from {}'.format(model_dir))
    corpus = pd.read_csv(data_fn, delimiter='\t', header=None,
                         names=columns, na_values='')
    logger.info('Loaded corpus from {}'.format(model_dir))
    raw_sentence_pairs = corpus[columns[:2]].as_matrix()
    sen1, sen2, vocab_processor, sen_lengths = vectorize_data(raw_sentence_pairs, model_dir, save_vocab)
    labels = corpus[columns[-1]].as_matrix()
    labels = np.reshape(labels, (-1, 1))
    return sen1, sen2, labels, vocab_processor, sen_lengths


def data_split():
    pass
