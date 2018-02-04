from data.data_loader import load_snli
from tflearn.data_utils import VocabularyProcessor
import numpy as np


class ParaphraseData:

    def __init__(self, model_dir, data_fn=None, force_save=False):
        self.model_dir = model_dir
        if force_save:
            self.sen1, self.sen2, self.labels, self.vocabulary_processor = load_snli(data_fn, model_dir)
        else:
            self.restore()

    @property
    def max_sentence_len(self):
        return self.vocabulary_processor.max_document_length

    @property
    def vocabulary_size(self):
        return len(self.vocabulary_processor.vocabulary_._mapping)

    def restore(self):
        self.vocabulary_processor = VocabularyProcessor.restore('{}/vocab'.format(self.model_dir))

    def vectorize(self, sentence):
        np.array(list(self.vocab_processor.transform([sentence])))