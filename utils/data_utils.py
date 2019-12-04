import logging
import os

import numpy as np
from tflearn.data_utils import VocabularyProcessor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DatasetVectorizer:
    
    def __init__(self, model_dir, char_embeddings, raw_sentence_pairs=None, save_vocab=True):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        if raw_sentence_pairs is None:
            self.restore()
        else:
            raw_sentence_pairs = raw_sentence_pairs.ravel()
            raw_sentence_pairs = [str(x) for x in list(raw_sentence_pairs)]
            if char_embeddings:
                self.sentences_lengths = [len(list(str(x))) for x in list(raw_sentence_pairs)]
            else:
                self.sentences_lengths = [len(str(x).split(' ')) for x in list(raw_sentence_pairs)]
            max_sentence_length = max(self.sentences_lengths)
            if char_embeddings:
                self.vocabulary = VocabularyProcessor(
                    max_document_length=max_sentence_length,
                    tokenizer_fn=char_tokenizer,
                )
            else:
                self.vocabulary = VocabularyProcessor(
                    max_document_length=max_sentence_length,
                )
            self.vocabulary.fit(raw_sentence_pairs)
            if save_vocab:
                self.vocabulary.save('{}/vocab'.format(self.model_dir))
    
    @property
    def max_sentence_len(self):
        return self.vocabulary.max_document_length
    
    @property
    def vocabulary_size(self):
        return len(self.vocabulary.vocabulary_._mapping)
    
    def restore(self):
        self.vocabulary = VocabularyProcessor.restore('{}/vocab'.format(self.model_dir))
    
    def vectorize(self, sentence):
        return np.array(list(self.vocabulary.transform([sentence])))
    
    def vectorize_2d(self, raw_sentence_pairs):
        num_instances, num_classes = raw_sentence_pairs.shape
        raw_sentence_pairs = raw_sentence_pairs.ravel()
        
        for i, v in enumerate(raw_sentence_pairs):
            if v is np.nan:
                print(i, v)
        
        vectorized_sentence_pairs = np.array(list(self.vocabulary.transform(raw_sentence_pairs)))
        
        vectorized_sentence_pairs = vectorized_sentence_pairs.reshape(num_instances, num_classes,
                                                                      self.max_sentence_len)
        
        vectorized_sentence1 = vectorized_sentence_pairs[:, 0, :]
        vectorized_sentence2 = vectorized_sentence_pairs[:, 1, :]
        return vectorized_sentence1, vectorized_sentence2


def char_tokenizer(iterator):
    """Tokenizer generator.
  
    Args:
      iterator: Input iterator with strings.
  
    Yields:
      array of tokens per each value in the input.
    """
    for value in iterator:
        yield list(value)
