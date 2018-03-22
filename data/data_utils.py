from tflearn.data_utils import VocabularyProcessor
import numpy as np


def vectorize_data(raw_sentence_pairs, model_dir, save_vocab=True):
    num_instances, num_classes = raw_sentence_pairs.shape
    raw_sentence_pairs = raw_sentence_pairs.ravel()

    raw_sentence_pairs = [str(x) for x in list(raw_sentence_pairs)]
    sentences_lengths = [len(str(x).split(' ')) for x in list(raw_sentence_pairs)]
    max_sentence_length = max(sentences_lengths)
    vocabulary = VocabularyProcessor(max_sentence_length)
    vectorized_sentence_pairs = np.array(list(vocabulary.fit_transform(raw_sentence_pairs)))

    if save_vocab:
        vocabulary.save('{}/vocab'.format(model_dir))

    vectorized_sentence_pairs = vectorized_sentence_pairs.reshape(num_instances, num_classes, max_sentence_length)

    vectorized_sentence1 = vectorized_sentence_pairs[:, 0, :]
    vectorized_sentence2 = vectorized_sentence_pairs[:, 1, :]
    return vectorized_sentence1, vectorized_sentence2, vocabulary, sentences_lengths

