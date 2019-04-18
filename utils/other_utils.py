import configparser
import os

import tensorflow as tf

from utils import constants
from utils.batch_helper import BatchHelper

logger = tf.logging
tf.logging.set_verbosity(tf.logging.INFO)


def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def evaluate_model(model, session, x1, x2, labels, batch_size=100):
    batch_helper = BatchHelper(x1, x2, labels, batch_size)
    num_batches = len(x1) // batch_size
    accuracy = 0.0
    for batch in range(num_batches):
        x1_batch, x2_batch, y_batch = batch_helper.next(batch)
        feed_dict = {model.x1: x1_batch,
                     model.x2: x2_batch,
                     model.is_training: False,
                     model.labels: y_batch}
        accuracy += session.run(model.accuracy, feed_dict=feed_dict)
    accuracy /= num_batches
    return accuracy


def set_visible_gpu(gpu_number: str):
    logger.info('Setting visible GPU to {}'.format(gpu_number))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number


def init_config(specific=None):
    config = configparser.ConfigParser()
    if specific is None:  # return main config
        logger.info('Reading main configuration.')
        config.read(constants.MAIN_CONFIG_PATH)
    else:
        config.read('config/model/{}.ini'.format(specific))
        logger.info('Reading configuration for {} model.'.format(specific))
    return config



