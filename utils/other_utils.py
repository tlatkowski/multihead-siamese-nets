import configparser
import os

import tensorflow as tf

from utils import constants
from utils.batch_helper import BatchHelper

PATH_TO_COMMON_EXPERIMENTS = 'config/model'

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
        feed_dict = {
            model.x1: x1_batch,
            model.x2: x2_batch,
            model.is_training: False,
            model.labels: y_batch,
        }
        accuracy += session.run(model.accuracy, feed_dict=feed_dict)
    accuracy /= num_batches
    return accuracy


def set_visible_gpu(gpu_number: str):
    logger.info('Setting visible GPU to {}'.format(gpu_number))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number


def read_config(specific=None):
    config = configparser.ConfigParser()
    if specific is None:  # return main config
        logger.info('Reading main configuration.')
        config.read(constants.MAIN_CONFIG_PATH)
    else:
        logger.info('Reading configuration for {} model.'.format(specific))
        model_path_config = os.path.join(PATH_TO_COMMON_EXPERIMENTS, '{}.ini'.format(specific))
        config.read(model_path_config)
    return config


def read_experiment_configs(path):
    logger.info('Reading experiment configuration from {} path.'.format(path))
    config_experiments = os.listdir(path)
    full_config_paths = [os.path.join(path, c) for c in config_experiments]
    configs = []
    for full_config_path in full_config_paths:
        config = configparser.ConfigParser()
        config.read(full_config_path)
        configs.append(config)
    return zip(configs, full_config_paths)
