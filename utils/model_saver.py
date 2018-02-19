import os

import tensorflow as tf


class ModelSaver:

    def __init__(self, model_dir, model_name, checkpoints_to_keep=10):
        self.model_saver = tf.train.Saver(max_to_keep=checkpoints_to_keep)
        self.model_path = '{}/{}/model'.format(model_dir, model_name)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

    def save(self, session, global_step):
        self.model_saver.save(session, self.model_path, global_step=global_step)
