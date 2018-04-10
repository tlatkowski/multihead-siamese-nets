import os

import tensorflow as tf


class LogSaver:

    def __init__(self, logs_path, model_name, dateset_name, graph: tf.Graph, scan_train=True):

        if not os.path.isdir(logs_path):
            os.makedirs(logs_path)
        self.dev_summary_writer = tf.summary.FileWriter('{}/{}/{}/dev/'.format(logs_path, dateset_name, model_name),
                                                        graph=graph)
        if scan_train:
            self.train_summary_writer = tf.summary.FileWriter('{}/{}/{}/train/'.format(logs_path, dateset_name, model_name),
                                                              graph=graph)

    def log_dev(self, summary, global_step):
        self.dev_summary_writer.add_summary(summary, global_step)

    def log_train(self, summary, global_step):
        self.train_summary_writer.add_summary(summary, global_step)

