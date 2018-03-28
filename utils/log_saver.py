import os

import tensorflow as tf


class LogSaver:

    def __init__(self, logs_path, model_name, dateset_name, graph: tf.Graph):

        if not os.path.isdir(logs_path):
            os.makedirs(logs_path)
        self.test_summary_writer = tf.summary.FileWriter('{}/{}/{}/test/'.format(logs_path, dateset_name, model_name),
                                                         graph=graph)
        self.train_summary_writer = tf.summary.FileWriter('{}/{}/{}/train/'.format(logs_path, dateset_name, model_name),
                                                          graph=graph)

    def log_test(self, summary, global_step):
        self.test_summary_writer.add_summary(summary, global_step)

    def log_train(self, summary, global_step):
        self.test_summary_writer.add_summary(summary, global_step)
