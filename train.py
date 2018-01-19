import configparser
from argparse import ArgumentParser
import os

import tensorflow as tf
from tqdm import tqdm

from data.dataset import Dataset
from models.multihead_attention import ScaledDotProductAttentionSiameseNet
from models.cnn import CNNbasedSiameseNet
from models.lstm import LSTMBasedSiameseNet

data_fn = 'train_snli.txt'
logs_path = 'logs/'

# TODO make generic class
models = {
    'cnn': CNNbasedSiameseNet,
    'rnn': LSTMBasedSiameseNet,
    'multihead': ScaledDotProductAttentionSiameseNet
}


def train(config, model, model_cfg):

    num_tests = 1000
    snli_dataset = Dataset(data_fn, num_tests)
    max_doc_len = snli_dataset.max_doc_len
    vocabulary_size = snli_dataset.vocabulary_size

    test_sen1, test_sen2 = snli_dataset.test_instances()
    test_labels = snli_dataset.test_labels()

    num_epochs = int(config['TRAINING']['num_epochs'])
    batch_size = int(config['TRAINING']['batch_size'])
    eval_every = int(config['TRAINING']['eval_every'])

    num_batches = len(snli_dataset.labels - num_tests) // batch_size

    with tf.Session() as session:
        model = model(max_doc_len, vocabulary_size, config, model_cfg)
        global_step = 0

        init = tf.global_variables_initializer()
        init_local = tf.local_variables_initializer()
        session.run(init)
        session.run(init_local)
        if not os.path.isdir(logs_path):
            os.makedirs(logs_path)
        eval_summary_writer = tf.summary.FileWriter(logs_path + 'test', graph=session.graph)
        train_summary_writer = tf.summary.FileWriter(logs_path + 'eval', graph=session.graph)

        metrics = {'acc': 0.0}
        for epoch in tqdm(range(num_epochs), desc="Epochs"):
            tqdm_iter = tqdm(range(num_batches),
                                    total=num_batches,
                                    desc="Batches",
                                    leave=False,
                                    postfix=metrics)

            train_sen1, train_sen2 = snli_dataset.train_instances(shuffle=True)
            train_labels = snli_dataset.train_labels()

            # ++++++++
            # small train set for measuring train accuracy
            eval_size = 1000
            small_train1 = train_sen1[-eval_size:]
            small_train2 = train_sen2[-eval_size:]
            small_train_labels = train_labels[-eval_size:]

            for batch in tqdm_iter:
                global_step += 1
                x1_batch = train_sen1[batch * batch_size:(batch + 1) * batch_size]
                x2_batch = train_sen2[batch * batch_size:(batch + 1) * batch_size]
                y_batch = train_labels[batch * batch_size:(batch+1) * batch_size]
                feed_dict = {model.x1: x1_batch, model.x2: x2_batch, model.labels: y_batch}
                loss, _ = session.run([model.loss, model.opt], feed_dict=feed_dict)
                if batch % eval_every == 0:
                    feed_dict = {model.x1: small_train1, model.x2: small_train2, model.labels: small_train_labels}
                    train_summary = session.run(model.summary_op, feed_dict=feed_dict)
                    train_summary_writer.add_summary(train_summary, global_step)

                    feed_dict = {model.x1: test_sen1, model.x2: test_sen2, model.labels: test_labels}
                    accuracy, summary = session.run([model.accuracy, model.summary_op], feed_dict=feed_dict)
                    eval_summary_writer.add_summary(summary, global_step)
                    tqdm_iter.set_postfix(acc=accuracy)


def main():

    parser = ArgumentParser()
    parser.add_argument('model',
                             default='multihead',
                             choices=['rnn', 'cnn', 'multihead'],
                             help='model used during training (default: %(default))')
    args = parser.parse_args()
    main_config = configparser.ConfigParser()
    main_config.read('config/config.ini')

    model_cfg = configparser.ConfigParser()
    model_cfg.read('config/model/{}.ini'.format(args.model))

    model = models[args.model]

    train(main_config, model, model_cfg)


if __name__ == '__main__':
    main()

