import configparser
import os
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from tflearn.data_utils import VocabularyProcessor
from tqdm import tqdm
from data.data import ParaphraseData
from data.dataset import Dataset
from models.cnn import CnnSiameseNet
from models.lstm import LSTMBasedSiameseNet
from models.multihead_attention import MultiheadAttentionSiameseNet

models = {
    'cnn': CnnSiameseNet,
    'rnn': LSTMBasedSiameseNet,
    'multihead': MultiheadAttentionSiameseNet
}


def train(config, model, model_cfg, model_name):

    num_epochs = int(config['TRAINING']['num_epochs'])
    batch_size = int(config['TRAINING']['batch_size'])
    eval_every = int(config['TRAINING']['eval_every'])
    checkpoints_to_keep = int(config['TRAINING']['checkpoints_to_keep'])
    save_every = int(config['TRAINING']['save_every'])

    num_tests = int(config['DATA']['num_tests'])
    data_fn = str(config['DATA']['file_name'])
    logs_path = str(config['DATA']['logs_path'])
    model_dir = str(config['DATA']['model_dir'])

    paraphrase_data = ParaphraseData(model_dir, data_fn, force_save=True)
    snli_dataset = Dataset(paraphrase_data, num_tests, batch_size)
    max_sentence_len = paraphrase_data.max_sentence_len
    vocabulary_size = paraphrase_data.vocabulary_size

    test_sen1, test_sen2 = snli_dataset.test_instances()
    test_labels = snli_dataset.test_labels()

    num_batches = snli_dataset.num_batches

    model = model(max_sentence_len, vocabulary_size, config, model_cfg)

    model_saver = tf.train.Saver(max_to_keep=checkpoints_to_keep)
    model_path = '{}/{}/model'.format(model_dir, model_name)

    with tf.Session() as session:
        global_step = 0

        init = tf.global_variables_initializer()
        init_local = tf.local_variables_initializer()

        session.run(init)
        session.run(init_local)
        if not os.path.isdir(logs_path):
            os.makedirs(logs_path)
        test_summary_writer = tf.summary.FileWriter('{}/test/'.format(logs_path), graph=session.graph)
        train_summary_writer = tf.summary.FileWriter('{}/train/'.format(logs_path), graph=session.graph)

        metrics = {'acc': 0.0}
        for epoch in tqdm(range(num_epochs), desc='Epochs'):

            train_sen1, train_sen2 = snli_dataset.train_instances(shuffle=True)
            train_labels = snli_dataset.train_labels()

            # ++++++++
            # small train set for measuring train accuracy
            eval_size = 1000
            val_sen1, val_sen2, val_labels = snli_dataset.validation_instances(eval_size)

            tqdm_iter = tqdm(range(num_batches), total=num_batches, desc="Batches", leave=False, postfix=metrics)

            for batch in tqdm_iter:
                global_step += 1
                x1_batch = train_sen1[batch * batch_size:(batch + 1) * batch_size]
                x2_batch = train_sen2[batch * batch_size:(batch + 1) * batch_size]
                y_batch = train_labels[batch * batch_size:(batch+1) * batch_size]
                feed_dict = {model.x1: x1_batch, model.x2: x2_batch, model.labels: y_batch}
                loss, _ = session.run([model.loss, model.opt], feed_dict=feed_dict)
                if batch % eval_every == 0:
                    feed_dict = {model.x1: val_sen1, model.x2: val_sen2, model.labels: val_labels}
                    train_accuracy, train_summary, loss = session.run([model.accuracy, model.summary_op, model.loss],
                                                                      feed_dict=feed_dict)
                    train_summary_writer.add_summary(train_summary, global_step)

                    feed_dict = {model.x1: test_sen1, model.x2: test_sen2, model.labels: test_labels}
                    test_accuracy, test_summary = session.run([model.accuracy, model.summary_op], feed_dict=feed_dict)
                    test_summary_writer.add_summary(test_summary, global_step)
                    tqdm_iter.set_postfix(train_test_acc='{:.2f}|{:.2f}'.format(float(train_accuracy), float(test_accuracy)),
                                          loss='{:.2f}'.format(float(loss)),
                                          epoch=epoch)

                if global_step % save_every == 0:
                    model_saver.save(session, model_path, global_step=global_step)

            model_saver.save(session, model_path, global_step=global_step)


def predict(model_name, model, config, model_cfg):
    model_dir = str(config['DATA']['model_dir'])

    paraphrase_data = ParaphraseData(model_dir)

    max_doc_len = paraphrase_data.max_sentence_len
    vocabulary_size = paraphrase_data.vocabulary_size

    model = model(max_doc_len, vocabulary_size, config, model_cfg)

    with tf.Session() as session:
        saver = tf.train.Saver()
        last_checkpoint = tf.train.latest_checkpoint('{}/{}/model'.format(model_dir, model_name))
        saver.restore(session, last_checkpoint)
        while True:
            x1 = input('First sentence:')
            x2 = input('Second sentence:')
            x1_sen = paraphrase_data.vectorize(x1)
            x2_sen = paraphrase_data.vectorize(x2)

            feed_dict = {model.x1: x1_sen, model.x2: x2_sen}
            prediction = session.run([model.temp_sim], feed_dict=feed_dict)
            print(prediction)


def main():

    parser = ArgumentParser()

    parser.add_argument('mode',
                        default='train',
                        choices=['train', 'predict'],
                        help='model mode (default: %(default))')

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
    mode = args.mode
    if 'train' in mode:
        train(main_config, model, model_cfg, args.model)
    else:
        predict(args.model, model, main_config, model_cfg)


if __name__ == '__main__':
    main()

