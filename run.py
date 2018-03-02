import configparser
import os
import time
from argparse import ArgumentParser

import tensorflow as tf
from tqdm import tqdm

from data.data import ParaphraseData
from data.dataset import Dataset
from models.cnn import CnnSiameseNet
from models.lstm import LSTMBasedSiameseNet
from models.multihead_attention import MultiheadAttentionSiameseNet
from utils.log_saver import LogSaver
from utils.model_saver import ModelSaver

models = {
    'cnn': CnnSiameseNet,
    'rnn': LSTMBasedSiameseNet,
    'multihead': MultiheadAttentionSiameseNet
}


def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def train(main_config, model, model_cfg, model_name):
    num_epochs = int(main_config['TRAINING']['num_epochs'])
    batch_size = int(main_config['TRAINING']['batch_size'])
    eval_every = int(main_config['TRAINING']['eval_every'])
    checkpoints_to_keep = int(main_config['TRAINING']['checkpoints_to_keep'])
    save_every = int(main_config['TRAINING']['save_every'])
    eval_size = int(main_config['TRAINING']['eval_size'])
    log_device_placement = bool(main_config['TRAINING']['log_device_placement'])

    num_tests = int(main_config['DATA']['num_tests'])
    data_fn = str(main_config['DATA']['file_name'])
    logs_path = str(main_config['DATA']['logs_path'])
    model_dir = str(main_config['DATA']['model_dir'])

    paraphrase_data = ParaphraseData(model_dir, data_fn, force_save=True)
    snli_dataset = Dataset(paraphrase_data, num_tests, batch_size)
    max_sentence_len = paraphrase_data.max_sentence_len
    vocabulary_size = paraphrase_data.vocabulary_size

    print(snli_dataset)

    test_sen1, test_sen2 = snli_dataset.test_instances()
    test_labels = snli_dataset.test_labels()

    num_batches = snli_dataset.num_batches

    model = model(max_sentence_len, vocabulary_size, main_config, model_cfg)

    model_saver = ModelSaver(model_dir, model_name, checkpoints_to_keep)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_device_placement)

    with tf.Session(config=config) as session:
        global_step = 0

        init = tf.global_variables_initializer()

        session.run(init)

        log_saver = LogSaver(logs_path, model_name, session.graph)

        metrics = {'acc': 0.0}

        test_acc_per_epoch = []
        time_per_epoch = []
        for epoch in tqdm(range(num_epochs), desc='Epochs'):
            start_time = time.time()

            train_sen1, train_sen2 = snli_dataset.train_instances(shuffle=True)
            train_labels = snli_dataset.train_labels()

            # ++++++++
            # small eval set for measuring train accuracy
            val_sen1, val_sen2, val_labels = snli_dataset.validation_instances(eval_size)

            tqdm_iter = tqdm(range(num_batches), total=num_batches, desc="Batches", leave=False, postfix=metrics)

            for batch in tqdm_iter:
                global_step += 1
                x1_batch = train_sen1[batch * batch_size:(batch + 1) * batch_size]
                x2_batch = train_sen2[batch * batch_size:(batch + 1) * batch_size]
                y_batch = train_labels[batch * batch_size:(batch + 1) * batch_size]
                feed_dict = {model.x1: x1_batch, model.x2: x2_batch, model.labels: y_batch}
                loss, _ = session.run([model.loss, model.opt], feed_dict=feed_dict)

                if batch % eval_every == 0:
                    feed_dict = {model.x1: val_sen1, model.x2: val_sen2, model.labels: val_labels}
                    train_accuracy, train_summary, loss = session.run([model.accuracy, model.summary_op, model.loss],
                                                                      feed_dict=feed_dict)
                    log_saver.log_train(train_summary, global_step)

                    test_small_batch = 5000

                    # feed_dict = {model.x1: test_sen1[:test_small_batch],
                    #              model.x2: test_sen2[:test_small_batch],
                    #              model.labels: test_labels[:test_small_batch]}
                    feed_dict = {model.x1: test_sen1,
                                 model.x2: test_sen2,
                                 model.labels: test_labels}
                    test_accuracy, test_summary = session.run([model.accuracy, model.summary_op], feed_dict=feed_dict)
                    log_saver.log_test(test_summary, global_step)

                    tqdm_iter.set_postfix(
                        train_test_acc='{:.2f}|{:.2f}'.format(float(train_accuracy), float(test_accuracy)),
                        loss='{:.2f}'.format(float(loss)),
                        epoch=epoch)

                if global_step % save_every == 0:
                    model_saver.save(session, global_step=global_step)

            # eval entire test set
            split_size = 100
            num_test_batches = len(test_sen1) // split_size
            all_test_accuracy = 0.0
            for b_id in range(num_test_batches):
                feed_dict = {model.x1: test_sen1[b_id * split_size: (b_id + 1) * split_size],
                             model.x2: test_sen2[b_id * split_size: (b_id + 1) * split_size],
                             model.labels: test_labels[b_id * split_size: (b_id + 1) * split_size]}
                all_test_accuracy += session.run(model.accuracy, feed_dict=feed_dict)

            all_test_accuracy /= num_test_batches
            test_acc_per_epoch.append(all_test_accuracy)

            end_time = time.time()
            total_time = timer(start_time, end_time)
            time_per_epoch.append(total_time)

            model_saver.save(session, global_step=global_step)

        print(test_acc_per_epoch)
        print(time_per_epoch)


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

    parser.add_argument('--gpu',
                        default='0',
                        help='index of GPU to be used')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    main_config = configparser.ConfigParser()
    main_config.read('config/main.ini')

    model_cfg = configparser.ConfigParser()
    model_cfg.read('config/model/{}.ini'.format(args.model))

    model = models[args.model]
    mode = args.mode

    model_name = '{}_{}'.format(args.model,
                                main_config['PARAMS']['embedding_size'])
    if 'train' in mode:
        train(main_config, model, model_cfg, model_name)
    else:
        predict(model_name, model, main_config, model_cfg)


if __name__ == '__main__':
    main()
