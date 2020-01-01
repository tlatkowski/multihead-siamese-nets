import os
import time
from argparse import ArgumentParser

import tensorflow as tf
from tqdm import tqdm

from data import dataset_type
from data.dataset import Dataset
from models import model_type
from models.model_type import MODELS
from utils import other_utils
from utils.batch_helper import BatchHelper
from utils.config_helpers import MainConfig
from utils.data_utils import DatasetVectorizer
from utils.log_saver import LogSaver
from utils.model_evaluator import ModelEvaluator
from utils.model_saver import ModelSaver

log = tf.logging.info


def create_experiment_name(model_name, main_config, model_config):
    experiment_name = '{}_{}'.format(model_name, main_config['PARAMS']['embedding_size'])
    if model_name == model_type.ModelType.rnn.name:
        experiment_name += ("_" + model_config['PARAMS']['cell_type'])
    
    experiment_name += ("_" + main_config['PARAMS']['loss_function'])
    
    return experiment_name


def train(
        main_config,
        model_config,
        model_name,
        experiment_name,
        dataset_name,
):
    main_cfg = MainConfig(main_config)
    model = MODELS[model_name]
    dataset = dataset_type.get_dataset(dataset_name)
    
    train_data = dataset.train_set_pairs()
    vectorizer = DatasetVectorizer(
        model_dir=main_cfg.model_dir,
        char_embeddings=main_cfg.char_embeddings,
        raw_sentence_pairs=train_data,
    )
    
    dataset_helper = Dataset(vectorizer, dataset, main_cfg.batch_size)
    max_sentence_len = vectorizer.max_sentence_len
    vocabulary_size = vectorizer.vocabulary_size
    
    train_mini_sen1, train_mini_sen2, train_mini_labels = dataset_helper.pick_train_mini_batch()
    train_mini_labels = train_mini_labels.reshape(-1, 1)
    
    test_sentence1, test_sentence2 = dataset_helper.test_instances()
    test_labels = dataset_helper.test_labels()
    test_labels = test_labels.reshape(-1, 1)
    
    num_batches = dataset_helper.num_batches
    
    model = model(
        max_sequence_len=max_sentence_len,
        vocabulary_size=vocabulary_size,
        loss_function=main_cfg.loss_function,
        embedding_size=main_cfg.embedding_size,
        learning_rate=main_cfg.learning_rate,
        model_cfg=model_config,
    )
    model_saver = ModelSaver(
        model_dir=main_cfg.model_dir,
        model_name=experiment_name,
        checkpoints_to_keep=main_cfg.checkpoints_to_keep,
    )
    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=main_cfg.log_device_placement,
    )
    
    with tf.Session(config=config) as session:
        global_step = 0
        init = tf.global_variables_initializer()
        session.run(init)
        log_saver = LogSaver(
            main_cfg.logs_path,
            experiment_name,
            dataset_name,
            session.graph,
        )
        model_evaluator = ModelEvaluator(model, session)
        
        metrics = {'acc': 0.0}
        time_per_epoch = []
        
        log('Training model for {} epochs'.format(main_cfg.num_epochs))
        for epoch in tqdm(range(main_cfg.num_epochs), desc='Epochs'):
            start_time = time.time()
            
            train_sentence1, train_sentence2 = dataset_helper.train_instances(shuffle=True)
            train_labels = dataset_helper.train_labels()
            
            train_batch_helper = BatchHelper(
                train_sentence1,
                train_sentence2,
                train_labels,
                main_cfg.batch_size,
            )
            
            # small eval set for measuring dev accuracy
            dev_sentence1, dev_sentence2, dev_labels = dataset_helper.dev_instances()
            dev_labels = dev_labels.reshape(-1, 1)
            
            tqdm_iter = tqdm(range(num_batches), total=num_batches, desc="Batches", leave=False,
                             postfix=metrics)
            for batch in tqdm_iter:
                global_step += 1
                sentence1_batch, sentence2_batch, labels_batch = train_batch_helper.next(batch)
                feed_dict_train = {
                    model.x1: sentence1_batch,
                    model.x2: sentence2_batch,
                    model.is_training: True,
                    model.labels: labels_batch,
                }
                loss, _ = session.run([model.loss, model.opt], feed_dict=feed_dict_train)
                
                if batch % main_cfg.eval_every == 0:
                    feed_dict_train = {
                        model.x1: train_mini_sen1,
                        model.x2: train_mini_sen2,
                        model.is_training: False,
                        model.labels: train_mini_labels,
                    }
                    
                    train_accuracy, train_summary = session.run(
                        [model.accuracy, model.summary_op],
                        feed_dict=feed_dict_train,
                    )
                    log_saver.log_train(train_summary, global_step)
                    
                    feed_dict_dev = {
                        model.x1: dev_sentence1,
                        model.x2: dev_sentence2,
                        model.is_training: False,
                        model.labels: dev_labels
                    }
                    
                    dev_accuracy, dev_summary = session.run(
                        [model.accuracy, model.summary_op],
                        feed_dict=feed_dict_dev,
                    )
                    log_saver.log_dev(dev_summary, global_step)
                    tqdm_iter.set_postfix(
                        dev_acc='{:.2f}'.format(float(dev_accuracy)),
                        train_acc='{:.2f}'.format(float(train_accuracy)),
                        loss='{:.2f}'.format(float(loss)),
                        epoch=epoch
                    )
                
                if global_step % main_cfg.save_every == 0:
                    model_saver.save(session, global_step=global_step)
            
            model_evaluator.evaluate_dev(
                x1=dev_sentence1,
                x2=dev_sentence2,
                labels=dev_labels,
            )
            
            end_time = time.time()
            total_time = other_utils.timer(start_time, end_time)
            time_per_epoch.append(total_time)
            
            model_saver.save(session, global_step=global_step)
        
        model_evaluator.evaluate_test(test_sentence1, test_sentence2, test_labels)
        model_evaluator.save_evaluation(
            model_path='{}/{}'.format(
                main_cfg.model_dir,
                experiment_name,
            ),
            epoch_time=time_per_epoch[-1],
            dataset=dataset,
        )


def predict(
        main_config,
        model_config,
        model,
        experiment_name,
):
    model = MODELS[model]
    main_cfg = MainConfig(main_config)
    
    # model_dir = str(main_config['DATA']['model_dir'])
    
    vectorizer = DatasetVectorizer(
        model_dir=main_cfg.model_dir,
        char_embeddings=main_cfg.char_embeddings,
    )
    
    max_doc_len = vectorizer.max_sentence_len
    vocabulary_size = vectorizer.vocabulary_size
    
    model = model(max_doc_len, vocabulary_size, main_config, model_config)
    
    with tf.Session() as session:
        saver = tf.train.Saver()
        last_checkpoint = tf.train.latest_checkpoint(
            '{}/{}'.format(
                main_cfg.model_dir,
                experiment_name,
            )
        )
        saver.restore(session, last_checkpoint)
        while True:
            x1 = input('First sentence:')
            x2 = input('Second sentence:')
            x1_sen = vectorizer.vectorize(x1)
            x2_sen = vectorizer.vectorize(x2)
            
            feed_dict = {model.x1: x1_sen, model.x2: x2_sen, model.is_training: False}
            prediction = session.run([model.temp_sim], feed_dict=feed_dict)
            print(prediction)


def main():
    parser = ArgumentParser()
    
    parser.add_argument(
        'model',
        choices=['rnn', 'cnn', 'multihead'],
        help='model to be used',
    )
    
    parser.add_argument(
        'dataset',
        choices=['QQP', 'SNLI', 'ANLI'],
        nargs='?',
        help='dataset to be used',
    )
    
    parser.add_argument(
        '--experiment_path',
        type=str,
        help='path to experiments',
    )
    
    parser.add_argument(
        '--gpu',
        default='0',
        help='index of GPU to be used (default: %(default))',
    )
    
    args = parser.parse_args()
    other_utils.set_visible_gpu(args.gpu)
    
    main_config = other_utils.read_config()
    model_configs = other_utils.read_experiment_configs(args.experiment_path)
    
    for model_config, experiment_name in model_configs:
        tf.reset_default_graph()
        experiment_name = os.path.basename(experiment_name).split('.ini')[0]
        log('Running experiment {} '.format(experiment_name))
        train(main_config, model_config, args.model, experiment_name, args.dataset)


if __name__ == '__main__':
    main()
