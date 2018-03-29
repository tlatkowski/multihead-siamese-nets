import configparser

from utils.batch_helper import BatchHelper


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
        feed_dict = {model.x1: x1_batch, model.x2: x2_batch, model.labels: y_batch}
        accuracy += session.run(model.accuracy, feed_dict=feed_dict)
    accuracy /= num_batches
    return accuracy


def save_model_eval(model_path, mean_dev_acc, last_dev_acc, test_acc, epoch_time, dataset):
    config = configparser.ConfigParser()
    config.add_section('EVALUATION')
    config.set('EVALUATION', 'MEAN_DEV_ACC', str(mean_dev_acc))
    config.set('EVALUATION', 'LAST_DEV_ACC', str(last_dev_acc))
    config.set('EVALUATION', 'TEST_ACC', str(test_acc))
    config.set('EVALUATION', 'EPOCH_TIME', str(epoch_time))
    config.set('EVALUATION', 'NUM_TRAINS', str(len(dataset.train_labels())))
    config.set('EVALUATION', 'NUM_DEVS', str(len(dataset.dev_labels())))
    config.set('EVALUATION', 'NUM_TESTS', str(len(dataset.test_labels())))

    with open('{}/evaluation.ini'.format(model_path), 'w') as configfile:  # save
        config.write(configfile)
