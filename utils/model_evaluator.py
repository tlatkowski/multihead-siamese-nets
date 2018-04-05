import configparser

from utils.batch_helper import BatchHelper


class ModelEvaluator:

    def __init__(self, model, session):
        self._model = model
        self._session = session
        self.dev_accuracies = []
        self.test_accuracies = []

    def _evaluate(self, x1, x2, labels, batch_size=100):
        batch_helper = BatchHelper(x1, x2, labels, batch_size)
        num_batches = len(x1) // batch_size
        accuracy = 0.0
        for batch in range(num_batches):
            x1_batch, x2_batch, y_batch = batch_helper.next(batch)
            feed_dict = {self._model.x1: x1_batch,
                         self._model.x2: x2_batch,
                         self._model.is_training: False,
                         self._model.labels: y_batch}
            accuracy += self._session.run(self._model.accuracy, feed_dict=feed_dict)
        accuracy /= num_batches
        return accuracy

    def evaluate_dev(self, x1, x2, labels):
        dev_accuracy = self._evaluate(x1, x2, labels)
        self.dev_accuracies.append(dev_accuracy)
        return dev_accuracy

    def evaluate_test(self, x1, x2, labels):
        test_accuracy = self._evaluate(x1, x2, labels)
        self.test_accuracies.append(test_accuracy)
        return test_accuracy

    def save_evaluation(self, model_path, epoch_time, dataset):
        mean_dev_acc = sum(self.dev_accuracies) / len(self.dev_accuracies)
        last_dev_acc = self.dev_accuracies[-1]
        test_acc = self.test_accuracies[-1]

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
