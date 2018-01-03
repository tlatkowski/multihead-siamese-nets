import os

import tensorflow as tf
from tqdm import tqdm

from data.data_loader import load_snli
from models.lstm import LSTMBasedSiameseNet

data_fn = 'train_snli.txt'
logs_path = 'logs/'

# ++++++++++++++++++++++++++++++++++
embedding_size = 64
hidden_size = 256
num_epochs = 100
batch_size = 32
eval_every = 10
# ++++++++++++++++++++++++++++++++++

sen1, sen2, labels, max_doc_len, vocabulary_size = load_snli(data_fn)
num_batches = len(labels) // batch_size
print('Num batches: ', num_batches)

with tf.Session() as session:
    model = LSTMBasedSiameseNet(max_doc_len, vocabulary_size, embedding_size, hidden_size, 100)
    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    session.run(init)
    session.run(init_local)
    if not os.path.isdir(logs_path):
        os.makedirs(logs_path)
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    for epoch in range(num_epochs):
        tqdm_iter = tqdm(range(num_batches))
        for batch in range(num_batches):
            x1_batch = sen1[batch * batch_size:(batch + 1) * batch_size]
            x2_batch = sen2[batch * batch_size:(batch + 1) * batch_size]
            y_batch = labels[batch * batch_size:(batch+1) * batch_size]
            feed_dict = {model.x1: x1_batch, model.x2: x2_batch, model.labels: y_batch}
            my_opt, summary = session.run([model.loss, model.opt], feed_dict=feed_dict)
            print(my_opt)
            if batch % eval_every == 0:
                print('count')
                feed_dict = {model.x1: sen1, model.x2: sen2, model.labels: labels}
                acc = session.run([model.acc], feed_dict=feed_dict)
                tqdm_iter.set_postfix(acc=acc,
                                      batches='{}/{}'.format(batch, num_batches),
                                      epochs='{}/{}'.format(epoch, num_epochs))



