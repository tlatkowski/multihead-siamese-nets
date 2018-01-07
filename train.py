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
num_epochs = 10
batch_size = 128
eval_every = 10
# ++++++++++++++++++++++++++++++++++

sen1, sen2, labels, max_doc_len, vocabulary_size = load_snli(data_fn)
num_batches = len(labels) // batch_size
print('Num batches: ', num_batches)

train_sen1 = sen1[:-1000]
train_sen2 = sen2[:-1000]
train_labels = labels[:-1000]

eval_sen1 = sen1[-1000:]
eval_sen2 = sen2[-1000:]
eval_labels = labels[-1000:]

with tf.Session() as session:
    model = LSTMBasedSiameseNet(max_doc_len, vocabulary_size, embedding_size, hidden_size)
    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    session.run(init)
    session.run(init_local)
    if not os.path.isdir(logs_path):
        os.makedirs(logs_path)
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    metrics = {'acc': 0.0}
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        tqdm_iter = tqdm(range(num_batches),
                                total=num_batches,
                                desc="Batches",
                                leave=False,
                                postfix=metrics)
        for batch in tqdm_iter:
            x1_batch = train_sen1[batch * batch_size:(batch + 1) * batch_size]
            x2_batch = train_sen2[batch * batch_size:(batch + 1) * batch_size]
            y_batch = train_labels[batch * batch_size:(batch+1) * batch_size]
            feed_dict = {model.x1: x1_batch, model.x2: x2_batch, model.labels: y_batch}
            loss, _ = session.run([model.loss, model.opt], feed_dict=feed_dict)
            if batch % eval_every == 0:
                feed_dict = {model.x1: eval_sen1, model.x2: eval_sen2, model.labels: eval_labels}
                acc = session.run([model.acc], feed_dict=feed_dict)
                tqdm_iter.set_postfix(acc=acc)





