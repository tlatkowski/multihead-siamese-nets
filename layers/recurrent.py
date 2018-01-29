import tensorflow as tf

rnn_cells = {
    'GRU': tf.nn.rnn_cell.GRUCell,
    'LSTM': tf.nn.rnn_cell.BasicLSTMCell,
}


def rnn_layer(embedded_x, hidden_size, cell_type='GRU', reuse=True):
    with tf.variable_scope('recurrent', reuse=reuse):
        cell = rnn_cells[cell_type]

        fw_rnn_cell = cell(hidden_size)
        bw_rnn_cell = cell(hidden_size)

        rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_rnn_cell,
                                                         bw_rnn_cell,
                                                         embedded_x,
                                                         dtype=tf.float32)
    return rnn_outputs


