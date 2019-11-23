import tensorflow as tf

rnn_cells = {
    'GRU': tf.nn.rnn_cell.GRUCell,
    'LSTM': tf.nn.rnn_cell.BasicLSTMCell,
}


def rnn_layer(
        embedded_x,
        hidden_size,
        bidirectional,
        cell_type='GRU',
        reuse=False,
):
    with tf.variable_scope('recurrent', reuse=reuse):
        cell = rnn_cells[cell_type]

        fw_rnn_cell = cell(hidden_size)

        if bidirectional:
            bw_rnn_cell = cell(hidden_size)
            rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_rnn_cell,
                bw_rnn_cell,
                embedded_x,
                dtype=tf.float32,
            )
            output = tf.concat([rnn_outputs[0], rnn_outputs[1]], axis=2)
        else:
            output, _ = tf.nn.dynamic_rnn(
                fw_rnn_cell,
                embedded_x,
                dtype=tf.float32,
            )
    return output
