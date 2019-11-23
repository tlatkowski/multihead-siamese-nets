import enum

import tensorflow as tf


class RNNCellType(enum.Enum):
    GRU = 0,
    LSTM = 1


def get_rnn_cell(rnn_cell_type: str):
    if rnn_cell_type == RNNCellType.GRU.name:
        return tf.nn.rnn_cell.GRUCell
    elif rnn_cell_type == RNNCellType.LSTM.name:
        return tf.nn.rnn_cell.BasicLSTMCell
    else:
        raise AttributeError('{} RNN cell type not supported.'.format(rnn_cell_type))


def rnn_layer(
        embedded_x,
        hidden_size,
        bidirectional,
        cell_type,
        reuse=False,
):
    with tf.variable_scope('recurrent', reuse=reuse):
        cell = get_rnn_cell(cell_type)
        
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
