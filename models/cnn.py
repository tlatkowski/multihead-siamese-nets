from layers.convolution import cnn_layers
from layers.losses import mse
from layers.similarity import manhattan_similarity
from models.base_model import BaseSiameseNet
from utils.config_helpers import parse_list


class CnnSiameseNet(BaseSiameseNet):

    def __init__(self, max_sequence_len, vocabulary_size, main_cfg, model_cfg):
        BaseSiameseNet.__init__(self, max_sequence_len, vocabulary_size, main_cfg, model_cfg, mse)

    def siamese_layer(self, sequence_len, model_cfg):
        num_filters = parse_list(model_cfg['PARAMS']['num_filters'])
        filter_sizes = parse_list(model_cfg['PARAMS']['filter_sizes'])
        dropout_rate = float(model_cfg['PARAMS']['dropout_rate'])

        out1 = cnn_layers(self.embedded_x1,
                          sequence_len,
                          num_filters=num_filters,
                          filter_sizes=filter_sizes,
                          is_training=self.is_training,
                          dropout_rate=dropout_rate)

        out2 = cnn_layers(self.embedded_x2,
                          sequence_len,
                          num_filters=num_filters,
                          filter_sizes=filter_sizes,
                          is_training=self.is_training,
                          dropout_rate=dropout_rate,
                          reuse=True)

        return manhattan_similarity(out1, out2)
