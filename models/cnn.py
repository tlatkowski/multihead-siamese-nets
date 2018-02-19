from layers.convolution import cnn_layers
from layers.losses import mse
from layers.similarity import manhattan_similarity
from models.base_model import SiameseNet
from utils.config_helpers import parse_list


class CnnSiameseNet(SiameseNet):

    def __init__(self, max_sequence_len, vocabulary_size, main_cfg, model_cfg):
        SiameseNet.__init__(self, max_sequence_len, vocabulary_size, main_cfg, model_cfg, mse)

    def siamese_layer(self, sequence_len, model_cfg):
        num_filters = parse_list(model_cfg['PARAMS']['num_filters'])
        filter_sizes = parse_list(model_cfg['PARAMS']['filter_sizes'])

        out1 = cnn_layers(self.embedded_x1,
                          sequence_len,
                          num_filters=num_filters,
                          filter_sizes=filter_sizes)

        out2 = cnn_layers(self.embedded_x2,
                          sequence_len,
                          num_filters=num_filters,
                          filter_sizes=filter_sizes,
                          reuse=True)

        return manhattan_similarity(out1, out2)
