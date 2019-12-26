from layers import convolution
from layers import similarity
from models import base_model
from utils import config_helpers


class CNNSiameseNet(base_model.BaseSiameseNet):
    
    def __init__(
            self,
            max_sequence_len,
            vocabulary_size,
            loss_function,
            embedding_size,
            learning_rate,
            model_cfg,
    ):
        super().__init__(
            max_sequence_len,
            vocabulary_size,
            loss_function,
            embedding_size,
            learning_rate,
            model_cfg,
        )
    
    def siamese_layer(
            self,
            sequence_len,
            model_cfg,
    ):
        num_filters = config_helpers.parse_list(
            model_cfg['PARAMS']['num_filters'],
        )
        filter_sizes = config_helpers.parse_list(
            model_cfg['PARAMS']['filter_sizes'],
        )
        dropout_rate = float(model_cfg['PARAMS']['dropout_rate'])
        
        out1 = convolution.cnn_layers(
            inputs=self.embedded_x1,
            num_filters=num_filters,
            filter_sizes=filter_sizes,
            is_training=self.is_training,
            dropout_rate=dropout_rate,
        )
        
        out2 = convolution.cnn_layers(
            inputs=self.embedded_x2,
            num_filters=num_filters,
            filter_sizes=filter_sizes,
            is_training=self.is_training,
            dropout_rate=dropout_rate,
            reuse=True,
        )
        
        return similarity.manhattan_similarity(out1, out2)
