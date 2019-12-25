from enum import Enum
from models.cnn import CNNSiameseNet
from models.rnn import LSTMBasedSiameseNet
from models.multihead_attention import MultiheadAttentionSiameseNet


class ModelType(Enum):
    multihead = 0,
    rnn = 1,
    cnn = 2


MODELS = {
    ModelType.cnn.name: CNNSiameseNet,
    ModelType.rnn.name: LSTMBasedSiameseNet,
    ModelType.multihead.name: MultiheadAttentionSiameseNet
}

