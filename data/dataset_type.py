import enum

from data import anli
from data import qqp
from data import snli


class DatasetType(enum.Enum):
    SNLI = 0,
    QQP = 1,
    ANLI = 2


DATASETS = {
    DatasetType.QQP.name: qqp.QQPDataset,
    DatasetType.SNLI.name: snli.SNLIDataset,
    DatasetType.ANLI.name: anli.ANLIDataset,
}

def get_dataset(dataset_name):
    return DATASETS[dataset_name]()