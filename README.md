[![Build Status](https://travis-ci.org/tlatkowski/multihead-siamese-nets.svg?branch=master)](https://travis-ci.org/tlatkowski/multihead-siamese-nets)

# Siamese Deep Neural Networks for semantic similarity.
This repository contains implementation of Siamese Neural Networks in Tensorflow built based on 3 different and major deep learning architectures:
- Convolutional Neural Networks
- Recurrent Neural Networks
- Multihead Attention Networks

The main reason of creating this repository is to compare well-known implementaions of Siamese Neural Networks available on GitHub mainly built upon CNN and RNN architectures with Siamese Neural Network built based on multihead attention mechanism originally proposed in Transformer model from [Attention is all you need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) paper.

# Installation
This project was developed in and has been tested on Python 3.5. The package requirements are stored in requirements.txt file.

To install the requirements:

```
pip install -r requirements.txt
```
Additionally, you need to have **git-lfs** installed to run model training on predefined corpora.

# Running models
To train model run the following command:

```
python3 run.py train SELECTED_MODEL
```

where SELECTED_MODEL represents one of the selected model among:
- cnn
- rnn
- multihead

Example:
Run the following command to train Siamese Neural Network based on CNN:
```
python3 run.py train cnn
```

# Training configuration
This repository contains main configuration training file placed in 'config/config.ini'.

```ini
[TRAINING]
num_epochs = 20
batch_size = 256
eval_every = 20
learning_rate = 0.001
checkpoints_to_keep = 10
save_every = 200

[DATA]
file_name = corpora/train_snli.txt
num_tests = 1000
logs_path = logs/
model_dir = model_dir/

[PARAMS]
embedding_size = 64
```

# Model configuration
Additionally each model contains its own specific configuration file in which changing hyperparameters is possible.

## Multihead Attention Network configuration file
```ini
[PARAMS]
num_blocks = 2
num_heads = 8
use_residual = False
```
## Convolutional Neural Network configuration file
```ini
[PARAMS]
num_filters = 50,50,50
filter_sizes = 2,3,4
```
## Recurrent Neural Network configuration file
```ini
[PARAMS]
hidden_size = 128
cell_type = GRU
bidirectional = True
```
# Comparison of models
To be done
