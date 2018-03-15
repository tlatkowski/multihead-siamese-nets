[![Build Status](https://travis-ci.org/tlatkowski/multihead-siamese-nets.svg?style=plastic?branch=master)](https://travis-ci.org/tlatkowski/multihead-siamese-nets)![](https://img.shields.io/badge/Python-3.6-brightgreen.svg) ![](https://img.shields.io/badge/TensorFlow-1.3.0-yellowgreen.svg)

# Siamese Deep Neural Networks for semantic similarity.
This repository contains implementation of Siamese Neural Networks in Tensorflow built based on 3 different and major deep learning architectures:
- Convolutional Neural Networks
- Recurrent Neural Networks
- Multihead Attention Networks

The main reason of creating this repository is to compare well-known implementaions of Siamese Neural Networks available on GitHub mainly built upon CNN and RNN architectures with Siamese Neural Network built based on multihead attention mechanism originally proposed in Transformer model from [Attention is all you need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) paper.

# Installation
This project was developed in and has been tested on Python 3.6. The package requirements are stored in requirements.txt file.

To install the requirements, execute the following command:

```
pip install -r requirements/requirements.txt
```
For GPU usage, execute:
```
pip install -r requirements/requirements-gpu.txt
```
and for CPU usage:
```
pip install -r requirements/requirements-cpu.txt
```

Additionally, you need to have **git-lfs** installed to run model training on predefined corpora.

# Running models
To train model run the following command:

```
python3 run.py train SELECTED_MODEL --gpu GPU_NUMBER
```

where SELECTED_MODEL represents one of the selected model among:
- cnn
- rnn
- multihead

**--gpu** is an optional argument, use it in order to indicate specific GPU on your machine (the default value is '0').

Example (GPU usage):
Run the following command to train Siamese Neural Network based on CNN:
```
python3 run.py train cnn --gpu 1
```

Example (CPU usage):
Run the following command to train Siamese Neural Network based on CNN:
```
python3 run.py train cnn
```
# Training configuration
This repository contains main configuration training file placed in 'config/main.ini'.

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
Models are compared upon SNLI corpora.

Experiment parameters:
```ini
Number of epochs : 10
Batch size : 512
Learning rate : 0.001

Number of train instances : 367373
Number of test instances : 50000
Embedding size : 64
```

Specific hyperparameters of models:

CNN | RNN | Multihead
------------ | ------------- | -------------
num_filters = 50,50,50 | hidden_size = 128 | num_blocks = 2
filter_sizes = 2,3,4 | cell_type = GRU | num_heads = 8
|  | bidirectional = True | use_residual = False

Evaluation results:

Model | Test Accuracy | Train Accuracy | Epoch Time
------------ | ------------ | ------------- | -------------
CNN |  |  |  
RNN |  |  |  
Multihead |  |  | 

Training curve (Accuracy): 
![alt text][results]

[results]: https://github.com/tlatkowski/multihead-siamese-nets/blob/master/pics/results.png "Evaluation results"

