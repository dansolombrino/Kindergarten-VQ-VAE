### Dataset ###

from math import isclose

DATASET_PATH = "./data/dSentences/dSentences_sentences.npy"

TRAIN_SPLIT_PCT = 0.6
VAL_SPLIT_PCT = 0.2
TEST_SPLIT_PCT = 0.2

assert isclose(TRAIN_SPLIT_PCT + VAL_SPLIT_PCT + TEST_SPLIT_PCT, 1), "Train/val/test percentage splits must sum up to 1"

DS_GEN_SEED = 69

### Dataset ###

### DataLoader ###

BATCH_SIZE = 256

NUM_WORKERS = 0

PIN_MEMORY = True

### DataLoader ###

### Model ###

ENCODER_MODEL_NAME = "bert-base-uncased"

VQ_N_E = 9 # dSentences has 9 generative factors!
VQ_E_DIM = 768
VQ_BETA = 0.69

DECODER_MODEL_NAME = "bert-base-uncased"

### Model ###

### Optimizer ###



### Optimizer ###

### Training ###

LIM_BATCHES_TRAIN_PCT = 0.005
LIM_BATCHES_VAL_PCT   = 0.025
LIM_BATCHES_TEST_PCT  = 0.025

N_EPOCHS = 2

LOSS_RECON_RESCALE_FACTOR = 1

### Training ###