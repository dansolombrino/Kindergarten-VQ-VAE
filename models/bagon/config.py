TO_EXCLUDE = [
    # this list 
    "TO_EXCLUDE"
] 

def exclude_arg(k, v):
    return k.endswith("__") or k.startswith("__") or callable(v) or k in TO_EXCLUDE

def get_config():
    
    config = globals()
    train_conf = {}

    for k, v in config.items():

        if not exclude_arg(k, v): 
            
            train_conf[k.lower()] = v if v is not None else "None"

    return train_conf

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

#            full --> train all model parameters
# dec-cls-head-ft --> fine-tune BERT decoder classification head
MODEL_MODE = "dec-cls-head-ft"

### Model ###

### Optimizer ###



### Optimizer ###

### Weights and Biases ###

WANDB_PROJECT_NAME = "Kindergarten-VQ-VAE"
WANDB_GROUP = "Bagon"
WANDB_JOB_TYPE = "code-testing"

WANDB_MODE = "disabled"
# WANDB_MODE = "online"

WANDB_WATCH_MODEL = True

### Weights and Biases ###

### Training ###

LIM_BATCHES_TRAIN_PCT = 0.01
LIM_BATCHES_VAL_PCT   = 0.01
LIM_BATCHES_TEST_PCT  = 0.01

N_EPOCHS = 2

LOSS_RECON_RESCALE_FACTOR = 1

### Training ###