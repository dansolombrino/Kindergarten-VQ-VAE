# Kindergarten-VQ-VAE

## Getting started

- Create a Python virtual environment using the preferred environment manager
- Install dependencies listed in `requirements.txt`

## Folder structure

This is the project folder structure, highlighting the goal of each directory

- `Kindergarten-VQ-VAE` $\to$ project root directory
    - `common` $\to$ common utility scripts, classes and constants

    - `datasets` $\to$ dataset files
        - `dSentences` $\to$ dSentences dataset files
    
    - `data` $\to$ data-related files (e.g. pre-processing, PyTorch DataSet implementation)
        - `dSentences` $\to$ dSentences data-related files

    - `models` $\to$ PyTorch models 
        - `bagon` $\to$ Attempt 1 $\to$ Pre-trained BERT Encoder, fine-tuned BERT Decoder LM head
        - `bagon` $\to$ Attempt 2 $\to$ Pre-trained BERT Encoder, fine-tuned BERT Decoder LM head, VQ from scratch

## How to train

### Bagon model

- Activate the virtual environment for this project
- `cd` into project root folder
- Change hyperparameters in `models/bagon/config.py`, if desired
- `PYTHONPATH=. python3 models/bagon/main.py` from the project root folder. Alternatively:
    - `export PYTHONPATH=.` once when you first start the terminal session you want to run the model in
    - `python3 models/bagon/main.py`
