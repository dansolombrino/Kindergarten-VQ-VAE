# Kindergarten-VQ-VAE

## Getting started

- Create a Python virtual environment using the preferred environment manager
- Install dependencies listed in `requirements.txt`

## Folder structure

This is the project folder structure, highlighting the goal of each directory

- `Kindergarten-VQ-VAE` $\to$ project root directory
    - `datasets` $\to$ dataset files
        - `dSentences` $\to$ dSentences dataset files
    
    - `data` $\to$ data-related files (e.g. pre-processing, PyTorch DataSet implementation)
        - `dSentences` $\to$ dSentences data-related files

    - `models` $\to$ PyTorch models 
        - `bagon` $\to$ Attempt 1 $\to$ Pre-trained BERT Encoder, fine-tuned BERT Decoder LM head, VQ-VAE quantizers trained from scratch
