from rich import print

import pandas as pd

import json

import torch

from models.bagon.Bagon import Bagon

from transformers import BertTokenizerFast
from transformers.utils import logging; logging.set_verbosity(40)

MODEL_NAME = "Bagon"

RUN_ID = "2024_02_11_16_31_10"

RUN_DIR = f"./runs/{MODEL_NAME}/{RUN_ID}"

DECODED_SENTENCES_DF_PATH = f"{RUN_DIR}/decoded_sentences.feather"

decoded_sentences = pd.read_feather(DECODED_SENTENCES_DF_PATH)

max_acc_decoded_sentences = decoded_sentences[
    (decoded_sentences["sentence_acc"] > 0.999)
]
max_acc_decoded_sentences.sort_values(by="input_sentence", inplace=True, ascending=True)
max_acc_decoded_sentences = max_acc_decoded_sentences.reset_index()
max_acc_decoded_sentences.to_markdown(
    DECODED_SENTENCES_DF_PATH.replace("decoded_sentences.feather", "decoded_sentences_max_acc_only.md"),
    index=False
)
max_acc_decoded_sentences.to_feather(DECODED_SENTENCES_DF_PATH.replace("decoded_sentences.feather", "decoded_sentences_max_acc_only.feather"))