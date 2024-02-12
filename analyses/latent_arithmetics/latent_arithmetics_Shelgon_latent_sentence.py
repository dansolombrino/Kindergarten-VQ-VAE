### --- Imports --- ###

from rich import print

import pandas as pd

import json

import torch

from models.shelgon.Shelgon import Shelgon

from transformers import BertTokenizerFast, GPT2Tokenizer
from transformers.utils import logging; logging.set_verbosity(40)

from torch import softmax, argmax

### --- Imports --- ###

################################################################################

### --- Definitions --- ###

MODEL_NAME = "Shelgon"

RUN_ID = "2024_02_12_09_29_33 - WandB run 4"

RUN_DIR = f"./runs/{MODEL_NAME}/{RUN_ID}"

DECODED_SENTENCES_DF_PATH = f"{RUN_DIR}/decoded_sentences_max_acc_only.feather"

### --- Definitions --- ###

################################################################################

### --- Loading and/or init required objects --- ###

sentences_df = pd.read_feather(DECODED_SENTENCES_DF_PATH)

with open(f"{RUN_DIR}/run_conf.json", "r") as file:
    run_conf = json.load(file)

torch.set_grad_enabled(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Shelgon(
    encoder_model_name=run_conf["encoder_model_name"], 
    emb_size=run_conf["emb_size"], seq_len=run_conf["tokenized_sentence_max_length"],
    num_latent_classes=run_conf["num_latent_classes"], num_labels_per_class=run_conf["num_labels_per_class"],
    decoder_model_name=run_conf["decoder_model_name"],
).to(device)
model.compile()
model.eval()
model.load_state_dict(state_dict=torch.load(f"{RUN_DIR}/{MODEL_NAME}_ckpt_loss_recon_val_best.pth")["model_state_dict"])

tokenizer_encoder: BertTokenizerFast = BertTokenizerFast.from_pretrained(run_conf["tokenizer_name_encoder"])

if "bert" in run_conf["tokenizer_name_encoder"]:
    tokenizer_decoder: BertTokenizerFast = BertTokenizerFast.from_pretrained(run_conf["tokenizer_name_encoder"])
elif "gpt" in run_conf["tokenizer_name_decoder"]:
    tokenizer_decoder: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(run_conf["tokenizer_name_decoder"])
    tokenizer_decoder.pad_token = tokenizer_decoder.eos_token

### --- Loading and/or init required objects --- ###

################################################################################

### --- Select generative factor --- ###

GENERATIVE_FACTOR = "sentence_negation"

all_neg_df = sentences_df[
    (sentences_df[GENERATIVE_FACTOR] == "negative")
]

all_aff_df = sentences_df[
    (sentences_df[GENERATIVE_FACTOR] == "affirmative")
]

### --- Select generative factor --- ###

################################################################################

### --- Extract positive and negatives training samples --- ###

SENTENCES_TOT = min(len(all_neg_df), len(all_aff_df))
print(f"SENTENCES_TOT: {SENTENCES_TOT}")

SENTENCES_TRAIN = int(SENTENCES_TOT / 2)

OVERRIDE_TRAIN = 33
# OVERRIDE_TRAIN = SENTENCES_TRAIN

s_neg_train = all_neg_df.head(OVERRIDE_TRAIN)
s_aff_train = all_aff_df.head(OVERRIDE_TRAIN)

s_neg_train = s_neg_train["input_sentence"].tolist()
s_aff_train = s_aff_train["input_sentence"].tolist()

### --- Extract positive and negatives training samples --- ###

################################################################################

### --- Encode train samples AND compute latent difference --- ###

tokenized_encoder = tokenizer_encoder(
    s_aff_train, 
    return_tensors="pt",
    add_special_tokens=run_conf["tokenizer_add_special_tokens"], 
    padding="max_length",
    max_length=run_conf["tokenized_sentence_max_length"]
)

input_ids_encoder = tokenized_encoder.input_ids.to(device)
attention_mask_encoder = tokenized_encoder.attention_mask.to(device)

enc_out_aff_train = model.encoder(
    input_ids_encoder, attention_mask=attention_mask_encoder
).last_hidden_state

tokenized_encoder = tokenizer_encoder(
    s_neg_train, 
    return_tensors="pt",
    add_special_tokens=run_conf["tokenizer_add_special_tokens"], 
    padding="max_length",
    max_length=run_conf["tokenized_sentence_max_length"]
)

input_ids_encoder = tokenized_encoder.input_ids.to(device)
attention_mask_encoder = tokenized_encoder.attention_mask.to(device)

enc_out_neg_train = model.encoder(
    input_ids_encoder, attention_mask=attention_mask_encoder
).last_hidden_state

enc_out_diff_train = enc_out_neg_train - enc_out_aff_train

### --- Encode train samples AND compute latent difference --- ###

################################################################################

### --- Encode test samples AND compute latent addition --- ###

SENTENCES_TEST  = SENTENCES_TOT - SENTENCES_TRAIN

OVERRIDE_TEST = 33
# OVERRIDE_TEST = SENTENCES_TEST

s_neg_test = all_neg_df.tail(OVERRIDE_TEST)

s_neg_test = s_neg_test["input_sentence"].tolist()

tokenized_encoder = tokenizer_encoder(
    s_neg_test, 
    return_tensors="pt",
    add_special_tokens=run_conf["tokenizer_add_special_tokens"], 
    padding="max_length",
    max_length=run_conf["tokenized_sentence_max_length"]
)

input_ids_encoder = tokenized_encoder.input_ids.to(device)
attention_mask_encoder = tokenized_encoder.attention_mask.to(device)

enc_out_neg_test = model.encoder(
    input_ids_encoder, attention_mask=attention_mask_encoder
).last_hidden_state

enc_out_add_test = enc_out_diff_train + enc_out_neg_test

### --- Encode test samples AND compute latent addition --- ###

################################################################################

### --- Decode test samples --- ###

tokenized_decoder = tokenizer_decoder(
    s_neg_test, 
    return_tensors="pt",
    add_special_tokens=run_conf["tokenizer_add_special_tokens"], 
    padding="max_length",
    max_length=run_conf["tokenized_sentence_max_length"]
)
input_ids_decoder = tokenized_decoder.input_ids.to(device)
attention_mask_decoder = tokenized_decoder.attention_mask.to(device)

pred_latent_classes = model.proj_in(enc_out_add_test)

decoder_input_conditioning = model.proj_out(pred_latent_classes)

recon_logits_airth = model.decoder(
    encoder_hidden_states=decoder_input_conditioning, 
    input_ids=input_ids_decoder, attention_mask=attention_mask_decoder
).logits

### --- Decode test samples --- ###

################################################################################

### --- Compare original vs. decoded test samples --- ###

recon_ids = argmax(softmax(recon_logits_airth, dim=-1), dim=-1)

recon_sentences = tokenizer_decoder.batch_decode(recon_ids, skip_special_tokens=True)

for original, modified in zip(s_neg_test, recon_sentences):
    print(f"{original}\n{modified}")
    print("\n ~~~ \n")

# recon_sentences = tokenizer_decoder.batch_decode(recon_ids)

# for original, modified in zip(s_neg, recon_sentences):
#     print(f"{original} --> {modified}")

### --- Compare original vs. decoded test samples --- ###

################################################################################
