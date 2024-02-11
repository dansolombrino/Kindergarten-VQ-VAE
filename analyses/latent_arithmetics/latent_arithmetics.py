from rich import print

import pandas as pd

import json

import torch

from models.bagon.Bagon import Bagon

from transformers import BertTokenizerFast, GPT2Tokenizer
from transformers.utils import logging; logging.set_verbosity(40)

from torch import softmax, argmax

MODEL_NAME = "Bagon"

RUN_ID = "2024_02_11_16_31_10"

RUN_DIR = f"./runs/{MODEL_NAME}/{RUN_ID}"

DECODED_SENTENCES_DF_PATH = f"{RUN_DIR}/decoded_sentences_max_acc_only.feather"

sentences_df = pd.read_feather(DECODED_SENTENCES_DF_PATH)

GENERATIVE_FACTOR = "verb_tense"

all_neg_df = sentences_df[
    (sentences_df[GENERATIVE_FACTOR] == "past")
]

all_aff_df = sentences_df[
    (sentences_df[GENERATIVE_FACTOR] == "present")
]

N_SENTENCES = 300

s_neg_df = all_neg_df.head(N_SENTENCES)
s_aff_df = all_aff_df.head(N_SENTENCES)

s_neg = s_neg_df["input_sentence"].tolist()
s_aff = s_aff_df["input_sentence"].tolist()

with open(f"{RUN_DIR}/run_conf.json", "r") as file:
    run_conf = json.load(file)

torch.set_grad_enabled(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Bagon(
    encoder_model_name=run_conf["encoder_model_name"], 
    decoder_model_name=run_conf["decoder_model_name"],
).to(device)
model.compile()
model.eval()
model.load_state_dict(state_dict=torch.load(f"{RUN_DIR}/{MODEL_NAME.lower()}_ckpt_loss_recon_val_best.pth")["model_state_dict"])

tokenizer_encoder: BertTokenizerFast = BertTokenizerFast.from_pretrained(run_conf["tokenizer_name_encoder"])

if "bert" in run_conf["tokenizer_name_encoder"]:
    tokenizer_decoder: BertTokenizerFast = BertTokenizerFast.from_pretrained(run_conf["tokenizer_name_encoder"])
elif "gpt" in run_conf["tokenizer_name_decoder"]:
    tokenizer_decoder: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(run_conf["tokenizer_name_decoder"])
    tokenizer_decoder.pad_token = tokenizer_decoder.eos_token

tokenized_encoder = tokenizer_encoder(
    s_aff, 
    return_tensors="pt",
    add_special_tokens=run_conf["tokenizer_add_special_tokens"], 
    padding="max_length",
    max_length=run_conf["tokenized_sentence_max_length"]
)

input_ids_encoder = tokenized_encoder.input_ids.to(device)
attention_mask_encoder = tokenized_encoder.attention_mask.to(device)

enc_out_aff = model.encoder(
    input_ids_encoder, attention_mask=attention_mask_encoder
).last_hidden_state

tokenized_encoder = tokenizer_encoder(
    s_neg, 
    return_tensors="pt",
    add_special_tokens=run_conf["tokenizer_add_special_tokens"], 
    padding="max_length",
    max_length=run_conf["tokenized_sentence_max_length"]
)

input_ids_encoder = tokenized_encoder.input_ids.to(device)
attention_mask_encoder = tokenized_encoder.attention_mask.to(device)

enc_out_neg = model.encoder(
    input_ids_encoder, attention_mask=attention_mask_encoder
).last_hidden_state

v = enc_out_neg - enc_out_aff

s_neg_df = all_neg_df.tail(N_SENTENCES)

s_neg = s_neg_df["input_sentence"].tolist()
print(s_neg)

tokenized_encoder = tokenizer_encoder(
    s_neg, 
    return_tensors="pt",
    add_special_tokens=run_conf["tokenizer_add_special_tokens"], 
    padding="max_length",
    max_length=run_conf["tokenized_sentence_max_length"]
)

input_ids_encoder = tokenized_encoder.input_ids.to(device)
attention_mask_encoder = tokenized_encoder.attention_mask.to(device)

enc_out_neg_arith = model.encoder(
    input_ids_encoder, attention_mask=attention_mask_encoder
).last_hidden_state

# v_arith = enc_out_neg_arith
v_arith = v + enc_out_neg_arith
# v_arith = torch.rand_like(v_arith)

tokenized_decoder = tokenizer_decoder(
    s_neg, 
    return_tensors="pt",
    add_special_tokens=run_conf["tokenizer_add_special_tokens"], 
    padding="max_length",
    max_length=run_conf["tokenized_sentence_max_length"]
)
input_ids_decoder = tokenized_decoder.input_ids.to(device)
attention_mask_decoder = tokenized_decoder.attention_mask.to(device)

recon_logits_airth = model.decoder(
    encoder_hidden_states=v_arith, 
    input_ids=input_ids_decoder, attention_mask=attention_mask_decoder
).logits

recon_ids = argmax(softmax(recon_logits_airth, dim=-1), dim=-1)

recon_sentences = tokenizer_decoder.batch_decode(recon_ids)

print(recon_sentences)

