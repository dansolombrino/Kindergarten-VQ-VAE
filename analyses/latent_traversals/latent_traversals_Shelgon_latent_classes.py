### --- Imports --- ###

from rich import print

import pandas as pd

import json

import torch

from models.shelgon.Shelgon import Shelgon

from transformers import BertTokenizerFast, GPT2Tokenizer
from transformers.utils import logging; logging.set_verbosity(40)

from torch import softmax, argmax
from torch.nn.functional import gumbel_softmax

### --- Imports --- ###

################################################################################

### --- Definitions --- ###

MODEL_NAME = "Shelgon"

RUN_ID = "2024_02_14_10_57_51 - WandB run 14"

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

all_gen_fact_df = sentences_df[
    (sentences_df[GENERATIVE_FACTOR] == "negative")
]

### --- Select generative factor --- ###

################################################################################

### --- Extract samples w/ desired generative factor --- ###

SENTENCES_TOT = len(all_gen_fact_df)
print(f"SENTENCES_TOT: {SENTENCES_TOT}")

# OVERRIDE_TOT = SENTENCES_TOT
OVERRIDE_TOT = 1

all_gen_fact_df = all_gen_fact_df.head(OVERRIDE_TOT)

all_gen_fact = all_gen_fact_df["input_sentence"].tolist()

### --- Extract samples w/ desired generative factor --- ###

################################################################################

### --- Encode samples --- ###

tokenized_encoder = tokenizer_encoder(
    all_gen_fact, 
    return_tensors="pt",
    add_special_tokens=run_conf["tokenizer_add_special_tokens"], 
    padding="max_length",
    max_length=run_conf["tokenized_sentence_max_length"]
)

input_ids_encoder = tokenized_encoder.input_ids.to(device)
attention_mask_encoder = tokenized_encoder.attention_mask.to(device)

if "use_mask_encoder" in run_conf.keys():
    attention_mask_encoder = attention_mask_encoder if run_conf["use_mask_encoder"] else None

enc_out_all_gen_fact = model.encoder(
    input_ids_encoder, attention_mask=attention_mask_encoder
).last_hidden_state

pred_latent_logits = model.proj_in(enc_out_all_gen_fact)
pred_latent_classes = gumbel_softmax(pred_latent_logits, dim=-1)

### --- Encode samples --- ###

################################################################################

### --- Change latent conditioning --- ###

print(pred_latent_classes)
print(pred_latent_classes.shape)
print()

pred_latent_classes_perm = torch.as_tensor(
    [
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
    ]
).float().to(device).unsqueeze(0)
print(pred_latent_classes_perm.shape)

### --- Change latent conditioning --- ###

################################################################################

### --- Decode test samples --- ###

decoder_input_conditioning = model.proj_out(pred_latent_classes_perm)

recon_logits_travs = model.decoder(
    encoder_hidden_states=decoder_input_conditioning, 
    # we can use the same input ids as the encoder
    input_ids=input_ids_encoder, attention_mask=attention_mask_encoder
).logits

### --- Decode test samples --- ###

################################################################################

### --- Compare original vs. decoded test samples --- ###

recon_ids = argmax(softmax(recon_logits_travs, dim=-1), dim=-1)

recon_sentences = tokenizer_decoder.batch_decode(recon_ids, skip_special_tokens=True)

for original, modified in zip(all_gen_fact, recon_sentences):
    print(f"{original}\n{modified}")
    print("\n ~~~ \n")

# recon_sentences = tokenizer_decoder.batch_decode(recon_ids)

# for original, modified in zip(s_neg, recon_sentences):
#     print(f"{original} --> {modified}")

### --- Compare original vs. decoded test samples --- ###

################################################################################
