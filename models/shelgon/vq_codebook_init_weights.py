from datasets.dSentences.dSentencesDataset import dSentencesDataset

import torch

from common.consts import *

from torch.utils.data import random_split, DataLoader

from transformers import BertTokenizer
from transformers.utils import logging
logging.set_verbosity(40)

from models.bagon.Bagon import Bagon

from rich import print

from rich.progress import track

from scipy.cluster.vq import kmeans2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_grad_enabled(False)

SENTENCES_PATH = "./data/dSentences/dSentences_sentences.npy"
LATENT_CLASSES_LABELS_PATH = "./data/dSentences/dSentences_latent_classes_labels.npy"

ds = dSentencesDataset(SENTENCES_PATH, LATENT_CLASSES_LABELS_PATH)

TRAIN_SPLIT_PCT = 0.6
VAL_SPLIT_PCT = 0.2
TEST_SPLIT_PCT = 0.2

ds_train_len = int(len(ds) * TRAIN_SPLIT_PCT)
ds_val_len = int(len(ds) * VAL_SPLIT_PCT)
ds_test_len = len(ds) - ds_train_len - ds_val_len
ds_gen = torch.Generator()
ds_gen.manual_seed(DS_GEN_SEED)
ds_train, ds_val, ds_test = random_split(ds, (ds_train_len, ds_val_len, ds_test_len), ds_gen)

BATCH_SIZE = 2048

print(f"using {len(ds_train)} examples")
dl_train_codebook_init = DataLoader(ds_train, batch_size=BATCH_SIZE, pin_memory=True)

TOKENIZER_NAME = "bert-base-uncased"
tokenizer: BertTokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)

ENCODER_MODEL_NAME = "bert-base-uncased"
DECODER_MODEL_NAME = "bert-base-uncased"
model = Bagon(encoder_model_name=ENCODER_MODEL_NAME, decoder_model_name=DECODER_MODEL_NAME).to(device)
model.load_state_dict(torch.load("./runs/Bagon/2024_01_27_19_11_06/bagon_ckpt_loss_recon_val_best.pth")["model_state_dict"])

recon_logits_list = []

print("tokenizing...")

for batch in track(dl_train_codebook_init, "[bold green]Encoding train dataset"):

    sentences = batch["sentence"]
    
    tokenized = tokenizer(sentences, return_tensors="pt", padding='max_length', max_length=12, add_special_tokens=False)
    input_ids     : torch.Tensor = tokenized.input_ids.to(device)
    attention_mask: torch.Tensor = tokenized.attention_mask.to(device)

    recon_logits = model.encoder.forward(input_ids, attention_mask=attention_mask).last_hidden_state
    recon_logits_list.append(recon_logits.cpu())

print("tokenized!")

recon_logits_tensor = torch.cat(recon_logits_list).cpu()

print(f"recon_logits_tensor.shape: {recon_logits_tensor.shape}")

E_DIM = 768
N_E = 9

# flatten input: (B, S, C) --> (B*S, C)
z_flattened = recon_logits_tensor.view((-1, E_DIM))

print('running kmeans!!') # data driven initialization for the embeddings
rp = torch.randperm(z_flattened.size(0))
print(f"rp.shape: {rp.shape}")
# kd = kmeans2(z_flattened[rp[:20000]].data.cpu().numpy(), N_E, minit='points')
kd = kmeans2(z_flattened[rp[:]].data.cpu().numpy(), N_E, minit='points')

print("kmeans done!!")

codebook_init_values = torch.from_numpy(kd[0])
print(codebook_init_values.shape)

print("exporting values to disk...")
torch.save(
    {
        "codebook_init_values": codebook_init_values,
        "encoder_model_name": ENCODER_MODEL_NAME,
        "decoder_model_name": DECODER_MODEL_NAME,
        "tokenizer_name": TOKENIZER_NAME,
    },
    "./data/dSentences/dSentences_codebook_init_values.pth"
)
print("values exported to disk!")