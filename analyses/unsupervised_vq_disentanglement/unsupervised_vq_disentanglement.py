from rich import print

import torch

from datasets.dSentences.dSentencesDataset import dSentencesDataset

from torch.utils.data import DataLoader

from transformers import BertTokenizer

from transformers.utils import logging; logging.set_verbosity(40)

from models.shelgon.Shelgon import Shelgon
from models.shelgon.VectorQuantizer import VectorQuantizer
from models.shelgon.GumbelQuantizer import GumbelQuantizer

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn,TimeElapsedColumn, TimeRemainingColumn
from rich.style import Style
from rich.console import Console

from torch import Tensor

import json

import os

from common.consts import *

from torch.utils.data import random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

BATCH_SIZE = 512
NUM_WORKERS = 0
PIN_MEMORY = True
dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True )
dl_val   = DataLoader(ds_val  , batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False)
dl_test  = DataLoader(ds_test , batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False)

TOKENIZER_NAME = "bert-base-uncased"
tokenizer: BertTokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)

ENCODER_MODEL_NAME = "bert-base-uncased"
VQ_N_E = 9
VQ_E_DIM = 768
VQ_BETA = 0.1
VQ_MODE = "GumbelQuantizer"
if VQ_MODE == "VectorQuantizer":

    VQ_CODEBOOK_INIT_VALUES_PATH = None

    if VQ_CODEBOOK_INIT_VALUES_PATH is not None:
        vq_codebook_init_values = torch.load(VQ_CODEBOOK_INIT_VALUES_PATH)["codebook_init_values"]
    else: 
        vq_codebook_init_values = None
    vector_quantizer = VectorQuantizer(
        n_e=VQ_N_E, e_dim=VQ_E_DIM, beta=VQ_BETA, 
        vq_codebook_init_values=vq_codebook_init_values
    )
elif VQ_MODE == "GumbelQuantizer":

    ENC_OUT_SIZE = 768
    VQ_TEMPERATURE = 1
    VQ_KL_DIV_SCALE = 1
    VQ_STRAIGHT_THROUGH = False

    vector_quantizer = GumbelQuantizer(
        enc_out_size=ENC_OUT_SIZE, n_embed=VQ_N_E, embedding_dim=VQ_E_DIM,
        temperature=VQ_TEMPERATURE, kl_div_scale=VQ_KL_DIV_SCALE,
        straight_through=VQ_STRAIGHT_THROUGH
    )
else:
    raise ValueError(f"{VQ_MODE} vector quantizer mode NOT supported. Supported modalities: {', '.join(SUPPORTED_VQ_MODES)}")

DECODER_MODEL_NAME = "bert-base-uncased"
FROM_PRETRAINED_BAGON = None
model = Shelgon(
    encoder_model_name=ENCODER_MODEL_NAME, 
    vector_quantizer=vector_quantizer,
    decoder_model_name=DECODER_MODEL_NAME,
    from_pretrained_bagon=FROM_PRETRAINED_BAGON
).to(device)
RUN_ID = "2024_02_04_12_30_10"
CKPT_PATH = f"./runs/Shelgon/{RUN_ID}/shelgon_ckpt_loss_recon_val_best.pth"
model.load_state_dict(torch.load(CKPT_PATH)["model_state_dict"])
model.compile()
model.eval()
torch.set_grad_enabled(False)

# with open("./data/dSentences/dSentences_hf_token_id_to_word_dict.json") as fp:
#     hf_token_id_to_word_dict = json.load(fp)

WORDS_OF_INTEREST = [
    "i", "you", "he", "she", "it", "we", "they",
    "am", "are", "is", "was", "were", 
    "not",
    "do", "does", "will"
]

words_of_interest_vq_distrib = {
    k: [] for k in WORDS_OF_INTEREST
}
vq_words_distrib = {
    k: [] for k in range(9)
}

console = Console()
prg = Progress(
    SpinnerColumn(spinner_name="monkey"),
    TextColumn("[progress.description]{task.description}"), TextColumn("[bold][progress.percentage]{task.percentage:>3.2f}%"),
    BarColumn(finished_style=Style(color="#008000")),
    MofNCompleteColumn(), TextColumn("[bold]•"), TimeElapsedColumn(), TextColumn("[bold]•"), TimeRemainingColumn(), TextColumn("[bold #5B4328]{task.speed} it/s"),
    SpinnerColumn(spinner_name="moon"),
    console=console
)

# dl = [
#     {
#         "sentence": [
#             "I was taming the tiger",
#             # "I will punish the hypocrisy",
#             "I was appreciating the coat"
#         ]
#     }
# ]

seen_v_is = set()

LIM_BATCHES_PCT = 0.1
LIM_BATCHES_TRAIN_PCT = LIM_BATCHES_PCT
LIM_BATCHES_VAL_PCT   = LIM_BATCHES_PCT
LIM_BATCHES_TEST_PCT  = LIM_BATCHES_PCT
n_batches_train = int(len(dl_train) * LIM_BATCHES_TRAIN_PCT)
n_batches_val   = int(len(dl_val  ) * LIM_BATCHES_VAL_PCT)
n_batches_test  = int(len(dl_test ) * LIM_BATCHES_TEST_PCT)
n_batches_tot = n_batches_train + n_batches_val + n_batches_test

prg.start()
ds_stats_task    = prg.add_task("[bold #C71585]Compiling VQ codebook distributions for dataset", total=n_batches_tot)
batch_stats_task = prg.add_task("[bold #FF4500]Compiling VQ codebook distributions for batch"  , total=BATCH_SIZE)

for dl, n_batches in zip([dl_train, dl_val, dl_test], [n_batches_train, n_batches_val, n_batches_test]):
    for batch in list(dl)[:n_batches]:
        
        sentences = batch["sentence"]

        tokenized = tokenizer(sentences, return_tensors="pt", padding=True, add_special_tokens=False)
        input_ids: Tensor = tokenized.input_ids.to(device)
        attention_mask: Tensor = tokenized.attention_mask.to(device)

        _, _, min_encoding_indices, _ = model.forward(input_ids, attention_mask, device, False)

        prg.reset(batch_stats_task)

        for s, ids, v_i in zip(sentences, input_ids, min_encoding_indices):

            v_i = v_i.flatten().tolist()
            # console.print(f"sentence           : {s}")
            # console.print(f"token ids          : {ids.tolist()}")
            # console.print(f"codebook vector ids: {v_i}")

            s_i = 0

            for word in s.split(" "):

                tokens = tokenizer(word, return_tensors="pt", padding=False, add_special_tokens=False).input_ids.flatten().tolist()

                v_is = []

                for j in range(len(tokens)):
                    v_is.append(v_i[s_i + j])
                    vq_words_distrib[v_is[-1]].append(word) # done on all words
                    seen_v_is.add(v_i[s_i + j])


                s_i += len(tokens)
                
                # console.print(f"{word} --> {tokens} --> {str(v_is)}")

                if word in WORDS_OF_INTEREST:
                    if len(tokens) != 1:
                        print(f"(len(tokens)) = {len(tokens)} for word {word}")
                    
                    if len(v_is) != 1:
                        print(f"(len(v_is)) = {len(v_is)} for word {word}")
                    
                    words_of_interest_vq_distrib[word].append(v_is[0])
                    # vq_words_distrib[v_is[0]].add(word) # done only on words of interest

            # console.print()
            prg.advance(batch_stats_task, 1)
            prg.advance(ds_stats_task, (1/BATCH_SIZE))


RESULTS_DIR = f"./analyses/unsupervised_vq_disentanglement/results/{RUN_ID}"
os.makedirs(RESULTS_DIR) if not os.path.exists(RESULTS_DIR) else None

with open(f"{RESULTS_DIR}/dSentences_vq_vector_populated.txt", 'w') as f:
    f.write(f"the following VQ latent vectors were populated: {str(seen_v_is)}")  # set of numbers & a tuple

words_of_interest_histograms = {}

for word in words_of_interest_vq_distrib.keys():
    vq_indexes = set(words_of_interest_vq_distrib[word])

    histogram = {
        k: 0 for k in range(9)
    }
    for ind in vq_indexes:
        histogram[ind] = words_of_interest_vq_distrib[word].count(ind)

    words_of_interest_histograms[word] = histogram

with open(f"{RESULTS_DIR}/dSentences_words_of_interest_histograms.json", 'w') as fp:
    json.dump(words_of_interest_histograms, fp)

vq_words_distrib = {
    k: list(set(v)) for (k, v) in vq_words_distrib.items()
}
with open(f"{RESULTS_DIR}/dSentences_vq_words_distrib.json", 'w') as fp:
    json.dump(vq_words_distrib, fp)
