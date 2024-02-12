from rich import print

import json

from datasets.dSentences.dSentencesDataset import dSentencesDataset

from torch.utils.data.dataloader import DataLoader

from transformers.utils import logging; logging.set_verbosity(40)

import torch

from models.shelgon.Shelgon import Shelgon

from transformers import BertTokenizerFast

from tqdm import tqdm

from torch import Tensor

from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

MODEL_NAME = "Shelgon"

RUN_ID = "2024_02_12_09_29_33 - WandB run 4"

RUN_DIR = f"./runs/{MODEL_NAME}/{RUN_ID}"

with open(f"{RUN_DIR}/run_conf.json", "r") as file:
    run_conf = json.load(file)

ds = dSentencesDataset(run_conf["dataset_path"], run_conf["latent_classes_labels_path"], run_conf["latent_classes_one_hot_path"])

BATCH_SIZE = 2048

dl = DataLoader(
    ds, batch_size=BATCH_SIZE, num_workers=run_conf["num_workers"], 
    pin_memory=run_conf["pin_memory"]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Shelgon(
    encoder_model_name=run_conf["encoder_model_name"], 
    emb_size=run_conf["emb_size"], seq_len=run_conf["tokenized_sentence_max_length"],
    num_latent_classes=run_conf["num_latent_classes"], num_labels_per_class=run_conf["num_labels_per_class"],
    decoder_model_name=run_conf["decoder_model_name"]
).to(device)
model.compile()
model.eval()
model.load_state_dict(state_dict=torch.load(f"{RUN_DIR}/{MODEL_NAME}_ckpt_loss_recon_val_best.pth")["model_state_dict"])
torch.set_grad_enabled(False)

tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(run_conf["tokenizer_name_encoder"])

cross_attns = []
attns = []

# had to limit to 69 batches in order to avoid memory crashes :)
for batch in tqdm(list(dl)[:69], colour="cyan"):
    
    sentences = batch["sentence"]

    tokenized = tokenizer(
        sentences, return_tensors="pt", 
        padding="max_length", max_length=run_conf["tokenized_sentence_max_length"], 
        add_special_tokens=run_conf["tokenizer_add_special_tokens"]
    )

    input_ids: Tensor = tokenized.input_ids.to(device)
    attention_mask: Tensor = tokenized.attention_mask.to(device)

    encoder_output = model.encoder(input_ids, attention_mask=attention_mask).last_hidden_state

    pred_latent_classes = model.proj_in(encoder_output)

    decoder_input_conditioning = model.proj_out(pred_latent_classes)

    decoder_output: CausalLMOutputWithCrossAttentions = model.decoder(
        encoder_hidden_states=decoder_input_conditioning, 
        input_ids=input_ids, attention_mask=attention_mask,
        output_attentions=True
    )

    cross_attns.append(torch.stack(decoder_output.cross_attentions, dim=0).cpu())
    attns.append(torch.stack(decoder_output.attentions, dim=0).cpu())

cross_attns = torch.stack(cross_attns, dim=0)
attns = torch.stack(attns, dim=0)
# {cross_attns,attns}.shape: [num_batches, num_decoder_layers, batch_size, num_heads, seq_len, seq_len]

# gotta avg across batches, otherwise Tensor gets too big :D

cross_attns = cross_attns.mean(dim=0)
attns = attns.mean(dim=0)
# {cross_attns,attns}.shape: [num_decoder_layers, batch_size, num_heads, seq_len, seq_len]

torch.save(cross_attns, f"{RUN_DIR}/cross_attentions_mean_across_num_batches.pth")
torch.save(cross_attns, f"{RUN_DIR}/attentions_mean_across_num_batches.pth")

# gotta avg across batch size too, otherwise Tensor gets too big :D

cross_attns = cross_attns.mean(dim=1)
attns = attns.mean(dim=1)
# {cross_attns,attns}.shape: [num_decoder_layers, num_heads, seq_len, seq_len]

torch.save(cross_attns, f"{RUN_DIR}/cross_attentions_mean_across_batch_size.pth")
torch.save(cross_attns, f"{RUN_DIR}/attentions_mean_across_batch_size.pth")


