from transformers.utils import logging
logging.set_verbosity(40)

from rich import print

import torch

from models.bagon.Bagon import Bagon

from transformers import BertTokenizer

from torch import Tensor

SUPPORTED_MODEL_NAMES = ["Bagon"]
MODEL_NAME = "Bagon"

loaded_checkpoint = torch.load(f"./runs/{MODEL_NAME}/2024_01_27_11_15_55/bagon_ckpt_loss_recon_train_best.pth")

ENCODER_MODEL_NAME = "bert-base-uncased"
DECODER_MODEL_NAME = "bert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if MODEL_NAME == "Bagon":
    model = Bagon(
        encoder_model_name=ENCODER_MODEL_NAME, 
        decoder_model_name=DECODER_MODEL_NAME
    ).to(device)
    model.load_state_dict(loaded_checkpoint["model_state_dict"])
else:
    raise ValueError(f"Invalid model name \"{MODEL_NAME}\", supported values: {', '.join(SUPPORTED_MODEL_NAMES)}")

TOKENIZER_NAME = "bert-base-uncased"
tokenizer: BertTokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)

batch = [
    "he accepted the payment",
    "are you not ruining the holidays",
    "they were touring the lakes"
]

tokenized = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False)
input_ids: Tensor = tokenized.input_ids.to(device)
attention_mask: Tensor = tokenized.attention_mask.to(device)

logits_recon: Tensor = model.forward(input_ids, attention_mask)