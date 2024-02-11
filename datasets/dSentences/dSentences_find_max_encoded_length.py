from rich import print

import numpy as np

from transformers import BertTokenizerFast, GPT2TokenizerFast

from rich.progress import *
from rich.console import Console
from rich.style import Style

import torch

CORPUS_PATH = "./data/dSentences/dSentences_sentences_clean.npy" 

if "_clean.npy" in CORPUS_PATH: 
    sentences = np.load(CORPUS_PATH)
else:
    sentences = [sentence.decode() for sentence in np.load(CORPUS_PATH)]

tokenizer_name = "gpt2"

if "bert" in tokenizer_name:
    tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(tokenizer_name)
elif "gpt" in tokenizer_name:
    tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
else:
    raise ValueError(f"{tokenizer_name} is not a valid tokenizer name")

console = Console()
prg = Progress(
    SpinnerColumn(spinner_name="monkey"),
    TextColumn("[progress.description]{task.description}"),
    TextColumn("[bold][progress.percentage]{task.percentage:>3.2f}%"),
    BarColumn(finished_style=Style(color="#008000")),
    MofNCompleteColumn(),
    TextColumn("[bold]•"),
    TimeElapsedColumn(),
    TextColumn("[bold]•"),
    TimeRemainingColumn(),
    TextColumn("[bold #5B4328]{task.speed} it/s"),
    SpinnerColumn(spinner_name="moon"),
    console=console
)

prg.start()
tsk = prg.add_task("[bold violet] Encoding sentences", total=len(sentences))

sentences_encoded_lengths = []

for s in sentences:

    sentences_encoded_lengths.append(tokenizer(s, return_tensors="pt", padding=True, add_special_tokens=True).input_ids.shape[1])

    prg.advance(tsk, 1)

console.print(torch.tensor(sentences_encoded_lengths).max())