from rich import print

import numpy as np

from transformers import BertTokenizer

from rich.progress import *
from rich.console import Console
from rich.style import Style

import torch

CORPUS_PATH = "./data/dSentences/dSentences_sentences.npy" 

sentences = [sentence.decode() for sentence in np.load(CORPUS_PATH)]

tokenizer_name = "bert-base-uncased"
tokenizer: BertTokenizer = BertTokenizer.from_pretrained(tokenizer_name)

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

    sentences_encoded_lengths.append(tokenizer(s, return_tensors="pt", padding=True).input_ids.shape[1])

    prg.advance(tsk, 1)

print(torch.tensor(sentences_encoded_lengths).max())