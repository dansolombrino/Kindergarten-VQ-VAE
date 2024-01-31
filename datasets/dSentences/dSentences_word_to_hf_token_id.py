from rich import print

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn,TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
from rich.style import Style

from transformers import BertTokenizer

from torch import Tensor

import pandas as pd

import json

VOCAB_DIR  = "./data/dSentences"
VOCAB_PATH = f"{VOCAB_DIR}/dSentences_vocab.txt"

with open(VOCAB_PATH) as file:
    vocab = [line.rstrip() for line in file]

console = Console()
prg = Progress(
    SpinnerColumn(spinner_name="monkey"),
    TextColumn("[progress.description]{task.description}"), TextColumn("[bold][progress.percentage]{task.percentage:>3.2f}%"),
    BarColumn(finished_style=Style(color="#008000")),
    MofNCompleteColumn(), TextColumn("[bold]•"), TimeElapsedColumn(), TextColumn("[bold]•"), TimeRemainingColumn(), TextColumn("[bold #5B4328]{task.speed} it/s"),
    SpinnerColumn(spinner_name="moon"),
    console=console
)

prg.start()
word_tokenization_task = prg.add_task(description="[bold green] Building vocab", total=len(vocab))

TOKENIZER_NAME = "bert-base-uncased"
tokenizer: BertTokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)

word_to_hf_token_ids_df_list = []
word_to_hf_token_id_dict     = {}
hf_token_id_to_word_dict     = {}

for word in vocab:

    tokenized = tokenizer(word, return_tensors="pt", padding=False, add_special_tokens=False)
    input_ids: list = tokenized.input_ids.flatten().tolist()
    token_ids = "_".join([str(i) for i in input_ids])

    word_to_hf_token_ids_df_list.append(
        {
            "word": word,
            f"token_ids_{TOKENIZER_NAME}": token_ids
        }
    )    
    word_to_hf_token_id_dict[word] = token_ids
    hf_token_id_to_word_dict[token_ids] = word
    
    prg.advance(word_tokenization_task, 1)

word_to_hf_token_ids_df = pd.DataFrame(word_to_hf_token_ids_df_list).to_feather(f"{VOCAB_DIR}/dSentences_word_to_hf_token_ids_df.feather")

with open(f"{VOCAB_DIR}/dSentences_word_to_hf_token_id_dict.json", 'w') as fp:
    json.dump(word_to_hf_token_id_dict, fp)

with open(f"{VOCAB_DIR}/dSentences_hf_token_id_to_word_dict.json", 'w') as fp:
    json.dump(hf_token_id_to_word_dict, fp)