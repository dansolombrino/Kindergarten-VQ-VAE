import numpy as np

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn,TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
from rich.style import Style

SENTENCES_PATH = "./data/dSentences/dSentences_sentences.npy"

sentences = np.load(SENTENCES_PATH)

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
vocab_build_task = prg.add_task(description="[bold green] Building vocab", total=sentences.shape[0])

vocab = set()

for s in sentences:
    s_decoded = s.decode()
    
    words = set(s_decoded.split(" "))
    
    vocab.update(words)
    
    prg.advance(vocab_build_task, 1)

vocab_export_task = prg.add_task(description="[bold magenta] Exporting vocab", total=len(vocab))

with open("./data/dSentences/dSentences_vocab.txt", "w") as f:

    for word in vocab:
        print(f"{word}", file=f)

        prg.advance(vocab_export_task, 1)
