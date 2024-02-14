from rich import print

import os

import torch

from torch import Tensor

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn,TimeElapsedColumn, TimeRemainingColumn
from rich.style import Style
from rich.console import Console

import seaborn

import matplotlib.pyplot as plt


MODEL_NAME = "Shelgon"

RUN_ID = "2024_02_12_09_29_33 - WandB run 4"

RUN_DIR = f"./runs/{MODEL_NAME}/{RUN_ID}"

ATTNS_DIR = f"{RUN_DIR}/attentions_visualizations"
CROSS_ATTNS_DIR = f"{ATTNS_DIR}/cross_attentions"
SELF_ATTNS_DIR = f"{ATTNS_DIR}/self_attentions"
os.makedirs(CROSS_ATTNS_DIR) if not os.path.exists(CROSS_ATTNS_DIR) else None
os.makedirs(SELF_ATTNS_DIR) if not os.path.exists(SELF_ATTNS_DIR) else None

cross_attentions = torch.load(f"{RUN_DIR}/cross_attentions_mean_across_batch_size.pth")
self_attentions = torch.load(f"{RUN_DIR}/attentions_mean_across_batch_size.pth")
# cross_attns.shape: [num_decoder_layers, num_heads, seq_len, seq_len]

console = Console()
prg = Progress(
    SpinnerColumn(spinner_name="hearts"),
    TextColumn("[progress.description]{task.description}"),
    TextColumn("[bold][progress.percentage]{task.percentage:>3.2f}%"),
    BarColumn(finished_style=Style(color="#008000")),
    MofNCompleteColumn(),
    TextColumn("[bold]•"),
    TimeElapsedColumn(),
    TextColumn("[bold]•"),
    TimeRemainingColumn(),
    TextColumn("[bold #5B4328]{task.speed} it/s"),
    SpinnerColumn(spinner_name="hearts"),
    console=console
)
layers_task = prg.add_task(description=f"[bold green] Layers", total=cross_attentions.shape[0])
heads_task = prg.add_task(description=f"[bold cyan] Heads" , total=cross_attentions.shape[1])
prg.start()

for layer_idx in range(cross_attentions.shape[0]):

    prg.reset(heads_task)

    for head_idx in range(cross_attentions.shape[1]):
        cross_attn: Tensor = cross_attentions[layer_idx, head_idx, ...]

        cross_attn_plot = seaborn.heatmap(data=cross_attn, vmin=0, vmax=1, cmap="mako")
        fig = cross_attn_plot.get_figure()
        fig.savefig(f"{CROSS_ATTNS_DIR}/layer_{layer_idx}_attn_head_{head_idx}.png", dpi=400) 
        fig.clear()
        
        self_attn: Tensor = self_attentions[layer_idx, head_idx, ...]

        self_attn_plot = seaborn.heatmap(data=self_attn, vmin=0, vmax=1, cmap="mako")
        fig = self_attn_plot.get_figure()
        fig.savefig(f"{SELF_ATTNS_DIR}/layer_{layer_idx}_attn_head_{head_idx}.png", dpi=400) 
        fig.clear()

        prg.advance(heads_task, 1)
    
    prg.advance(layers_task, 1)


