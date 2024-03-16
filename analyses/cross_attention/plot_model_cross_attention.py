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

RUN_ID = "2024_02_14_10_57_51 - WandB run 14"

RUN_DIR = f"./runs/{MODEL_NAME}/{RUN_ID}"

ATTNS_DIR = f"{RUN_DIR}/attentions_visualizations"
CROSS_ATTNS_DIR = f"{ATTNS_DIR}/cross_attentions"
SELF_ATTNS_DIR = f"{ATTNS_DIR}/self_attentions"
os.makedirs(CROSS_ATTNS_DIR) if not os.path.exists(CROSS_ATTNS_DIR) else None
os.makedirs(SELF_ATTNS_DIR) if not os.path.exists(SELF_ATTNS_DIR) else None

cross_attentions = torch.load(f"{RUN_DIR}/cross_attentions_mean_across_batch_size.pth")
self_attentions = torch.load(f"{RUN_DIR}/attentions_mean_across_batch_size.pth")
# cross_attns.shape: [num_decoder_layers, num_heads, seq_len, seq_len]

cross_attn: torch.Tensor = cross_attentions
cross_attn: torch.Tensor = cross_attn.mean(dim=0).mean(dim=0)

cross_attn_plot = seaborn.heatmap(data=cross_attn, vmin=cross_attn.min(), vmax=cross_attn.max(), cmap="viridis")
cross_attn_plot.set_title(f"Cross attention, avg across {cross_attentions.shape[0]} layers and {cross_attentions.shape[1]} heads")
cross_attn_plot.set_xlabel("Decoder input tokens")
cross_attn_plot.set_ylabel("Encoder contidioning tokens")
fig = cross_attn_plot.get_figure()
fig.savefig(f"{CROSS_ATTNS_DIR}/layer_AVG_attn_head_AVG.png", dpi=400) 
fig.clear()

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

    cross_attn: torch.Tensor = cross_attentions[layer_idx, ...]
    cross_attn: torch.Tensor = cross_attn.mean(dim=0)
    
    cross_attn_plot = seaborn.heatmap(data=cross_attn, vmin=cross_attn.min(), vmax=cross_attn.max(), cmap="viridis")
    cross_attn_plot.set_title(f"Cross attention for decoder layer {layer_idx}, avg across {cross_attentions.shape[1]} heads")
    cross_attn_plot.set_xlabel("Decoder input tokens")
    cross_attn_plot.set_ylabel("Encoder contidioning tokens")
    fig = cross_attn_plot.get_figure()
    fig.savefig(f"{CROSS_ATTNS_DIR}/layer_{layer_idx}_attn_head_AVG.png", dpi=400) 
    fig.clear()

    for head_idx in range(cross_attentions.shape[1]):
        cross_attn: Tensor = cross_attentions[layer_idx, head_idx, ...]

        cross_attn_plot = seaborn.heatmap(data=cross_attn, vmin=cross_attn.min(), vmax=cross_attn.max(), cmap="viridis")
        cross_attn_plot.set_title(f"Cross attention for decoder layer {layer_idx}, head {head_idx}")
        cross_attn_plot.set_xlabel("Decoder input tokens")
        cross_attn_plot.set_ylabel("Encoder contidioning tokens")
        fig = cross_attn_plot.get_figure()
        fig.savefig(f"{CROSS_ATTNS_DIR}/layer_{layer_idx}_attn_head_{head_idx}.png", dpi=400) 
        fig.clear()
        
        self_attn: Tensor = self_attentions[layer_idx, head_idx, ...]

        self_attn_plot = seaborn.heatmap(data=self_attn, vmin=self_attn.min(), vmax=self_attn.max(), cmap="viridis")
        fig = self_attn_plot.get_figure()
        fig.savefig(f"{SELF_ATTNS_DIR}/layer_{layer_idx}_attn_head_{head_idx}.png", dpi=400) 
        fig.clear()

        prg.advance(heads_task, 1)
    
    prg.advance(layers_task, 1)


