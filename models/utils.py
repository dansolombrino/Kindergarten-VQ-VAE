from rich import print

from torch.nn import Module

from torch import Tensor

from rich.console import Console

def n_trainable_params(model: Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def n_not_trainable_params(model: Module):
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)

def n_params(model: Module):
    return sum(p.numel() for p in model.parameters())

def print_module_params_summary(
    module: Module, module_name: str, 
    color_train: str, color_frozen: str, color_tot: str
):

    print(f"{module_name} params summary:")
    print(f"Trainable params: [bold {color_train}]{n_trainable_params(module):9d} {(n_trainable_params(module) / n_params(module) * 100):06.2f}%[/bold {color_train}]")
    print(f"   Frozen params: [bold {color_frozen}]{n_not_trainable_params(module):9d} {(n_not_trainable_params(module) / n_params(module) * 100):06.2f}%[/bold {color_frozen}]")
    print(f"      Tot params: [bold {color_tot}]{n_params(module):9d}[/bold {color_tot}]")
    print()

def count_pct_padding_tokens(input_ids: Tensor, console: Console):

    mask = input_ids == 0
    # console.print(mask)
    # console.print(mask.shape)
    num_pad_tokens = mask.sum(dim=-1)
    # console.print(num_pad_tokens)
    # console.print(num_pad_tokens.shape)

    pct_pad_tokens = num_pad_tokens / mask.shape[-1] * 100
    # console.print(pct_pad_tokens)
    # console.print(pct_pad_tokens.shape)

    mean_pct_pad_tokens = pct_pad_tokens.mean()
    # console.print(mean_pct_pad_tokens)
    # console.print(mean_pct_pad_tokens.shape)

    return mean_pct_pad_tokens.item()

