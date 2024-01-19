from rich import print

from torch.nn import Module

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