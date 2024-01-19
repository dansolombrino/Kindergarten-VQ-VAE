from rich import print

from rich.progress import Progress

from rich.console import Console

from torch.cuda import device

from torch.utils.data import DataLoader

from Bagon import Bagon

from transformers import PreTrainedTokenizer

from torch.optim.optimizer import Optimizer

from torch import Tensor

from torch.nn.functional import one_hot

from torch.nn.functional import cross_entropy

from torchmetrics.classification import MulticlassAccuracy

from torch.nn.functional import softmax

from torch import argmax

from dotmap import DotMap

import numpy as np

from torch import no_grad

from consts import *

from wandb.wandb_run import Run


def step(
    device: device,
    model: Bagon, tokenizer: PreTrainedTokenizer, opt: Optimizer,
    batch: list, vocab_size: int,
    loss_recon_rescale_factor: float, multi_class_acc: MulticlassAccuracy,
    console: Console

):
    input_ids: Tensor = tokenizer(batch, return_tensors="pt", padding=True).input_ids.to(device)

    loss_vq_step: Tensor
    logits_recon: Tensor
    loss_vq_step, logits_recon = model.forward(input_ids, device)

    input_ids_one_hot = one_hot(input_ids, vocab_size).float()
    loss_recon_step = loss_recon_rescale_factor * cross_entropy(input=logits_recon, target=input_ids_one_hot)
    
    recon_ids = argmax(softmax(logits_recon, dim=-1), dim=-1)
    metric_acc_step = multi_class_acc(recon_ids, input_ids)

    loss_full_step: Tensor = loss_vq_step + loss_recon_step

    # passing opt  --> training time
    # passing None --> inference time
    if opt is not None:
        opt.zero_grad()
        loss_full_step.backward()
        opt.step()

    return DotMap(
        loss_vq_step=loss_vq_step, 
        loss_recon_step=loss_recon_step, 
        loss_full_step=loss_full_step, 
        metric_acc_step=metric_acc_step
    )

def end_of_step_stats_update(stats_train_run: DotMap, stats_step: DotMap):
    
    stats_train_run.loss_vq_run += stats_step.loss_vq_step
    stats_train_run.loss_recon_run += stats_step.loss_recon_step
    stats_train_run.loss_full_run += stats_step.loss_full_step
    stats_train_run.metric_acc_run += stats_step.metric_acc_step
    
    return stats_train_run

def end_of_epoch_stats_update(stats_train_run: DotMap, stats_train_best: DotMap, n_batches_train: int):

    stats_train_run.loss_vq_run /= n_batches_train
    stats_train_run.loss_recon_run /= n_batches_train
    stats_train_run.loss_full_run /= n_batches_train
    stats_train_run.metric_acc_run /= n_batches_train * 1e-2

    stats_train_best.loss_vq_is_best = stats_train_run.loss_vq_run < stats_train_best.loss_vq_best
    stats_train_best.loss_vq_best = stats_train_run.loss_vq_run if stats_train_best.loss_vq_is_best else stats_train_best.loss_vq_best
    stats_train_best.loss_recon_is_best = stats_train_run.loss_recon_run < stats_train_best.loss_recon_best
    stats_train_best.loss_recon_best = stats_train_run.loss_recon_run if stats_train_best.loss_recon_is_best else stats_train_best.loss_recon_best
    stats_train_best.loss_full_is_best = stats_train_run.loss_full_run < stats_train_best.loss_full_best
    stats_train_best.loss_full_best = stats_train_run.loss_full_run if stats_train_best.loss_full_is_best else stats_train_best.loss_full_best
    stats_train_best.metric_acc_is_best = stats_train_run.metric_acc_run > stats_train_best.metric_acc_best
    stats_train_best.metric_acc_best = stats_train_run.metric_acc_run if stats_train_best.metric_acc_is_best else stats_train_best.metric_acc_best

    return stats_train_run, stats_train_best

def end_of_epoch_print(
    stats_train_run: DotMap, stats_train_best: DotMap, 
    console: Console, 
    epoch: int, print_epoch: bool,
    stat_color: str, stat_emojis: list,
    print_new_line: bool
):
    epoch_str = f"[bold {COLOR_EPOCH}]{epoch:03d}[/bold {COLOR_EPOCH}] | " if print_epoch else "    | "
    suffix = "\n" if print_new_line else ""

    console.print(
        epoch_str  + \
        f"loss_vq: [bold {stat_color}] {stats_train_run.loss_vq_run:02.6f}[/bold {stat_color}] {stat_emojis[0] if stats_train_best.loss_vq_is_best else '  '} | " + \
        f"loss_recon: [bold {stat_color}] {stats_train_run.loss_recon_run:02.6f}[/bold {stat_color}] {stat_emojis[1] if stats_train_best.loss_recon_is_best else '  '} | " + \
        f"acc: [bold {stat_color}]{stats_train_run.metric_acc_run:02.6f}%[/bold {stat_color}] {stat_emojis[2] if stats_train_best.metric_acc_is_best else '  '} | " + \
        suffix
    )


def train(
    prg: Progress, console: Console,
    device: device, 
    dl_train: DataLoader, dl_val: DataLoader, n_batches_train: int, n_batches_val: int,
    model: Bagon, tokenizer: PreTrainedTokenizer,
    opt: Optimizer,
    n_epochs: int, 
    vocab_size: int,
    loss_recon_rescale_factor: float,
    wandb_run: Run
):
    
    prg.start()
    epochs_task = prg.add_task(f"[bold {COLOR_EPOCH}] Epochs", total=n_epochs)
    batches_task_train = prg.add_task(f"[bold {COLOR_TRAIN}] Train batches", total=n_batches_train)
    batches_task_val   = prg.add_task(f"[bold {COLOR_VAL}] Val   batches", total=n_batches_val)

    multi_class_acc = MulticlassAccuracy(num_classes=vocab_size).to(device)
    stats_train_best = DotMap(
        loss_vq_best = np.Inf,
        loss_vq_is_best = False,
        loss_recon_best = np.Inf,
        loss_recon_is_best = False,
        loss_full_best = np.Inf,
        loss_full_is_best = False,
        metric_acc_best = 0,
        metric_acc_is_best = False
    )
    stats_val_best = DotMap(
        loss_vq_best = np.Inf,
        loss_recon_best = np.Inf,
        loss_full_best = np.Inf,
        metric_acc_best = 0
    )

    ### Begin epochs loop ###

    for epoch in range(n_epochs):

        prg.reset(batches_task_train)
        prg.reset(batches_task_val)

        ### Begin trainining part ### 
        
        stats_train_run = DotMap(
            loss_vq_run = 0,
            loss_recon_run = 0,
            loss_full_run = 0,
            metric_acc_run = 0
        )
        model.train()

        ### Begin train batches loop ### 

        for batch in list(dl_train)[:n_batches_train]:

            stats_step: DotMap = step(
                device=device,
                model=model, tokenizer=tokenizer, opt=opt,
                batch=batch, vocab_size=vocab_size,
                loss_recon_rescale_factor=loss_recon_rescale_factor, multi_class_acc=multi_class_acc,
                console=console
            )

            stats_train_run = end_of_step_stats_update(stats_train_run, stats_step)
            
            prg.advance(batches_task_train, 1)
            prg.advance(epochs_task, (1 / (n_batches_train + n_batches_val)))

        ### End train batches loop ### 
            
        stats_train_run, stats_train_best = end_of_epoch_stats_update(stats_train_run, stats_train_best, n_batches_train)
        end_of_epoch_print(stats_train_run, stats_train_best, console, epoch, True, COLOR_TRAIN, STATS_EMOJI_TRAIN, False)
        wandb_run.log(
            {
                "epoch": epoch,
                "train/loss_vq": stats_train_run.loss_vq_run,
                "train/loss_recon": stats_train_run.loss_recon_run,
                "train/loss_full": stats_train_run.loss_full_run,
                "train/acc": stats_train_run.metric_acc_run
            }
        )

        ### End training part ### 
        
        ### Beging validating part ### 

        stats_val_run = DotMap(
            loss_vq_run = 0,
            loss_recon_run = 0,
            loss_full_run = 0,
            metric_acc_run = 0
        )
        model.eval()    

        ### Begin val batches loop ### 

        for batch in list(dl_val)[:n_batches_val]:

            with no_grad():

                stats_step: DotMap = step(
                    device=device,
                    model=model, tokenizer=tokenizer, opt=None,
                    batch=batch, vocab_size=vocab_size,
                    loss_recon_rescale_factor=loss_recon_rescale_factor, multi_class_acc=multi_class_acc,
                    console=console
                )
            
            stats_val_run = end_of_step_stats_update(stats_val_run, stats_step)

            prg.advance(batches_task_val, 1)
            prg.advance(epochs_task, (1 / (n_batches_train + n_batches_val)))

        ### End val batches loop ### 
            
        stats_val_run, stats_val_best = end_of_epoch_stats_update(stats_val_run, stats_val_best, n_batches_val)
        end_of_epoch_print(stats_val_run, stats_val_best, console, epoch, False, COLOR_VAL, STATS_EMOJI_VAL, epoch != (n_epochs - 1))
        wandb_run.log(
            {
                "epoch": epoch,
                "val/loss_vq": stats_val_run.loss_vq_run,
                "val/loss_recon": stats_val_run.loss_recon_run,
                "val/loss_full": stats_val_run.loss_full_run,
                "val/acc": stats_val_run.metric_acc_run
            }
        )

        ### End validating part ### 

    ### End epochs loop ###
        
    return

def test(
    prg: Progress, console: Console,
    device: device, 
    dl_test: DataLoader, n_batches_test,
    model: Bagon, tokenizer: PreTrainedTokenizer,
    vocab_size: int,
    loss_recon_rescale_factor: float,
    epoch: int,
    wandb_run: Run
):
    
    batches_task_test  = prg.add_task(f"[bold {COLOR_TEST}] Test  batches", total=n_batches_test)
    
    multi_class_acc = MulticlassAccuracy(num_classes=vocab_size).to(device)
    stats_test_best = DotMap(
        loss_vq_best = np.Inf,
        loss_recon_best = np.Inf,
        loss_full_best = np.Inf,
        metric_acc_best = 0
    )
    
    ### Beging testing part ### 

    stats_test_run = DotMap(
        loss_vq_run = 0,
        loss_recon_run = 0,
        loss_full_run = 0,
        metric_acc_run = 0
    )
    model.eval()    

    ### Begin val batches loop ### 

    for batch in list(dl_test)[:n_batches_test]:

        with no_grad():

            stats_step: DotMap = step(
                device=device,
                model=model, tokenizer=tokenizer, opt=None,
                batch=batch, vocab_size=vocab_size,
                loss_recon_rescale_factor=loss_recon_rescale_factor, multi_class_acc=multi_class_acc,
                console=console
            )
        
        stats_test_run = end_of_step_stats_update(stats_test_run, stats_step)

        prg.advance(batches_task_test, 1)

    ### End test batches loop ### 
        
    stats_test_run, stats_test_best = end_of_epoch_stats_update(stats_test_run, stats_test_best, n_batches_test)
    end_of_epoch_print(stats_test_run, stats_test_best, console, epoch, False, COLOR_TEST, STATS_EMOJI_TEST, True)
    wandb_run.log(
        {
            "epoch": epoch,
            "test/loss_vq": stats_test_run.loss_vq_run,
            "test/loss_recon": stats_test_run.loss_recon_run,
            "test/loss_full": stats_test_run.loss_full_run,
            "test/acc": stats_test_run.metric_acc_run
        }
    )

    ### End testing part ###





        
    
    
    
    
    
    
    
    
    
    
    
    