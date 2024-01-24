from rich import print

from rich.progress import Progress

from rich.console import Console

from torch.cuda import device

from torch.utils.data import DataLoader

from Bagon import Bagon

from transformers import PreTrainedTokenizer

from torch.optim.optimizer import Optimizer

from torch.optim.lr_scheduler import LRScheduler

from torch import Tensor

from torch.nn.functional import one_hot

from torch.nn.functional import cross_entropy
from torch.nn.functional import kl_div

from torchmetrics.classification import MulticlassAccuracy
from metrics import seq_acc

from torch.nn.functional import softmax
from torch.nn.functional import log_softmax

from torch import argmax

from dotmap import DotMap

import numpy as np

from torch import no_grad

from consts import *

from wandb.wandb_run import Run


def step(
    device: device,
    model: Bagon, tokenizer: PreTrainedTokenizer, 
    opt: Optimizer, 
    lr_sched: LRScheduler,
    batch: list, vocab_size: int,
    console: Console

):
    # TODO NOTE remove the hardcoded max length to 14, if another dataset is used
    # TODO NOTE or improve its handling even if dSentences will be the only dataset used
    # tokenized = tokenizer(batch, return_tensors="pt", padding="max_length", max_length=14)
    tokenized = tokenizer(batch, return_tensors="pt", padding=True)
    input_ids: Tensor = tokenized.input_ids.to(device)
    attention_mask: Tensor = tokenized.attention_mask.to(device)

    logits_recon: Tensor = model.forward(input_ids, attention_mask)

    # input and targets reshaped to use cross-entropy with sequential data, 
    # as per https://github.com/florianmai/emb2emb/blob/master/autoencoders/autoencoder.py#L116C13-L116C58
    loss_recon_step = cross_entropy(
        input=logits_recon.reshape(-1, vocab_size), target=input_ids.reshape(-1), ignore_index=0
        # input=logits_recon.reshape(-1, vocab_size), target=input_ids.reshape(-1)
    )
    # loss_recon_step = kl_div(
    #     input=log_softmax(logits_recon.reshape(-1, vocab_size), dim=-1), 
    #     target=one_hot(input_ids, vocab_size).float().reshape(-1, vocab_size)
    # )
    
    recon_ids = argmax(softmax(logits_recon, dim=-1), dim=-1)

    metric_acc_step = seq_acc(recon_ids, input_ids)

    loss_full_step: Tensor = loss_recon_step

    # passing opt  --> training time
    # passing None --> inference time
    if opt is not None:
        opt.zero_grad()
        loss_full_step.backward()
        opt.step()

        if lr_sched is not None:
            lr_sched.step()

    return DotMap(
        loss_recon_step=loss_recon_step, 
        loss_full_step=loss_full_step, 
        metric_acc_step=metric_acc_step
    )

def end_of_step_stats_update(stats_train_run: DotMap, stats_step: DotMap, n_els_batch: int):
    
    stats_train_run.loss_recon_run += stats_step.loss_recon_step * n_els_batch
    stats_train_run.loss_full_run += stats_step.loss_full_step * n_els_batch
    stats_train_run.metric_acc_run += stats_step.metric_acc_step * n_els_batch * 1e2
    
    return stats_train_run

def end_of_epoch_stats_update(stats_train_run: DotMap, stats_train_best: DotMap, n_els_epoch: int):

    stats_train_run.loss_recon_run /= n_els_epoch
    stats_train_run.loss_full_run /= n_els_epoch
    stats_train_run.metric_acc_run /= n_els_epoch

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
    suffix_str = "\n" if print_new_line else ""

    console.print(
        epoch_str  + \
        f"loss_recon: [bold {stat_color}] {stats_train_run.loss_recon_run:08.6f}[/bold {stat_color}] {stat_emojis[1] if stats_train_best.loss_recon_is_best else '  '} | " + \
        f"acc: [bold {stat_color}]{stats_train_run.metric_acc_run:08.6f}%[/bold {stat_color}] {stat_emojis[2] if stats_train_best.metric_acc_is_best else '  '} | " + \
        suffix_str
    )


def train(
    prg: Progress, console: Console,
    device: device, 
    dl_train: DataLoader, dl_val: DataLoader, n_batches_train: int, n_batches_val: int,
    model: Bagon, tokenizer: PreTrainedTokenizer,
    opt: Optimizer, lr_sched: LRScheduler, 
    n_epochs: int, 
    vocab_size: int,
    wandb_run: Run
):
    
    prg.start()
    epochs_task = prg.add_task(f"[bold {COLOR_EPOCH}] Epochs", total=n_epochs)
    batches_task_train = prg.add_task(f"[bold {COLOR_TRAIN}] Train batches", total=n_batches_train)
    batches_task_val   = prg.add_task(f"[bold {COLOR_VAL}] Val   batches", total=n_batches_val)

    stats_train_best = DotMap(
        loss_recon_best = np.Inf,
        loss_recon_is_best = False,
        loss_full_best = np.Inf,
        loss_full_is_best = False,
        metric_acc_best = 0,
        metric_acc_is_best = False
    )
    stats_val_best = DotMap(
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
            loss_recon_run = 0,
            loss_full_run = 0,
            metric_acc_run = 0
        )
        n_els_epoch = 0
        model.train()

        ### Begin train batches loop ### 

        for batch in list(dl_train)[:n_batches_train]:
            n_els_batch = len(batch)
            n_els_epoch += n_els_batch

            stats_step: DotMap = step(
                device=device,
                model=model, tokenizer=tokenizer, 
                opt=opt, 
                lr_sched=lr_sched,
                batch=batch, vocab_size=vocab_size,
                console=console
            )

            stats_train_run = end_of_step_stats_update(stats_train_run, stats_step, n_els_batch)
            
            prg.advance(batches_task_train, 1)
            prg.advance(epochs_task, (1 / (n_batches_train + n_batches_val)))

        ### End train batches loop ### 
            
        stats_train_run, stats_train_best = end_of_epoch_stats_update(stats_train_run, stats_train_best, n_els_epoch)
        end_of_epoch_print(stats_train_run, stats_train_best, console, epoch, True, COLOR_TRAIN, STATS_EMOJI_TRAIN, False)
        wandb_run.log(
            {
                "epoch": epoch,
                "lr": lr_sched.get_last_lr()[0] if lr_sched is not None else opt.param_groups[0]['lr'],
                "train/loss_recon": stats_train_run.loss_recon_run,
                "train/loss_full": stats_train_run.loss_full_run,
                "train/acc": stats_train_run.metric_acc_run
            }
        )

        ### End training part ### 
        
        ### Beging validating part ### 

        stats_val_run = DotMap(
            loss_recon_run = 0,
            loss_full_run = 0,
            metric_acc_run = 0
        )
        n_els_epoch = 0
        model.eval()    

        ### Begin val batches loop ### 

        for batch in list(dl_val)[:n_batches_val]:

            n_els_batch = len(batch)
            n_els_epoch += n_els_batch

            with no_grad():

                stats_step: DotMap = step(
                    device=device,
                    model=model, tokenizer=tokenizer, 
                    opt=None, 
                    lr_sched=None,
                    batch=batch, vocab_size=vocab_size,
                    console=console
                )
            
            stats_val_run = end_of_step_stats_update(stats_val_run, stats_step, n_els_batch)

            prg.advance(batches_task_val, 1)
            prg.advance(epochs_task, (1 / (n_batches_train + n_batches_val)))

        ### End val batches loop ### 
            
        stats_val_run, stats_val_best = end_of_epoch_stats_update(stats_val_run, stats_val_best, n_els_epoch)
        end_of_epoch_print(stats_val_run, stats_val_best, console, epoch, False, COLOR_VAL, STATS_EMOJI_VAL, epoch != (n_epochs - 1))
        wandb_run.log(
            {
                "epoch": epoch,
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
    epoch: int,
    wandb_run: Run
):
    
    batches_task_test  = prg.add_task(f"[bold {COLOR_TEST}] Test  batches", total=n_batches_test)
    
    ### Beging testing part ### 

    stats_test_run = DotMap(
        loss_recon_run = 0,
        loss_full_run = 0,
        metric_acc_run = 0
    )
    n_els_epoch = 0
    model.eval()    

    ### Begin val batches loop ### 

    for batch in list(dl_test)[:n_batches_test]:

        n_els_batch = len(batch)
        n_els_epoch += n_els_batch

        with no_grad():

            stats_step: DotMap = step(
                device=device,
                model=model, tokenizer=tokenizer, 
                opt=None, 
                lr_sched=None,
                batch=batch, vocab_size=vocab_size,
                console=console
            )
        
        stats_test_run = end_of_step_stats_update(stats_test_run, stats_step, n_els_batch)

        prg.advance(batches_task_test, 1)

    ### End test batches loop ### 
        
    stats_test_run, stats_test_best = end_of_epoch_stats_update(stats_test_run, stats_test_best, n_els_epoch)
    end_of_epoch_print(stats_test_run, stats_test_best, console, epoch, False, COLOR_TEST, STATS_EMOJI_TEST, True)
    wandb_run.log(
        {
            "epoch": epoch,
            "test/loss_recon": stats_test_run.loss_recon_run,
            "test/loss_full": stats_test_run.loss_full_run,
            "test/acc": stats_test_run.metric_acc_run
        }
    )

    ### End testing part ###





        
    
    
    
    
    
    
    
    
    
    
    
    