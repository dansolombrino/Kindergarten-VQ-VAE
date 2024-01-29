from rich import print

from rich.progress import Progress

from rich.console import Console

from torch.cuda import device

from torch.utils.data import DataLoader

from Shelgon import Shelgon

from transformers import PreTrainedTokenizer

from torch.optim.optimizer import Optimizer

from torch.optim.lr_scheduler import LRScheduler

from torch import Tensor

from torch.nn.functional import one_hot

from torch.nn.functional import cross_entropy
from torch.nn.functional import kl_div

from torchmetrics.classification import MulticlassAccuracy
from common.metrics import seq_acc

from torch.nn.functional import softmax
from torch.nn.functional import log_softmax

from torch import argmax

import numpy as np

from torch import no_grad

from common.consts import *

from wandb.wandb_run import Run

from torch import save

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

def step(
    device: device,
    model: Shelgon, tokenizer: PreTrainedTokenizer, tokenizer_add_special_tokens: bool,
    opt: Optimizer, 
    loss_recon_rescale_factor: float, loss_recon_weight: float,
    loss_vq_rescale_factor: float, loss_vq_weight: float,
    lr_sched: LRScheduler,
    batch: list, vocab_size: int,
    console: Console

):
    sentences = batch["sentence"]
    latent_classes_labels = batch["latent_classes_labels"]
    tokenized = tokenizer(sentences, return_tensors="pt", padding=True, add_special_tokens=tokenizer_add_special_tokens)
    input_ids: Tensor = tokenized.input_ids.to(device)
    attention_mask: Tensor = tokenized.attention_mask.to(device)

    loss_vq_step: Tensor; min_encoding_indices: Tensor; logits_recon: Tensor
    loss_vq_step, min_encoding_indices, logits_recon = model.forward(input_ids, attention_mask, device)

    # input and targets reshaped to use cross-entropy with sequential data, 
    # as per https://github.com/florianmai/emb2emb/blob/master/autoencoders/autoencoder.py#L116C13-L116C58
    # loss_recon_step: Tensor = cross_entropy(
    #     input=logits_recon.reshape(-1, vocab_size), target=input_ids.reshape(-1)
    # )
    loss_recon_step: Tensor = kl_div(
        input=log_softmax(logits_recon.reshape(-1, vocab_size), dim=-1), 
        target=one_hot(input_ids, vocab_size).reshape(-1, vocab_size).float(),
        reduction="batchmean"
    )
    
    recon_ids = argmax(softmax(logits_recon, dim=-1), dim=-1)
    metric_acc_step = seq_acc(recon_ids, input_ids)

    loss_recon_step *= loss_recon_rescale_factor * loss_recon_weight
    loss_vq_step *= loss_vq_rescale_factor * loss_vq_weight
    loss_full_step: Tensor = loss_recon_step + loss_vq_step

    # passing opt  --> training time
    # passing None --> inference time
    if opt is not None:
        opt.zero_grad()
        loss_full_step.backward()
        opt.step()

        if lr_sched is not None:
            lr_sched.step()

    return {
        "loss_recon_step": loss_recon_step, 
        "loss_vq_step": loss_vq_step,
        "loss_full_step": loss_full_step, 
        "metric_acc_step": metric_acc_step,
        "padding_tokens_pct_step": -69 #count_pct_padding_tokens(input_ids, console)
    }, input_ids, recon_ids

def end_of_step_stats_update(stats_stage_run: dict, stats_step: dict, n_els_batch: int):
    
    stats_stage_run["loss_recon_run"] += stats_step["loss_recon_step"] * n_els_batch
    stats_stage_run["loss_vq_run"] += stats_step["loss_vq_step"] * n_els_batch
    stats_stage_run["loss_full_run"] += stats_step["loss_full_step"] * n_els_batch
    stats_stage_run["metric_acc_run"] += stats_step["metric_acc_step"] * n_els_batch * 1e2
    stats_stage_run["padding_tokens_pct_run"] += stats_step["padding_tokens_pct_step"]
    
    return stats_stage_run

def end_of_epoch_stats_update(stats_stage_run: dict, stats_stage_best: dict, n_els_epoch: int, n_steps: int):

    stats_stage_run["loss_recon_run"] /= n_els_epoch
    stats_stage_run["loss_vq_run"] /= n_els_epoch
    stats_stage_run["loss_full_run"] /= n_els_epoch
    stats_stage_run["metric_acc_run"] /= n_els_epoch
    stats_stage_run["padding_tokens_pct_run"] /= n_steps

    stats_stage_best["loss_recon_is_best"] = stats_stage_run["loss_recon_run"] < stats_stage_best["loss_recon_best"]
    stats_stage_best["loss_recon_best"] = stats_stage_run["loss_recon_run"] if stats_stage_best["loss_recon_is_best"] else stats_stage_best["loss_recon_best"]
    stats_stage_best["loss_vq_is_best"] = stats_stage_run["loss_vq_run"] < stats_stage_best["loss_vq_best"]
    stats_stage_best["loss_vq_best"] = stats_stage_run["loss_vq_run"] if stats_stage_best["loss_vq_is_best"] else stats_stage_best["loss_vq_best"]
    stats_stage_best["loss_full_is_best"] = stats_stage_run["loss_full_run"] < stats_stage_best["loss_full_best"]
    stats_stage_best["loss_full_best"] = stats_stage_run["loss_full_run"] if stats_stage_best["loss_full_is_best"] else stats_stage_best["loss_full_best"]
    stats_stage_best["metric_acc_is_best"] = stats_stage_run["metric_acc_run"] > stats_stage_best["metric_acc_best"]
    stats_stage_best["metric_acc_best"] = stats_stage_run["metric_acc_run"] if stats_stage_best["metric_acc_is_best"] else stats_stage_best["metric_acc_best"]

    return stats_stage_run, stats_stage_best

def end_of_epoch_print(
    stats_stage_run: dict, stats_stage_best: dict, 
    console: Console, 
    epoch: int, print_epoch: bool,
    stat_color: str, stat_emojis: list,
    print_new_line: bool
):
    epoch_str = f"[bold {COLOR_EPOCH}]{epoch:03d}[/bold {COLOR_EPOCH}] | " if print_epoch else "    | "
    suffix_str = "\n" if print_new_line else ""

    console.print(
        epoch_str  + \
        f"loss_recon: [bold {stat_color}] {stats_stage_run['loss_recon_run']:08.6f}[/bold {stat_color}] {stat_emojis[1] if stats_stage_best['loss_recon_is_best'] else '  '} | " + \
        f"loss_vq: [bold {stat_color}] {stats_stage_run['loss_vq_run']:08.6f}[/bold {stat_color}] {stat_emojis[0] if stats_stage_best['loss_vq_is_best'] else '  '} | " + \
        f"acc: [bold {stat_color}]{stats_stage_run['metric_acc_run']:08.6f}%[/bold {stat_color}] {stat_emojis[2] if stats_stage_best['metric_acc_is_best'] else '  '} | " + \
        suffix_str
    )

def init_stats_best(): 
    return {
        "loss_recon_best": np.Inf,
        "loss_recon_is_best": False,
        "loss_vq_best": np.Inf,
        "loss_vq_is_best": False,
        "loss_full_best": np.Inf,
        "loss_full_is_best": False,
        "metric_acc_best": 0,
        "metric_acc_is_best": False
    }

def init_stats_run(): 
    return {
        "loss_recon_run": 0,
        "loss_vq_run": 0,
        "loss_full_run": 0,
        "metric_acc_run": 0,
        "padding_tokens_pct_run": 0
    }

def create_wandb_log_dict(epoch: int, stats_stage_run: dict, stage: str):
    return {
        "epoch": epoch,
        f"{stage}/loss_recon": stats_stage_run["loss_recon_run"],
        f"{stage}/loss_vq": stats_stage_run["loss_vq_run"],
        f"{stage}/loss_full": stats_stage_run["loss_full_run"],
        f"{stage}/acc": stats_stage_run["metric_acc_run"],
        f"padding_tokens_pct/{stage}": stats_stage_run["padding_tokens_pct_run"]
    } 

def decode_sentences(
    input_ids: Tensor, recon_ids: Tensor, 
    tokenizer: PreTrainedTokenizer, 
    decoded_sentences: list, 
    epoch: int,
    stage: str,
    console: Console
):

    input_ids_decoded = tokenizer.batch_decode(sequences=input_ids)
    recon_ids_decoded = tokenizer.batch_decode(sequences=recon_ids)

    for i, r in zip(input_ids_decoded, recon_ids_decoded):

        decoded_sentences.append(
            {
                "epoch": epoch,
                "stage": stage,
                "input_sentence": i,
                "recon_sentence":  r
            }
        )

    return 

def _save_ckpt(model: Shelgon, checkpoint_file_path: str, stage: str):
    save(
            {
                "model_state_dict": model.state_dict(),
                "encoder_state_dict": model.encoder.state_dict(),
                "decoder_state_dict": model.decoder.state_dict(),
            }, 
            checkpoint_file_path
            
        )

def checkpoint(stats_train_best: dict, model: Shelgon, checkpoint_dir: str, stage: str):
    
    if stats_train_best["loss_recon_is_best"]:
        _save_ckpt(model, f"{checkpoint_dir}/shelgon_ckpt_loss_recon_{stage}_best.pth", stage)
    
    if stats_train_best["loss_vq_is_best"]:
        _save_ckpt(model, f"{checkpoint_dir}/shelgon_ckpt_loss_vq_{stage}_best.pth", stage)
    
    # if stats_train_best["loss_full_is_best"]:
    #     _save_ckpt(model, f"{checkpoint_dir}/shelgon_ckpt_loss_full_{stage}_best.pth", stage)
    
    # if stats_train_best["metric_acc_is_best"]:
    #     _save_ckpt(model, f"{checkpoint_dir}/shelgon_ckpt_metric_acc_{stage}_best.pth", stage)


def train(
    prg: Progress, console: Console,
    device: device, 
    dl_train: DataLoader, dl_val: DataLoader, n_batches_train: int, n_batches_val: int,
    model: Shelgon, 
    tokenizer: PreTrainedTokenizer, tokenizer_add_special_tokens: bool, 
    n_epochs_to_decode_after: int, decoded_sentences: list,
    opt: Optimizer, 
    loss_recon_rescale_factor: float, loss_recon_weight: float, 
    loss_vq_rescale_factor: float, loss_vq_weight: float, 
    lr_sched: LRScheduler, 
    n_epochs: int, 
    vocab_size: int,
    wandb_run: Run, run_path: str,
    export_checkpoint: bool
):
    
    if not export_checkpoint:
        console.print(f"[bold {COLOR_WARNING}]Warning[/bold {COLOR_WARNING}] checkpoint exporting is [bold {COLOR_OFF}]OFF[/bold {COLOR_OFF}]!\n")
    
    prg.start()
    epochs_task = prg.add_task(f"[bold {COLOR_EPOCH}] Epochs", total=n_epochs)
    batches_task_train = prg.add_task(f"[bold {COLOR_TRAIN}] Train batches", total=n_batches_train)
    batches_task_val   = prg.add_task(f"[bold {COLOR_VAL}] Val   batches", total=n_batches_val)

    stats_train_best = init_stats_best()
    stats_val_best   = init_stats_best()

    ### Begin epochs loop ###

    for epoch in range(1, n_epochs + 1):

        prg.reset(batches_task_train)
        prg.reset(batches_task_val)

        ### Begin trainining part ### 
        
        stats_train_run = init_stats_run()
        n_els_epoch = 0
        n_steps = 0
        model.train()

        ### Begin train batches loop ### 

        for batch in list(dl_train)[:n_batches_train]:
            n_els_batch = len(batch)
            n_els_epoch += n_els_batch
            n_steps += 1

            stats_step, input_ids, recon_ids = step(
                device=device,
                model=model, 
                tokenizer=tokenizer, tokenizer_add_special_tokens=tokenizer_add_special_tokens,
                opt=opt, 
                loss_recon_rescale_factor=loss_recon_rescale_factor, loss_recon_weight=loss_recon_weight,
                loss_vq_rescale_factor=loss_vq_rescale_factor, loss_vq_weight=loss_vq_weight,
                lr_sched=lr_sched,
                batch=batch, vocab_size=vocab_size,
                console=console
            )

            if epoch % n_epochs_to_decode_after == 0:
                decode_sentences(input_ids, recon_ids, tokenizer, decoded_sentences, epoch, "train", console)

            stats_train_run = end_of_step_stats_update(stats_train_run, stats_step, n_els_batch)
            
            prg.advance(batches_task_train, 1)
            prg.advance(epochs_task, (1 / (n_batches_train + n_batches_val)))

        ### End train batches loop ### 
            
        stats_train_run, stats_train_best = end_of_epoch_stats_update(stats_train_run, stats_train_best, n_els_epoch, n_steps)
        end_of_epoch_print(stats_train_run, stats_train_best, console, epoch, True, COLOR_TRAIN, STATS_EMOJI_TRAIN, False)
        wandb_run.log(create_wandb_log_dict(epoch, stats_train_run, "train"))
        # if export_checkpoint: checkpoint(stats_train_best, model, run_path, "train")

        ### End training part ### 
        
        ### Beging validating part ### 

        stats_val_run = init_stats_run()
        n_els_epoch = 0
        n_steps = 0
        model.eval()    

        ### Begin val batches loop ### 

        for batch in list(dl_val)[:n_batches_val]:

            n_els_batch = len(batch)
            n_els_epoch += n_els_batch
            n_steps += 1

            with no_grad():

                stats_step, input_ids, recon_ids = step(
                    device=device,
                    model=model,
                    tokenizer=tokenizer, tokenizer_add_special_tokens=tokenizer_add_special_tokens,
                    opt=None, 
                    loss_recon_rescale_factor=loss_recon_rescale_factor, loss_recon_weight=loss_recon_weight,
                    loss_vq_rescale_factor=loss_vq_rescale_factor, loss_vq_weight=loss_vq_weight,
                    lr_sched=None,
                    batch=batch, vocab_size=vocab_size,
                    console=console
                )

            if epoch % n_epochs_to_decode_after == 0:
                decode_sentences(input_ids, recon_ids, tokenizer, decoded_sentences, epoch, "val", console)
            
            stats_val_run = end_of_step_stats_update(stats_val_run, stats_step, n_els_batch)

            prg.advance(batches_task_val, 1)
            prg.advance(epochs_task, (1 / (n_batches_train + n_batches_val)))

        ### End val batches loop ### 
            
        stats_val_run, stats_val_best = end_of_epoch_stats_update(stats_val_run, stats_val_best, n_els_epoch, n_steps)
        end_of_epoch_print(stats_val_run, stats_val_best, console, epoch, False, COLOR_VAL, STATS_EMOJI_VAL, epoch != n_epochs)
        wandb_run.log(create_wandb_log_dict(epoch, stats_val_run, "val"))
        if export_checkpoint: checkpoint(stats_train_best, model, run_path, "val")

        ### End validating part ### 

    ### End epochs loop ###
        
    return

def test(
    prg: Progress, console: Console,
    device: device, 
    dl_test: DataLoader, n_batches_test,
    model: Shelgon, tokenizer: PreTrainedTokenizer, tokenizer_add_special_tokens: bool,
    loss_recon_rescale_factor: float, loss_recon_weight: float,
    loss_vq_rescale_factor: float, loss_vq_weight: float,
    decoded_sentences: list,
    vocab_size: int,
    epoch: int,
    wandb_run: Run
):
    
    batches_task_test  = prg.add_task(f"[bold {COLOR_TEST}] Test  batches", total=n_batches_test)
    stats_test_best = init_stats_best()
    
    ### Beging testing part ### 

    stats_test_run = init_stats_run()
    n_els_epoch = 0
    n_steps = 0
    model.eval()    

    ### Begin val batches loop ### 

    for batch in list(dl_test)[:n_batches_test]:

        n_els_batch = len(batch)
        n_els_epoch += n_els_batch
        n_steps += 1

        with no_grad():

            stats_step, input_ids, recon_ids = step(
                device=device,
                model=model,
                tokenizer=tokenizer, tokenizer_add_special_tokens=tokenizer_add_special_tokens,
                opt=None, 
                loss_recon_rescale_factor=loss_recon_rescale_factor, loss_recon_weight=loss_recon_weight, 
                loss_vq_rescale_factor=loss_vq_rescale_factor, loss_vq_weight=loss_vq_weight,
                lr_sched=None,
                batch=batch, vocab_size=vocab_size,
                console=console
            )

        decode_sentences(input_ids, recon_ids, tokenizer, decoded_sentences, epoch, "test", console)
        
        stats_test_run = end_of_step_stats_update(stats_test_run, stats_step, n_els_batch)

        prg.advance(batches_task_test, 1)

    ### End test batches loop ### 
        
    stats_test_run, stats_test_best = end_of_epoch_stats_update(stats_test_run, stats_test_best, n_els_epoch, n_steps)
    end_of_epoch_print(stats_test_run, stats_test_best, console, epoch, False, COLOR_TEST, STATS_EMOJI_TEST, True)
    wandb_run.log(create_wandb_log_dict(epoch, stats_test_run, "test"))

    ### End testing part ###





        
    
    
    
    
    
    
    
    
    
    
    
    