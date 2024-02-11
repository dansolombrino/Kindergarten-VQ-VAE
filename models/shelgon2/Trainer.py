from rich import print

from rich.progress import Progress

from rich.console import Console

from torch.cuda import device

from torch.utils.data import DataLoader

from Shelgon2 import Shelgon2

from transformers import PreTrainedTokenizer

from torch.optim.optimizer import Optimizer

from torch.optim.lr_scheduler import LRScheduler

from torch import Tensor

from torch.nn.functional import one_hot

from torch.nn.functional import cross_entropy
from torch.nn.functional import kl_div
from torch.nn.functional import mse_loss

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

from math import isclose

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
    model: Shelgon2, tokenizer: PreTrainedTokenizer, tokenizer_add_special_tokens: bool, tokenizer_max_length: int,
    opt: Optimizer, 
    loss_recon_rescale_factor: float, loss_recon_weight: float,
    loss_latent_rescale_factor: float, loss_latent_weight: float,
    lr_sched: LRScheduler,
    batch: list, vocab_size: int,
    stage: str, 
    console: Console

):
    sentences: Tensor = batch["sentence"]
    input_latent_classes_labels: Tensor = batch["latent_classes_labels"]
    input_latent_classes_labels = input_latent_classes_labels.to(device)
    input_latent_classes_one_hot: Tensor = batch["latent_classes_one_hot"]
    input_latent_classes_one_hot = input_latent_classes_one_hot.to(device)

    tokenized = tokenizer(
        sentences, return_tensors="pt", 
        padding="max_length", max_length=tokenizer_max_length, 
        add_special_tokens=tokenizer_add_special_tokens
    )
    input_ids: Tensor = tokenized.input_ids.to(device)
    attention_mask: Tensor = tokenized.attention_mask.to(device)

    recon_word_logits: Tensor; recon_latent_classes_logits: Tensor; recon_latent_classes_labels: Tensor
    recon_word_logits, recon_latent_classes_logits, recon_latent_classes_labels = model.forward(input_ids, attention_mask)
    recon_latent_classes_logits = recon_latent_classes_logits.permute((0, 2, 1))

    loss_recon_step: Tensor = kl_div(
        input=log_softmax(recon_word_logits.reshape(-1, vocab_size), dim=-1), 
        target=one_hot(input_ids, vocab_size).reshape(-1, vocab_size).float(),
        reduction="batchmean"
    )
    loss_latent_step: Tensor = kl_div(
        input=log_softmax(recon_latent_classes_logits.reshape(-1, 3), dim=-1),
        target=input_latent_classes_one_hot.reshape(-1, 3).float(),
        reduction="batchmean"
    )
    
    recon_ids = argmax(softmax(recon_word_logits, dim=-1), dim=-1)
    metric_acc_step, metric_acc_per_sentence_step = seq_acc(recon_ids, input_ids)

    metric_latent_acc_step, metric_latent_acc_per_sentence_step = seq_acc(input=recon_latent_classes_labels, target=input_latent_classes_labels)

    loss_recon_step *= loss_recon_rescale_factor * loss_recon_weight
    loss_latent_step *= loss_latent_rescale_factor * loss_latent_weight
    loss_full_step: Tensor = loss_recon_step + loss_latent_step

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
        "loss_latent_step": loss_latent_step, 
        "loss_full_step": loss_full_step, 
        "metric_acc_step": metric_acc_step,
        "metric_latent_acc_step": metric_latent_acc_step,
        "metric_acc_per_sentence_step": metric_acc_per_sentence_step,
        "metric_latent_acc_per_sentence_step": metric_latent_acc_per_sentence_step,
        "padding_tokens_pct_step": -69 #count_pct_padding_tokens(input_ids, console)
    }, input_ids, recon_ids, input_latent_classes_labels, recon_latent_classes_labels, input_latent_classes_one_hot, recon_latent_classes_logits

def end_of_step_stats_update(stats_stage_run: dict, stats_step: dict, n_els_batch: int):
    
    stats_stage_run["loss_recon_run"] += stats_step["loss_recon_step"] * n_els_batch
    stats_stage_run["loss_latent_run"] += stats_step["loss_latent_step"] * n_els_batch
    stats_stage_run["loss_full_run"] += stats_step["loss_full_step"] * n_els_batch
    stats_stage_run["metric_acc_run"] += stats_step["metric_acc_step"] * n_els_batch * 1e2
    stats_stage_run["metric_latent_acc_run"] += stats_step["metric_latent_acc_step"] * n_els_batch * 1e2
    stats_stage_run["padding_tokens_pct_run"] += stats_step["padding_tokens_pct_step"]
    
    return stats_stage_run

def end_of_epoch_stats_update(stats_stage_run: dict, stats_stage_best: dict, n_els_epoch: int, n_steps: int):

    stats_stage_run["loss_recon_run"] /= n_els_epoch
    stats_stage_run["loss_latent_run"] /= n_els_epoch
    stats_stage_run["loss_full_run"] /= n_els_epoch
    stats_stage_run["metric_acc_run"] /= n_els_epoch
    stats_stage_run["metric_latent_acc_run"] /= n_els_epoch
    stats_stage_run["padding_tokens_pct_run"] /= n_steps

    stats_stage_best["loss_recon_is_best"] = stats_stage_run["loss_recon_run"] < stats_stage_best["loss_recon_best"]
    stats_stage_best["loss_recon_best"] = stats_stage_run["loss_recon_run"] if stats_stage_best["loss_recon_is_best"] else stats_stage_best["loss_recon_best"]
    stats_stage_best["loss_latent_is_best"] = stats_stage_run["loss_latent_run"] < stats_stage_best["loss_latent_best"]
    stats_stage_best["loss_latent_best"] = stats_stage_run["loss_latent_run"] if stats_stage_best["loss_latent_is_best"] else stats_stage_best["loss_latent_best"]
    stats_stage_best["loss_full_is_best"] = stats_stage_run["loss_full_run"] < stats_stage_best["loss_full_best"]
    stats_stage_best["loss_full_best"] = stats_stage_run["loss_full_run"] if stats_stage_best["loss_full_is_best"] else stats_stage_best["loss_full_best"]
    stats_stage_best["metric_acc_is_best"] = stats_stage_run["metric_acc_run"] > stats_stage_best["metric_acc_best"]
    stats_stage_best["metric_acc_best"] = stats_stage_run["metric_acc_run"] if stats_stage_best["metric_acc_is_best"] else stats_stage_best["metric_acc_best"]
    stats_stage_best["metric_latent_acc_is_best"] = stats_stage_run["metric_latent_acc_run"] > stats_stage_best["metric_latent_acc_best"]
    stats_stage_best["metric_latent_acc_best"] = stats_stage_run["metric_latent_acc_run"] if stats_stage_best["metric_latent_acc_is_best"] else stats_stage_best["metric_latent_acc_best"]

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
        f"loss_recon: [bold {stat_color}] {stats_stage_run['loss_recon_run']:08.6f}[/bold {stat_color}] {stat_emojis[0] if stats_stage_best['loss_recon_is_best'] else '  '} | " + \
        f"loss_latent: [bold {stat_color}] {stats_stage_run['loss_latent_run']:08.6f}[/bold {stat_color}] {stat_emojis[1] if stats_stage_best['loss_latent_is_best'] else '  '} | " + \
        f"acc_recon: [bold {stat_color}]{stats_stage_run['metric_acc_run']:08.6f}%[/bold {stat_color}] {stat_emojis[2] if stats_stage_best['metric_acc_is_best'] else '  '} | " + \
        f"acc_latent: [bold {stat_color}]{stats_stage_run['metric_latent_acc_run']:08.6f}%[/bold {stat_color}] {stat_emojis[3] if stats_stage_best['metric_latent_acc_is_best'] else '  '} | " + \
        suffix_str
    )

def init_stats_best(): 
    return {
        "loss_recon_best": np.Inf,
        "loss_recon_is_best": False,
        "loss_latent_best": np.Inf,
        "loss_latent_is_best": False,
        "loss_full_best": np.Inf,
        "loss_full_is_best": False,
        "metric_acc_best": 0,
        "metric_acc_is_best": False,
        "metric_latent_acc_best": 0,
        "metric_latent_acc_is_best": False
    }

def init_stats_run(): 
    return {
        "loss_recon_run": 0,
        "loss_latent_run": 0,
        "loss_full_run": 0,
        "metric_acc_run": 0,
        "metric_latent_acc_run": 0,
        "padding_tokens_pct_run": 0
    }

def create_wandb_log_dict(epoch: int, stats_stage_run: dict, stage: str):
    return {
        "epoch": epoch,
        f"{stage}/loss_recon": stats_stage_run["loss_recon_run"],
        f"{stage}/loss_latent": stats_stage_run["loss_latent_run"],
        f"{stage}/loss_full": stats_stage_run["loss_full_run"],
        f"{stage}/acc": stats_stage_run["metric_acc_run"],
        f"{stage}/latent_acc": stats_stage_run["metric_latent_acc_run"],
        f"padding_tokens_pct/{stage}": stats_stage_run["padding_tokens_pct_run"]
    } 

def decode_sentences(
    input_ids: Tensor, recon_ids: Tensor, 
    sentence_accs: Tensor, sentence_latent_accs: Tensor,
    input_latent_classes_labels: Tensor, recon_latent_classes_labels: Tensor, 
    input_latent_classes_one_hot: Tensor, recon_latent_classes_logits: Tensor,
    tokenizer: PreTrainedTokenizer, 
    decoded_sentences: list, 
    epoch: int,
    stage: str,
    console: Console
):

    input_ids_decoded = tokenizer.batch_decode(sequences=input_ids, skip_special_tokens=True)
    recon_ids_decoded = tokenizer.batch_decode(sequences=recon_ids, skip_special_tokens=True)

    for in_sentence, recon_sentence, sentence_acc, sentence_latent_acc, in_labels, recon_labels, in_one_hot, recon_one_hot in zip(
        input_ids_decoded, recon_ids_decoded,
        sentence_accs, sentence_latent_accs,
        input_latent_classes_labels, recon_latent_classes_labels, 
        input_latent_classes_one_hot, recon_latent_classes_logits
    ):
        
        in_one_hot = in_one_hot.cpu().numpy()
        recon_one_hot = recon_one_hot.detach().cpu().numpy()

        decoded_sentences.append(
            {
                "epoch": epoch,
                "stage": stage,
                "input_sentence": in_sentence,
                "recon_sentence":  recon_sentence,
                "input_latent_classes_labels": in_labels.cpu().numpy(),
                "recon_latent_classes_labels": recon_labels.cpu().numpy(),
                "input_latent_classes_one_hot_0": in_one_hot[0],
                "input_latent_classes_one_hot_1": in_one_hot[1],
                "input_latent_classes_one_hot_2": in_one_hot[2],
                "input_latent_classes_one_hot_3": in_one_hot[3],
                "input_latent_classes_one_hot_4": in_one_hot[4],
                "input_latent_classes_one_hot_5": in_one_hot[5],
                "input_latent_classes_one_hot_6": in_one_hot[6],
                "input_latent_classes_one_hot_7": in_one_hot[7],
                "recon_latent_classes_one_hot_0": recon_one_hot[0],
                "recon_latent_classes_one_hot_1": recon_one_hot[1],
                "recon_latent_classes_one_hot_2": recon_one_hot[2],
                "recon_latent_classes_one_hot_3": recon_one_hot[3],
                "recon_latent_classes_one_hot_4": recon_one_hot[4],
                "recon_latent_classes_one_hot_5": recon_one_hot[5],
                "recon_latent_classes_one_hot_6": recon_one_hot[6],
                "recon_latent_classes_one_hot_7": recon_one_hot[7],
                "sentence_acc": sentence_acc.cpu().item(),
                "sentence_latent_acc": sentence_latent_acc.cpu().item()
            }
        )

    return 

def _save_ckpt(model: Shelgon2, checkpoint_file_path: str, stage: str):
    save(
            {
                "model_state_dict": model.state_dict(),
                "encoder_state_dict": model.encoder.state_dict(),
                "decoder_state_dict": model.decoder.state_dict(),
            }, 
            checkpoint_file_path
            
        )

def checkpoint(stats_train_best: dict, model: Shelgon2, checkpoint_dir: str, stage: str):
    
    if stats_train_best["loss_recon_is_best"]:
        _save_ckpt(model, f"{checkpoint_dir}/shelgon2_ckpt_loss_recon_{stage}_best.pth", stage)
    
    if stats_train_best["loss_latent_is_best"]:
        _save_ckpt(model, f"{checkpoint_dir}/shelgon2_ckpt_loss_latent_{stage}_best.pth", stage)


def train(
    prg: Progress, console: Console,
    device: device, 
    dl_train: DataLoader, dl_val: DataLoader, n_batches_train: int, n_batches_val: int,
    model: Shelgon2, mask_pct_train: float, mask_pct_val: float,
    tokenizer: PreTrainedTokenizer, tokenizer_add_special_tokens: bool, tokenizer_max_length: int,
    n_epochs_to_decode_after: int, decoded_sentences: list,
    opt: Optimizer, 
    loss_recon_rescale_factor: float, loss_recon_weight: float, 
    loss_latent_rescale_factor: float, loss_latent_weight: float, 
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

            stats_step, input_ids, recon_ids, input_latent_classes_labels, recon_latent_classes_labels, input_latent_classes_one_hot, recon_latent_classes_logits = step(
                device=device,
                model=model, mask_pct=mask_pct_train,
                tokenizer=tokenizer, tokenizer_add_special_tokens=tokenizer_add_special_tokens, tokenizer_max_length=tokenizer_max_length,
                opt=opt, 
                loss_recon_rescale_factor=loss_recon_rescale_factor, loss_recon_weight=loss_recon_weight,
                loss_latent_rescale_factor=loss_latent_rescale_factor, loss_latent_weight=loss_latent_weight,
                lr_sched=lr_sched,
                batch=batch, vocab_size=vocab_size,
                stage="train",
                console=console
            )

            if epoch % n_epochs_to_decode_after == 0:
                decode_sentences(
                    input_ids, recon_ids, 
                    stats_step["metric_acc_per_sentence_step"],
                    stats_step["metric_latent_acc_per_sentence_step"],
                    input_latent_classes_labels, recon_latent_classes_labels, 
                    input_latent_classes_one_hot, recon_latent_classes_logits,
                    tokenizer, decoded_sentences, epoch, "train", console
                )

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

                stats_step, input_ids, recon_ids, input_latent_classes_labels, recon_latent_classes_labels, input_latent_classes_one_hot, recon_latent_classes_logits = step(
                    device=device,
                    model=model, mask_pct=mask_pct_val,
                    tokenizer=tokenizer, tokenizer_add_special_tokens=tokenizer_add_special_tokens, tokenizer_max_length=tokenizer_max_length,
                    opt=None, 
                    loss_recon_rescale_factor=loss_recon_rescale_factor, loss_recon_weight=loss_recon_weight,
                    loss_latent_rescale_factor=loss_latent_rescale_factor, loss_latent_weight=loss_latent_weight,
                    lr_sched=None,
                    batch=batch, vocab_size=vocab_size,
                    stage="val",
                    console=console
                )

            if epoch % n_epochs_to_decode_after == 0:
                decode_sentences(
                    input_ids, recon_ids, 
                    stats_step["metric_acc_per_sentence_step"],
                    stats_step["metric_latent_acc_per_sentence_step"],
                    input_latent_classes_labels, recon_latent_classes_labels, 
                    input_latent_classes_one_hot, recon_latent_classes_logits,
                    tokenizer, decoded_sentences, epoch, "val", console
                )
            
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
    model: Shelgon2, mask_pct: float,
    tokenizer: PreTrainedTokenizer, tokenizer_add_special_tokens: bool, tokenizer_max_length: int,
    loss_recon_rescale_factor: float, loss_recon_weight: float,
    loss_latent_rescale_factor: float, loss_latent_weight: float,
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

            stats_step, input_ids, recon_ids, input_latent_classes_labels, recon_latent_classes_labels, input_latent_classes_one_hot, recon_latent_classes_logits = step(
                device=device,
                model=model, mask_pct=mask_pct,
                tokenizer=tokenizer, tokenizer_add_special_tokens=tokenizer_add_special_tokens, tokenizer_max_length=tokenizer_max_length,
                opt=None, 
                loss_recon_rescale_factor=loss_recon_rescale_factor, loss_recon_weight=loss_recon_weight, 
                loss_latent_rescale_factor=loss_latent_rescale_factor, loss_latent_weight=loss_latent_weight,
                lr_sched=None,
                batch=batch, vocab_size=vocab_size,
                stage="test",
                console=console
            )

        decode_sentences(
            input_ids, recon_ids, 
            stats_step["metric_acc_per_sentence_step"],
            stats_step["metric_latent_acc_per_sentence_step"], 
            input_latent_classes_labels, recon_latent_classes_labels, 
            input_latent_classes_one_hot, recon_latent_classes_logits,
            tokenizer, decoded_sentences, epoch, "test", console
        )
        
        stats_test_run = end_of_step_stats_update(stats_test_run, stats_step, n_els_batch)

        prg.advance(batches_task_test, 1)

    ### End test batches loop ### 
        
    stats_test_run, stats_test_best = end_of_epoch_stats_update(stats_test_run, stats_test_best, n_els_epoch, n_steps)
    end_of_epoch_print(stats_test_run, stats_test_best, console, epoch, False, COLOR_TEST, STATS_EMOJI_TEST, True)
    wandb_run.log(create_wandb_log_dict(epoch, stats_test_run, "test"))

    ### End testing part ###





        
    
    
    
    
    
    
    
    
    
    
    
    