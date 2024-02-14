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

from common.tensor_utils import replace_pct_rand_values, change_percentage_of_elements

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
    model: Shelgon, 
    tokenizer_encoder: PreTrainedTokenizer, tokenizer_decoder: PreTrainedTokenizer, 
    tokenizer_encoder_add_special_tokens: bool, tokenized_encoder_sentence_max_length: int,
    tokenizer_decoder_add_special_tokens: bool, tokenized_decoder_sentence_max_length: int,
    encoder_perturb_pct: float, decoder_perturb_pct: float,
    use_mask_encoder: bool, use_mask_decoder: bool,
    num_labels_per_class: int,
    opt: Optimizer, 
    lr_sched: LRScheduler,
    batch: list, 
    vocab_size_encoder: int, vocab_size_decoder: int, 
    console: Console

):

    sentences: Tensor = batch["sentence"]
    gt_one_hot: Tensor = batch["latent_classes_one_hot"].to(device)
    gt_labels: Tensor = batch["latent_classes_labels"].to(device)

    tokenized_encoder = tokenizer_encoder(
        sentences, return_tensors="pt", 
        padding="max_length", max_length=tokenized_encoder_sentence_max_length, 
        add_special_tokens=tokenizer_encoder_add_special_tokens
    )
    input_ids_encoder: Tensor = tokenized_encoder.input_ids.to(device)
    # input_ids_encoder = replace_pct_rand_values(input_ids_encoder, encoder_perturb_pct, 0, vocab_size_encoder)
    input_ids_encoder_perturbed = change_percentage_of_elements(
        input_ids_encoder, 1, encoder_perturb_pct, 0, vocab_size_encoder
    )
    attention_mask_encoder: Tensor = tokenized_encoder.attention_mask.to(device)

    tokenized_decoder = tokenizer_decoder(
        sentences, return_tensors="pt", 
        padding="max_length", max_length=tokenized_decoder_sentence_max_length, 
        add_special_tokens=tokenizer_decoder_add_special_tokens
    )
    input_ids_decoder: Tensor = tokenized_decoder.input_ids.to(device)
    # input_ids_decoder = replace_pct_rand_values(input_ids_decoder, decoder_perturb_pct, 0, vocab_size_decoder)
    input_ids_decoder_perturbed = change_percentage_of_elements(
        input_ids_decoder, 1, decoder_perturb_pct, 0, vocab_size_decoder
    )
    attention_mask_decoder: Tensor = tokenized_decoder.attention_mask.to(device)
    
    logits_recon: Tensor; logits_pred: Tensor
    logits_recon, logits_pred = model.forward(
        input_ids_encoder_perturbed, attention_mask_encoder if use_mask_encoder else None, 
        input_ids_decoder_perturbed, attention_mask_decoder if use_mask_decoder else None
    )

    # input and targets reshaped to use KL divergence with sequential data, as per https://github.com/florianmai/emb2emb/blob/master/autoencoders/autoencoder.py#L116C13-L116C58
    loss_recon_step = kl_div(
        input=log_softmax(logits_recon.reshape(-1, vocab_size_decoder), dim=-1), 
        target=one_hot(input_ids_decoder, vocab_size_decoder).reshape(-1, vocab_size_decoder).float(),
        reduction="batchmean"
    )
    
    loss_pred_step = kl_div(
        input=log_softmax(logits_pred, dim=-1).reshape(-1, num_labels_per_class),
        target=gt_one_hot.reshape(-1, num_labels_per_class).float().to(device),
        reduction="batchmean"
    )

    loss_full_step: Tensor = loss_recon_step + loss_pred_step
    
    recon_ids = argmax(softmax(logits_recon, dim=-1), dim=-1)
    acc_recon_step_per_batch, acc_recon_step_per_sentence = seq_acc(recon_ids, input_ids_decoder)

    acc_pred_step_per_batch, acc_pred_step_per_sentence = seq_acc(
        argmax(softmax(logits_pred, dim=-1), dim=-1), gt_labels
    )

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
        "loss_pred_step": loss_pred_step, 
        "loss_full_step": loss_full_step, 
        "acc_recon_step_per_batch": acc_recon_step_per_batch,
        "acc_recon_step_per_sentence": acc_recon_step_per_sentence,
        "acc_pred_step_per_batch": acc_pred_step_per_batch,
        "acc_pred_step_per_sentence": acc_pred_step_per_sentence,
        "padding_tokens_pct_step": -69 #count_pct_padding_tokens(input_ids, console)
    }, input_ids_encoder, input_ids_decoder, recon_ids, logits_pred, batch["latent_classes_labels"]

def end_of_step_stats_update(stats_stage_run: dict, stats_step: dict, n_els_batch: int):
    
    stats_stage_run["loss_recon_run"] += stats_step["loss_recon_step"] * n_els_batch
    stats_stage_run["loss_pred_run"] += stats_step["loss_pred_step"] * n_els_batch
    stats_stage_run["loss_full_run"] += stats_step["loss_full_step"] * n_els_batch
    stats_stage_run["acc_recon_run"] += stats_step["acc_recon_step_per_batch"] * n_els_batch * 1e2
    stats_stage_run["acc_pred_run"] += stats_step["acc_pred_step_per_batch"] * n_els_batch * 1e2
    stats_stage_run["padding_tokens_pct_run"] += stats_step["padding_tokens_pct_step"]
    
    return stats_stage_run

def end_of_epoch_stats_update(stats_stage_run: dict, stats_stage_best: dict, n_els_epoch: int, n_steps: int):

    stats_stage_run["loss_recon_run"] /= n_els_epoch
    stats_stage_run["loss_pred_run"] /= n_els_epoch
    stats_stage_run["loss_full_run"] /= n_els_epoch
    stats_stage_run["acc_recon_run"] /= n_els_epoch
    stats_stage_run["acc_pred_run"] /= n_els_epoch
    stats_stage_run["padding_tokens_pct_run"] /= n_steps

    stats_stage_best["loss_recon_is_best"] = stats_stage_run["loss_recon_run"] < stats_stage_best["loss_recon_best"]
    stats_stage_best["loss_recon_best"] = stats_stage_run["loss_recon_run"] if stats_stage_best["loss_recon_is_best"] else stats_stage_best["loss_recon_best"]
    stats_stage_best["loss_pred_is_best"] = stats_stage_run["loss_pred_run"] < stats_stage_best["loss_pred_best"]
    stats_stage_best["loss_pred_best"] = stats_stage_run["loss_pred_run"] if stats_stage_best["loss_pred_is_best"] else stats_stage_best["loss_pred_best"]
    stats_stage_best["loss_full_is_best"] = stats_stage_run["loss_full_run"] < stats_stage_best["loss_full_best"]
    stats_stage_best["loss_full_best"] = stats_stage_run["loss_full_run"] if stats_stage_best["loss_full_is_best"] else stats_stage_best["loss_full_best"]
    stats_stage_best["acc_recon_is_best"] = stats_stage_run["acc_recon_run"] > stats_stage_best["acc_recon_best"]
    stats_stage_best["acc_recon_best"] = stats_stage_run["acc_recon_run"] if stats_stage_best["acc_recon_is_best"] else stats_stage_best["acc_recon_best"]
    stats_stage_best["acc_pred_is_best"] = stats_stage_run["acc_pred_run"] > stats_stage_best["acc_pred_best"]
    stats_stage_best["acc_pred_best"] = stats_stage_run["acc_pred_run"] if stats_stage_best["acc_pred_is_best"] else stats_stage_best["acc_pred_best"]

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
        f"loss_recon: [bold {stat_color}] {stats_stage_run['loss_recon_run']:09.6f}[/bold {stat_color}] {stat_emojis[0] if stats_stage_best['loss_recon_is_best'] else '  '} | " + \
        f"acc_recon: [bold {stat_color}]{stats_stage_run['acc_recon_run']:09.6f}%[/bold {stat_color}] {stat_emojis[1] if stats_stage_best['acc_recon_is_best'] else '  '} | " + \
        f"loss_pred: [bold {stat_color}] {stats_stage_run['loss_pred_run']:09.6f}[/bold {stat_color}] {stat_emojis[2] if stats_stage_best['loss_pred_is_best'] else '  '} | " + \
        f"acc_pred: [bold {stat_color}]{stats_stage_run['acc_pred_run']:09.6f}%[/bold {stat_color}] {stat_emojis[3] if stats_stage_best['acc_pred_is_best'] else '  '} | " + \
        suffix_str
    )

def init_stats_best(): 
    return {
        "loss_recon_best": np.Inf,
        "loss_recon_is_best": False,
        "loss_pred_best": np.Inf,
        "loss_pred_is_best": False,
        "loss_full_best": np.Inf,
        "loss_full_is_best": False,
        "acc_recon_best": 0,
        "acc_recon_is_best": False,
        "acc_pred_best": 0,
        "acc_pred_is_best": False
    }

def init_stats_run(): 
    return {
        "loss_recon_run": 0,
        "loss_pred_run": 0,
        "loss_full_run": 0,
        "acc_recon_run": 0,
        "acc_pred_run": 0,
        "padding_tokens_pct_run": 0
    }

def create_wandb_log_dict(epoch: int, stats_stage_run: dict, stage: str):
    return {
        "epoch": epoch,
        f"{stage}/loss_recon": stats_stage_run["loss_recon_run"],
        f"{stage}/loss_pred": stats_stage_run["loss_pred_run"],
        f"{stage}/loss_full": stats_stage_run["loss_full_run"],
        f"{stage}/acc_recon": stats_stage_run["acc_recon_run"],
        f"{stage}/acc_pred": stats_stage_run["acc_pred_run"],
        f"padding_tokens_pct/{stage}": stats_stage_run["padding_tokens_pct_run"]
    } 


def explicit_latent_classes_labels(latent_classes_labels: Tensor, console: Console):

    latent_to_explicit_map = [
        # latent factor 0 --> paper latent factor 3 --> sentence type
        {
            "0": "declarative",
            "1": "interrogative"
        }, 
        
        # latent factor 1 --> paper latent factor 6 --> grammatical number person
        {
            "0": "1st",
            "1": "2nd",
            "2": "3rd"
        }, 
        
        # latent factor 2 --> paper latent factor 7 --> sentence negation
        {
            "0": "affirmative",
            "1": "negative"
        }, 
        
        # latent factor 3 --> paper latent factor 8 --> verb tense
        {
            "0": "past",
            "1": "present",
            "2": "future"
        }, 
        
        # latent factor 4 --> paper latent factor 9 --> style
        {
            "0": "not_progressive",
            "1": "progressive",
        }, 
    ]

    explicit = {
        "sentence_type": latent_to_explicit_map[0][str(latent_classes_labels[0].item())],
        "grammatical_number_person": latent_to_explicit_map[1][str(latent_classes_labels[1].item())],
        "sentence_negation": latent_to_explicit_map[2][str(latent_classes_labels[2].item())],
        "verb_tense": latent_to_explicit_map[3][str(latent_classes_labels[3].item())],
        "sentence_style": latent_to_explicit_map[4][str(latent_classes_labels[4].item())],
    }

    return explicit


def decode_sentences(
    input_ids_encoder: Tensor, input_ids_decoder: Tensor, recon_ids: Tensor, latent_classes_labels: Tensor,
    stats_step: dict,
    tokenizer_encoder: PreTrainedTokenizer, tokenizer_decoder: PreTrainedTokenizer, 
    decoded_sentences: list, 
    epoch: int,
    stage: str,
    console: Console
):

    input_ids_decoded = tokenizer_encoder.batch_decode(sequences=input_ids_encoder, skip_special_tokens=True)
    recon_ids_decoded = tokenizer_decoder.batch_decode(sequences=recon_ids, skip_special_tokens=True)
    recon_accs: Tensor = stats_step["acc_recon_step_per_sentence"]
    pred_accs: Tensor = stats_step["acc_pred_step_per_sentence"]

    for i, r, a_r, a_p, l in zip(input_ids_decoded, recon_ids_decoded, recon_accs, pred_accs, latent_classes_labels):

        decoded_sentences.append(
            {
                "epoch": epoch,
                "stage": stage,
                "input_sentence": i,
                "recon_sentence":  r,
                "recon_acc": a_r.cpu().item(),
                "pred_acc": a_p.cpu().item()
            }
        )

        decoded_sentences[-1].update(explicit_latent_classes_labels(l, console))

    return 

def _save_ckpt(model: Shelgon, checkpoint_file_path: str, stage: str):
    save(
            {
                "model_state_dict": model.state_dict(),
                "encoder_state_dict": model.encoder.state_dict(),
                "decoder_state_dict": model.decoder.state_dict()
            }, 
            checkpoint_file_path
            
        )

def checkpoint(stats_train_best: dict, model: Shelgon, checkpoint_dir: str, stage: str):
    
    if stats_train_best["loss_recon_is_best"]:
        _save_ckpt(model, f"{checkpoint_dir}/Shelgon_ckpt_loss_recon_{stage}_best.pth", stage)
    
    if stats_train_best["acc_recon_is_best"]:
        _save_ckpt(model, f"{checkpoint_dir}/Shelgon_ckpt_metric_acc_{stage}_best.pth", stage)


def train(
    prg: Progress, console: Console,
    device: device, 
    dl_train: DataLoader, dl_val: DataLoader, n_batches_train: int, n_batches_val: int,
    model: Shelgon, 
    tokenizer_encoder: PreTrainedTokenizer, tokenizer_decoder: PreTrainedTokenizer, 
    tokenizer_encoder_add_special_tokens: bool, tokenized_encoder_sentence_max_length: int, 
    tokenizer_decoder_add_special_tokens: bool, tokenized_decoder_sentence_max_length: int, 
    encoder_perturb_train_pct: float, encoder_perturb_val_pct: float,
    decoder_perturb_train_pct: float, decoder_perturb_val_pct: float,
    use_mask_encoder: bool, use_mask_decoder: bool,
    num_labels_per_class: int,
    n_epochs_to_decode_after: int, decoded_sentences: list,
    opt: Optimizer, lr_sched: LRScheduler, 
    n_epochs: int, 
    vocab_size_encoder: int, vocab_size_decoder: int,
    wandb_run: Run, run_path: str
):
    
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

            stats_step, input_ids_encoder, input_ids_decoder, recon_ids, logits_pred, latent_classes_labels = step(
                device=device,
                model=model, 
                tokenizer_encoder=tokenizer_encoder, tokenizer_decoder=tokenizer_decoder,
                tokenizer_encoder_add_special_tokens=tokenizer_encoder_add_special_tokens, tokenized_encoder_sentence_max_length=tokenized_encoder_sentence_max_length,
                tokenizer_decoder_add_special_tokens=tokenizer_decoder_add_special_tokens, tokenized_decoder_sentence_max_length=tokenized_decoder_sentence_max_length,
                encoder_perturb_pct=encoder_perturb_train_pct, decoder_perturb_pct=decoder_perturb_train_pct,
                use_mask_encoder=use_mask_encoder, use_mask_decoder=use_mask_decoder,
                num_labels_per_class=num_labels_per_class,
                opt=opt, 
                lr_sched=lr_sched,
                batch=batch, 
                vocab_size_encoder=vocab_size_encoder, vocab_size_decoder=vocab_size_decoder,
                console=console
            )

            if epoch % n_epochs_to_decode_after == 0:
                decode_sentences(
                    input_ids_encoder, input_ids_decoder, recon_ids, latent_classes_labels, 
                    stats_step, 
                    tokenizer_encoder, tokenizer_decoder, 
                    decoded_sentences, 
                    epoch, "train", 
                    console
                )

            stats_train_run = end_of_step_stats_update(stats_train_run, stats_step, n_els_batch)
            
            prg.advance(batches_task_train, 1)
            prg.advance(epochs_task, (1 / (n_batches_train + n_batches_val)))

        ### End train batches loop ### 
            
        stats_train_run, stats_train_best = end_of_epoch_stats_update(stats_train_run, stats_train_best, n_els_epoch, n_steps)
        end_of_epoch_print(stats_train_run, stats_train_best, console, epoch, True, COLOR_TRAIN, STATS_EMOJI_TRAIN, False)
        wandb_run.log(create_wandb_log_dict(epoch, stats_train_run, "train"))
        checkpoint(stats_train_best, model, run_path, "train")

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

                stats_step, input_ids_encoder, input_ids_decoder, recon_ids, logits_pred, latent_classes_labels = step(
                    device=device,
                    model=model, 
                    tokenizer_encoder=tokenizer_encoder, tokenizer_decoder=tokenizer_decoder,
                    tokenizer_encoder_add_special_tokens=tokenizer_encoder_add_special_tokens, tokenized_encoder_sentence_max_length=tokenized_encoder_sentence_max_length,
                    tokenizer_decoder_add_special_tokens=tokenizer_decoder_add_special_tokens, tokenized_decoder_sentence_max_length=tokenized_decoder_sentence_max_length,
                    encoder_perturb_pct=encoder_perturb_val_pct, decoder_perturb_pct=decoder_perturb_val_pct,
                    use_mask_encoder=use_mask_encoder, use_mask_decoder=use_mask_decoder,
                    num_labels_per_class=num_labels_per_class,
                    opt=None, 
                    lr_sched=None,
                    batch=batch, 
                    vocab_size_encoder=vocab_size_encoder, vocab_size_decoder=vocab_size_decoder,
                    console=console
                )

            if epoch % n_epochs_to_decode_after == 0:
                decode_sentences(
                    input_ids_encoder, input_ids_decoder, recon_ids, latent_classes_labels, 
                    stats_step, 
                    tokenizer_encoder, tokenizer_decoder, 
                    decoded_sentences, 
                    epoch, "val", 
                    console
                )
            
            stats_val_run = end_of_step_stats_update(stats_val_run, stats_step, n_els_batch)

            prg.advance(batches_task_val, 1)
            prg.advance(epochs_task, (1 / (n_batches_train + n_batches_val)))

        ### End val batches loop ### 
            
        stats_val_run, stats_val_best = end_of_epoch_stats_update(stats_val_run, stats_val_best, n_els_epoch, n_steps)
        end_of_epoch_print(stats_val_run, stats_val_best, console, epoch, False, COLOR_VAL, STATS_EMOJI_VAL, epoch != n_epochs)
        wandb_run.log(create_wandb_log_dict(epoch, stats_val_run, "val"))
        checkpoint(stats_train_best, model, run_path, "val")

        ### End validating part ### 

    ### End epochs loop ###
        
    return

def test(
    prg: Progress, console: Console,
    device: device, 
    dl_test: DataLoader, n_batches_test,
    model: Shelgon, 
    tokenizer_encoder: PreTrainedTokenizer, tokenizer_decoder: PreTrainedTokenizer,
    tokenizer_encoder_add_special_tokens: bool, tokenized_encoder_sentence_max_length: int,
    tokenizer_decoder_add_special_tokens: bool, tokenized_decoder_sentence_max_length: int,
    encoder_perturb_test_pct: float, decoder_perturb_test_pct: float,
    use_mask_encoder: bool, use_mask_decoder: bool,
    num_labels_per_class: int,
    decoded_sentences: list,
    vocab_size_encoder: int, vocab_size_decoder: int,
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

            stats_step, input_ids_encoder, input_ids_decoder, recon_ids, logits_pred, latent_classes_labels = step(
                device=device,
                model=model, 
                tokenizer_encoder=tokenizer_encoder, tokenizer_decoder=tokenizer_decoder,
                tokenizer_encoder_add_special_tokens=tokenizer_encoder_add_special_tokens, tokenized_encoder_sentence_max_length=tokenized_encoder_sentence_max_length,
                tokenizer_decoder_add_special_tokens=tokenizer_decoder_add_special_tokens, tokenized_decoder_sentence_max_length=tokenized_decoder_sentence_max_length,
                encoder_perturb_pct=encoder_perturb_test_pct, decoder_perturb_pct=decoder_perturb_test_pct,
                use_mask_encoder=use_mask_encoder, use_mask_decoder=use_mask_decoder,
                num_labels_per_class=num_labels_per_class,
                opt=None, 
                lr_sched=None,
                batch=batch, 
                vocab_size_encoder=vocab_size_encoder, vocab_size_decoder=vocab_size_decoder,
                console=console
            )

        decode_sentences(
            input_ids_encoder, input_ids_decoder, recon_ids, latent_classes_labels,
            stats_step, 
            tokenizer_encoder, tokenizer_decoder, 
            decoded_sentences, 
            epoch, "test", 
            console
        )
        
        stats_test_run = end_of_step_stats_update(stats_test_run, stats_step, n_els_batch)

        prg.advance(batches_task_test, 1)

    ### End test batches loop ### 
        
    stats_test_run, stats_test_best = end_of_epoch_stats_update(stats_test_run, stats_test_best, n_els_epoch, n_steps)
    end_of_epoch_print(stats_test_run, stats_test_best, console, epoch, False, COLOR_TEST, STATS_EMOJI_TEST, True)
    wandb_run.log(create_wandb_log_dict(epoch, stats_test_run, "test"))

    ### End testing part ###





        
    
    
    
    
    
    
    
    
    
    
    
    