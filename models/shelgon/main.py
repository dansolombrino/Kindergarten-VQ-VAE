from config import *

import torch

from datasets.dSentences.dSentencesDataset import dSentencesDataset

from torch.utils.data import random_split, DataLoader

from Shelgon import Shelgon

from transformers import BertTokenizerFast, PreTrainedTokenizer, GPT2TokenizerFast

from torch.optim.adam import Adam

from torch.optim.lr_scheduler import MultiStepLR

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn,TimeElapsedColumn, TimeRemainingColumn
from rich.style import Style
from rich.console import Console

import wandb

from Trainer import train, test

from datetime import datetime

from common.consts import *

import os

import pandas as pd

import json

from rich import print

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = dSentencesDataset(DATASET_PATH, LATENT_CLASSES_LABELS_PATH, LATENT_CLASSES_ONE_HOT_PATH)
    
    ds_train_len = int(len(ds) * TRAIN_SPLIT_PCT)
    ds_val_len = int(len(ds) * VAL_SPLIT_PCT)
    ds_test_len = len(ds) - ds_train_len - ds_val_len
    ds_gen = torch.Generator()
    ds_gen.manual_seed(DS_GEN_SEED)
    ds_train, ds_val, ds_test = random_split(ds, (ds_train_len, ds_val_len, ds_test_len), ds_gen)

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True )
    dl_val   = DataLoader(ds_val  , batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False)
    dl_test  = DataLoader(ds_test , batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False)

    from transformers.utils import logging
    logging.set_verbosity(40)
    model = Shelgon(
        encoder_model_name=ENCODER_MODEL_NAME, 
        emb_size=EMB_SIZE, seq_len=TOKENIZED_SENTENCE_MAX_LENGTH,
        num_latent_classes=NUM_LATENT_CLASSES, num_labels_per_class=NUM_LABELS_PER_CLASS,
        decoder_model_name=DECODER_MODEL_NAME
    ).to(device)
    model.compile()
    model.set_mode(MODEL_MODE)
    model.model_params_summary_print()

    tokenizer_encoder: BertTokenizerFast = BertTokenizerFast.from_pretrained(TOKENIZER_NAME_ENCODER)
    
    if "bert" in TOKENIZER_NAME_DECODER:
        tokenizer_decoder: BertTokenizerFast = BertTokenizerFast.from_pretrained(TOKENIZER_NAME_DECODER)
    elif "gpt" in TOKENIZER_NAME_DECODER:
        tokenizer_decoder: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained(TOKENIZER_NAME_DECODER)
        tokenizer_decoder.pad_token = tokenizer_decoder.eos_token
    else:
        raise ValueError(f"{TOKENIZER_NAME_DECODER} is not a valid decoder tokenizer name.")
    

    opt = Adam(params=model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, amsgrad=AMSGRAD)
    if LR_SCHEDULER == "MultiStepLR":
        lr_sched = MultiStepLR(optimizer=opt, milestones=MILESTONES, gamma=GAMMA)
    else:
        lr_sched = None

    console = Console()
    prg = Progress(
        SpinnerColumn(spinner_name="monkey"),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("[bold][progress.percentage]{task.percentage:>3.2f}%"),
        BarColumn(finished_style=Style(color="#008000")),
        MofNCompleteColumn(),
        TextColumn("[bold]•"),
        TimeElapsedColumn(),
        TextColumn("[bold]•"),
        TimeRemainingColumn(),
        TextColumn("[bold #5B4328]{task.speed} it/s"),
        SpinnerColumn(spinner_name="moon"),
        console=console
    )

    run_id = datetime.now().strftime(RUN_ID_TIMESTAMP_FORMAT)
    run_path = f"{RUNS_DIR}/{run_id}"
    os.makedirs(run_path) if not os.path.exists(run_path) else None
    
    run_conf = get_config()
    run_conf.update(
        {
            "n_params": model.model_params_summary_dict(), 
            "optimizer": str(opt),
            "run_id": run_id
        }
    )
    with open(f"{run_path}/run_conf.json", 'w') as fp:
        json.dump(run_conf, fp)
    os.environ["WANDB_SILENT"] = WANDB_SILENT
    wandb_run = wandb.init(
        project=WANDB_PROJECT_NAME, group=WANDB_GROUP, job_type=WANDB_JOB_TYPE,
        config=run_conf, mode=WANDB_MODE
    )
    if WANDB_WATCH_MODEL:
        wandb_run.watch(model, log='all')
    if WANDB_LOG_CODE:
        wandb.run.log_code(".")
    
    n_batches_train = int(len(dl_train) * LIM_BATCHES_TRAIN_PCT)
    n_batches_val   = int(len(dl_val  ) * LIM_BATCHES_VAL_PCT)
    decoded_sentences = []
    train(
        prg=prg, console=console,
        device=device, 
        dl_train=dl_train, dl_val=dl_val, n_batches_train=n_batches_train, n_batches_val=n_batches_val,
        model=model, 
        tokenizer_encoder=tokenizer_encoder, tokenizer_decoder=tokenizer_decoder, 
        tokenizer_encoder_add_special_tokens=TOKENIZER_ADD_SPECIAL_TOKENS, tokenized_encoder_sentence_max_length=TOKENIZED_SENTENCE_MAX_LENGTH,
        tokenizer_decoder_add_special_tokens=TOKENIZER_ADD_SPECIAL_TOKENS, tokenized_decoder_sentence_max_length=TOKENIZED_SENTENCE_MAX_LENGTH,
        encoder_perturb_train_pct=ENCODER_PERTURB_TRAIN_PCT, encoder_perturb_val_pct=ENCODER_PERTURB_VAL_PCT,
        decoder_perturb_train_pct=DECODER_PERTURB_TRAIN_PCT, decoder_perturb_val_pct=DECODER_PERTURB_VAL_PCT,
        num_labels_per_class=NUM_LABELS_PER_CLASS,
        n_epochs_to_decode_after=N_EPOCHS_TO_DECODE_AFTER, decoded_sentences=decoded_sentences,
        opt=opt, lr_sched=lr_sched,
        n_epochs=N_EPOCHS, 
        vocab_size_encoder=VOCAB_SIZE_ENCODER, vocab_size_decoder=VOCAB_SIZE_DECODER,
        wandb_run=wandb_run, run_path=run_path
    )
    n_batches_test = int(len(dl_test) * LIM_BATCHES_TEST_PCT)
    model_best_val_checkpoint = torch.load(f"{run_path}/{MODEL_NAME}_ckpt_loss_recon_val_best.pth")
    model.load_state_dict(model_best_val_checkpoint["model_state_dict"])
    test(
        prg=prg, console=console,
        device=device,
        dl_test=dl_test, n_batches_test=n_batches_test,
        model=model, 
        tokenizer_encoder=tokenizer_encoder, tokenizer_decoder=tokenizer_decoder, 
        tokenizer_encoder_add_special_tokens=TOKENIZER_ADD_SPECIAL_TOKENS, tokenized_encoder_sentence_max_length=TOKENIZED_SENTENCE_MAX_LENGTH,
        tokenizer_decoder_add_special_tokens=TOKENIZER_ADD_SPECIAL_TOKENS, tokenized_decoder_sentence_max_length=TOKENIZED_SENTENCE_MAX_LENGTH,
        encoder_perturb_test_pct=ENCODER_PERTURB_TEST_PCT, decoder_perturb_test_pct=DECODER_PERTURB_TEST_PCT,
        num_labels_per_class=NUM_LABELS_PER_CLASS,
        decoded_sentences=decoded_sentences,
        vocab_size_encoder=VOCAB_SIZE_ENCODER, vocab_size_decoder=VOCAB_SIZE_DECODER,
        # TODO NOTE handle this in case of resuming from checkpoint!
        epoch=N_EPOCHS,
        wandb_run=wandb_run
    )
    decoded_sentences_df = pd.DataFrame(decoded_sentences)
    decoded_sentences_df.to_feather(f"{run_path}/decoded_sentences.feather")
    

    return


if __name__ == "__main__":

    main()