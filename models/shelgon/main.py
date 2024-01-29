from config import *

import torch

from datasets.dSentences.dSentencesDataset import dSentencesDataset

from torch.utils.data import random_split, DataLoader

from Shelgon import Shelgon

from transformers import BertTokenizer

from torch.optim.adam import Adam

from torch.optim.lr_scheduler import MultiStepLR

from rich.progress import *
from rich.style import Style

from rich.console import Console

import wandb

from Trainer import train, test

from datetime import datetime

from common.consts import *

import os

import pandas as pd


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = dSentencesDataset(SENTENCES_PATH, LATENT_CLASSES_LABELS_PATH)
    
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
        vq_n_e=VQ_N_E, vq_e_dim=VQ_E_DIM, vq_beta=VQ_BETA,
        decoder_model_name=DECODER_MODEL_NAME,
        from_pretrained_bagon=FROM_PRETRAINED_BAGON
    ).to(device)
    model.compile()
    model.set_mode(MODEL_MODE)
    model.model_params_summary_print()

    tokenizer_name = TOKENIZER_NAME
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    VOCAB_SIZE = 30522

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
        tokenizer=tokenizer, tokenizer_add_special_tokens=TOKENIZER_ADD_SPECIAL_TOKENS, 
        n_epochs_to_decode_after=N_EPOCHS_TO_DECODE_AFTER, decoded_sentences=decoded_sentences,
        opt=opt, 
        loss_recon_rescale_factor=LOSS_RECON_RESCALE_FACTOR, loss_recon_weight=LOSS_RECON_WEIGHT, 
        loss_vq_rescale_factor=LOSS_VQ_RESCALE_FACTOR, loss_vq_weight=LOSS_VQ_WEIGHT, 
        lr_sched=lr_sched,
        n_epochs=N_EPOCHS, 
        vocab_size=VOCAB_SIZE,
        wandb_run=wandb_run, run_path=run_path,
        export_checkpoint=EXPORT_CHECKPOINT
    )
    
    if EXPORT_CHECKPOINT:
        # testing requires loading the best val checkpoint
        # so, if no checkpoint has been exported, no testing can be done
        # For now, it's ok this way!
        # TODO improve handling of this. 

        n_batches_test = int(len(dl_test) * LIM_BATCHES_TEST_PCT)
        model_best_val_checkpoint = torch.load(f"{run_path}/shelgon_ckpt_loss_recon_val_best.pth")
        model.load_state_dict(model_best_val_checkpoint["model_state_dict"])
        test(
            prg=prg, console=console,
            device=device,
            dl_test=dl_test, n_batches_test=n_batches_test,
            model=model, 
            loss_recon_rescale_factor=LOSS_RECON_RESCALE_FACTOR, loss_recon_weight=LOSS_RECON_WEIGHT,
            loss_vq_rescale_factor=LOSS_VQ_RESCALE_FACTOR, loss_vq_weight=LOSS_VQ_WEIGHT,
            tokenizer=tokenizer, tokenizer_add_special_tokens=TOKENIZER_ADD_SPECIAL_TOKENS,
            decoded_sentences=decoded_sentences,
            vocab_size=VOCAB_SIZE,
            # TODO NOTE handle this in case of resuming from checkpoint!
            epoch=N_EPOCHS,
            wandb_run=wandb_run
        )
    decoded_sentences_df = pd.DataFrame(decoded_sentences)
    decoded_sentences_df.to_feather(f"{run_path}/decoded_sentences.feather")
    

    return


if __name__ == "__main__":

    main()