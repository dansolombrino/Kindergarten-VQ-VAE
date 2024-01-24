from config import *

import torch

from datasets.dSentences.dSentencesDataset import dSentencesDataset

from torch.utils.data import random_split, DataLoader

from Bagon import Bagon

from transformers import BertTokenizer

from torch.optim.adam import Adam

from torch.optim.lr_scheduler import MultiStepLR

from rich.progress import *
from rich.style import Style

from rich.console import Console

import wandb

from Trainer import train, test


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = dSentencesDataset(DATASET_PATH)
    
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
    model = Bagon(
        encoder_model_name=ENCODER_MODEL_NAME, 
        decoder_model_name=DECODER_MODEL_NAME
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
    
    run_conf = get_config()
    run_conf.update(
        {
            "n_params": model.model_params_summary_dict(), 
            "optimizer": str(opt)
        }
    )
    import os
    os.environ["WANDB_SILENT"] = WANDB_SILENT
    wandb_run = wandb.init(
        project=WANDB_PROJECT_NAME, group=WANDB_GROUP, job_type=WANDB_JOB_TYPE,
        config=run_conf, mode=WANDB_MODE
    )
    if WANDB_WATCH_MODEL:
        wandb_run.watch(model, log='all')
    
    n_batches_train = int(len(dl_train) * LIM_BATCHES_TRAIN_PCT)
    n_batches_val   = int(len(dl_val  ) * LIM_BATCHES_VAL_PCT)
    train(
        prg=prg, console=console,
        device=device, 
        dl_train=dl_train, dl_val=dl_val, n_batches_train=n_batches_train, n_batches_val=n_batches_val,
        model=model, tokenizer=tokenizer,
        opt=opt, lr_sched=lr_sched,
        n_epochs=N_EPOCHS, 
        vocab_size=VOCAB_SIZE,
        wandb_run=wandb_run
    )
    n_batches_test = int(len(dl_test) * LIM_BATCHES_TEST_PCT)
    test(
        prg=prg, console=console,
        device=device,
        dl_test=dl_test, n_batches_test=n_batches_test,
        model=model, tokenizer=tokenizer,
        vocab_size=VOCAB_SIZE,
        # TODO NOTE handle this in case of resuming from checkpoint!
        epoch=N_EPOCHS - 1,
        wandb_run=wandb_run
    )
    

    return


if __name__ == "__main__":

    main()