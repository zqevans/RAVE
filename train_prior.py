import torch
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl

from rave.core import search_for_run

from prior.model import Model
from effortless_config import Config
from os import environ, path

from udls import SimpleDataset, simple_audio_preprocess
import numpy as np

import math

import GPUtil as gpu
import wandb


class args(Config):
    RESOLUTION = 32

    RES_SIZE = 512
    SKP_SIZE = 256
    KERNEL_SIZE = 3
    CYCLE_SIZE = 4
    N_LAYERS = 10
    PRETRAINED_VAE = None

    PREPROCESSED = None
    WAV = None
    N_SIGNAL = 65536

    BATCH = 8
    CKPT = None
    MAX_STEPS = 10000000

    N_GPUS = 1

    NAME = None


args.parse_args()
assert args.NAME is not None


def get_n_signal(a, m):
    k = a.KERNEL_SIZE
    cs = a.CYCLE_SIZE
    l = a.N_LAYERS

    rf = (k - 1) * sum(2**(np.arange(l) % cs)) + 1
    ratio = m.encode_params[-1].item()

    return 2**math.ceil(math.log2(rf * ratio))


model = Model(
    resolution=args.RESOLUTION,
    res_size=args.RES_SIZE,
    skp_size=args.SKP_SIZE,
    kernel_size=args.KERNEL_SIZE,
    cycle_size=args.CYCLE_SIZE,
    n_layers=args.N_LAYERS,
    pretrained_vae=args.PRETRAINED_VAE,
)

args.N_SIGNAL = max(args.N_SIGNAL, get_n_signal(args, model.synth))

print(f"Receptive field: {args.N_SIGNAL} samples ({args.N_SIGNAL / model.sr} seconds)")

dataset = SimpleDataset(
    args.PREPROCESSED,
    args.WAV,
    extension="*.wav,*.aif,*.flac",
    preprocess_function=simple_audio_preprocess(model.sr, args.N_SIGNAL),
    split_set="full",
    transforms=lambda x: x.reshape(1, -1),
)

val = (2 * len(dataset)) // 100
train = len(dataset) - val
train, val = random_split(dataset, [train, val])

train = DataLoader(train, args.BATCH, True, drop_last=True, num_workers=8)
val = DataLoader(val, args.BATCH, False, num_workers=8)

# CHECKPOINT CALLBACKS
validation_checkpoint = pl.callbacks.ModelCheckpoint(
    monitor="validation",
    filename="best",
)
last_checkpoint = pl.callbacks.ModelCheckpoint(filename="last")

wandb_logger = pl.loggers.WandbLogger(project=args.NAME)
#wandb_logger.watch(model)

val_check = {}
if len(train) >= 5000:
    val_check["val_check_interval"] = 5000
else:
    nepoch = 5000 // len(train)
    val_check["check_val_every_n_epoch"] = nepoch

trainer = pl.Trainer(
    logger=wandb_logger,
    gpus=args.N_GPUS,
    callbacks=[validation_checkpoint, last_checkpoint],
    resume_from_checkpoint=search_for_run(args.CKPT),
    max_epochs=100000,
    max_steps=args.MAX_STEPS,
    **val_check,
)
trainer.fit(model, train, val)
