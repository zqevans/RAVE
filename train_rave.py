import torch
from torch.utils.data import DataLoader, random_split

from rave.model import RAVE
from rave.core import random_phase_mangle, EMAModelCheckPoint
from rave.core import search_for_run

from udls import SimpleDataset, simple_audio_preprocess
from effortless_config import Config
import pytorch_lightning as pl
from os import environ, path
import numpy as np

import GPUtil as gpu

from udls.transforms import Compose, RandomApply, Dequantize, RandomCrop
import wandb

if __name__ == "__main__":

    class args(Config):
        DATA_SIZE = 16
        CAPACITY = 32
        LATENT_SIZE = 128
        RATIOS = [4, 4, 2, 2, 2]
        TAYLOR_DEGREES = 0
        BIAS = True
        NO_LATENCY = False

        MIN_KL = 1e-4
        MAX_KL = 1e-1
        CROPPED_LATENT_SIZE = 0
        FEATURE_MATCH = True

        LOUD_STRIDE = 1

        USE_NOISE = True
        NOISE_RATIOS = [4, 4, 4]
        NOISE_BANDS = 5

        D_CAPACITY = 16
        D_MULTIPLIER = 4
        D_N_LAYERS = 4

        WARMUP = 1000000
        MODE = "hinge"
        CKPT = None

        PREPROCESSED = None
        WAV = None
        SR = 48000
        N_SIGNAL = 65536
        MAX_STEPS = 2000000

        N_GPUS = 1
        BATCH = 8

        NAME = None

    args.parse_args()

    assert args.NAME is not None
    model = RAVE(data_size=args.DATA_SIZE,
                 capacity=args.CAPACITY,
                 latent_size=args.LATENT_SIZE,
                 ratios=args.RATIOS,
                 bias=args.BIAS,
                 loud_stride=args.LOUD_STRIDE,
                 use_noise=args.USE_NOISE,
                 noise_ratios=args.NOISE_RATIOS,
                 noise_bands=args.NOISE_BANDS,
                 d_capacity=args.D_CAPACITY,
                 d_multiplier=args.D_MULTIPLIER,
                 d_n_layers=args.D_N_LAYERS,
                 warmup=args.WARMUP,
                 mode=args.MODE,
                 no_latency=args.NO_LATENCY,
                 sr=args.SR,
                 min_kl=args.MIN_KL,
                 max_kl=args.MAX_KL,
                 cropped_latent_size=args.CROPPED_LATENT_SIZE,
                 feature_match=args.FEATURE_MATCH,
                 taylor_degrees=args.TAYLOR_DEGREES)

    x = torch.zeros(args.BATCH, 2**14)
    model.validation_step(x, 0)

    dataset = SimpleDataset(
        args.PREPROCESSED,
        args.WAV,
        extension="*.wav,*.aif,*.flac",
        preprocess_function=simple_audio_preprocess(args.SR,
                                                    2 * args.N_SIGNAL),
        split_set="full",
        transforms=Compose([
            RandomCrop(args.N_SIGNAL),
            # RandomApply(
            #     lambda x: random_phase_mangle(x, 20, 2000, .99, args.SR),
            #     p=.8,
            # ),
            Dequantize(16),
            lambda x: x.astype(np.float32),
        ]),
    )

    val = (2 * len(dataset)) // 100
    train = len(dataset) - val
    train, val = random_split(
        dataset,
        [train, val],
        generator=torch.Generator().manual_seed(42),
    )

    train = DataLoader(train, args.BATCH, True, drop_last=True, num_workers=8)
    val = DataLoader(val, args.BATCH, False, num_workers=8)

    # CHECKPOINT CALLBACKS
    # validation_checkpoint = pl.callbacks.ModelCheckpoint(
    #     monitor="validation",
    #     filename="best",
    # )
    last_checkpoint = pl.callbacks.ModelCheckpoint(every_n_train_steps=100000)

    val_check = {}
    if len(train) >= 10000:
        val_check["val_check_interval"] = 10000
    else:
        nepoch = 10000 // len(train)
        val_check["check_val_every_n_epoch"] = nepoch

    wandb_logger = pl.loggers.WandbLogger(project=args.NAME)
    wandb_logger.watch(model)

    trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=args.N_GPUS,
        strategy='ddp',
        callbacks=[last_checkpoint],
        resume_from_checkpoint=search_for_run(args.CKPT),
        max_epochs=100000,
        max_steps=args.MAX_STEPS,
        **val_check,
    )
    trainer.fit(model, train, val)