#! /usr/bin/env python3

import accelerate
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from rave.model_accel import RAVE, Profiler
from rave.core import random_phase_mangle, EMAModelCheckPoint
from rave.core import search_for_run

#from udls import SimpleDataset, simple_audio_preprocess
#from effortless_config import Config
#from rave.audiodata import AudioDataset
from aeiou.datasets import AudioDataset
from prefigure.prefigure import get_all_args, push_wandb_config

from os import environ, path
import numpy as np
from torch import multiprocessing as mp
import GPUtil as gpu

from udls.transforms import Compose, RandomApply, Dequantize, RandomCrop
import wandb

if __name__ == "__main__":


    args = get_all_args()

    torch.manual_seed(args.seed)

    try:
        mp.set_start_method(args.start_method)
    except RuntimeError:
        pass

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print('Using device:', device, flush=True)

    # special parsing for arg lists (TODO: could add this functionality to prefigure later):
    args.ratios = eval(''.join(args.ratios))
    args.transforms = eval(args.transforms)
    print("args = ",args)
    assert args.name is not None

    model = RAVE(data_size=args.data_size,
                 capacity=args.capacity,
                 latent_size=args.latent_size,
                 ratios=args.ratios,
                 bias=args.bias,
                 loud_stride=args.loud_stride,
                 use_noise=args.use_noise,
                 noise_ratios=args.noise_ratios,
                 noise_bands=args.noise_bands,
                 d_capacity=args.d_capacity,
                 d_multiplier=args.d_multiplier,
                 d_n_layers=args.d_n_layers,
                 warmup=args.warmup,
                 mode=args.mode,
                 no_latency=args.no_latency,
                 sr=args.sr,
                 min_kl=args.min_kl,
                 max_kl=args.max_kl,
                 cropped_latent_size=args.cropped_latent_size,
                 feature_match=args.feature_match,
                 device=accelerator.device)

    gen_opt, dis_opt = model.configure_optimizers()

    if True: # new aeiou dataset class
        dataset = AudioDataset(args.wav, sample_size=args.n_signal, sample_rate=args.sr, augs=args.augs, load_frac=args.load_frac)
    else:  # antoine's old class that called preprocessing for you.
        dataset = SimpleDataset(
            args.preprocessed,
            args.wav,
            extension="*.wav,*.aif,*.flac",
            preprocess_function=simple_audio_preprocess(args.sr, 2 * args.n_signal),
            split_set="full",
            transforms=Compose( args.transforms ),
        ) 

    train = DataLoader(dataset, args.batch, True, drop_last=True, num_workers=8)
    
    model, gen_opt, dis_opt, train = accelerator.prepare(model, gen_opt, dis_opt, train)

    if accelerator.is_main_process:
        x = torch.zeros(args.batch, 2**14).to(device)
        accelerator.unwrap_model(model).validation_step(x)

    use_wandb = accelerator.is_main_process and args.name
    if use_wandb:
        import wandb
        config = vars(args) # dict(args)
        #config['params'] = utils.n_params(model)
        wandb.init(project=args.name, config=config, save_code=True)

    if use_wandb:
        wandb.watch(model)

    step = 0
    epoch = 0

    try:
        while step < args.max_steps:
            for batch in tqdm(train, disable=not accelerator.is_main_process):
                #print(f"\nbatch.shape = {batch.shape}",flush=True)
                #print(f"batch = {batch}",flush=True)
                p = Profiler()
                model_unwrap = accelerator.unwrap_model(model)
                p.tick("unwrap model")
                loss_gen, loss_dis, log_dict = model_unwrap.training_step(batch, step)
                p.tick("training step")
                #sched.step()

                # OPTIMIZATION
                if step % 2 and model_unwrap.warmed_up:
                    dis_opt.zero_grad()
                    loss_dis.backward()
                    dis_opt.step()
                    p.tick("dis_opt step")
                else:
                    gen_opt.zero_grad()
                    loss_gen.backward()
                    gen_opt.step()
                    p.tick("gen_opt step")

                if accelerator.is_main_process:
                    if step % 500 == 0:
                        tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss_gen.item():g}')

                    if use_wandb and step % 50 == 0:
                        log_dict['epoch'] = epoch
                       # log_dict['loss'] = loss.item(),
                       # log_dict['lr'] = sched.get_last_lr()[0]
                        
                        wandb.log(log_dict, step=step)
                        p.tick("logging")

                    
                    output = model_unwrap.validation_step(batch.detach())
                    p.tick("Validation")

                    if step % args.val_every == 0:
                        model_unwrap.validation_epoch_end([output])
                        p.tick("Demo")

                # if step > 0 and step % args.checkpoint_every == 0:
                #     save()
                #print(p)

                step += 1
            epoch += 1
    except RuntimeError as err:
            # Error reporting / detect faulty GPUs on AWS cluster
            import requests
            import datetime
            ts = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            resp = requests.get('http://169.254.169.254/latest/meta-data/instance-id')
            print(f'ERROR at {ts} on {resp.text} {device}: {type(err).__name__}: {err}', flush=True)
            raise err
    except KeyboardInterrupt:
        pass