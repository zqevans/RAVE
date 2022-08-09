import accelerate
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from rave.model_accel import RAVE, Profiler
from rave.core import random_phase_mangle, EMAModelCheckPoint
from rave.core import search_for_run

from udls import SimpleDataset, simple_audio_preprocess
from effortless_config import Config
from os import environ, path
import numpy as np
from torch import multiprocessing as mp
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

        BATCH = 8

        NAME = None
        SEED = 42
        START_METHOD="forkserver"

        VAL_EVERY = 10000
        CHECKPOINT_EVERY = 25000

    args.parse_args()

    torch.manual_seed(args.SEED)

    try:
        mp.set_start_method(args.START_METHOD)
    except RuntimeError:
        pass

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print('Using device:', device, flush=True)

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
                 device=accelerator.device)

    gen_opt, dis_opt = model.configure_optimizers()

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

    train = DataLoader(dataset, args.BATCH, True, drop_last=True, num_workers=8)
    
    model, gen_opt, dis_opt, train = accelerator.prepare(model, gen_opt, dis_opt, train)

    if accelerator.is_main_process:
        x = torch.zeros(args.BATCH, 2**14).to(device)
        accelerator.unwrap_model(model).validation_step(x)

    use_wandb = accelerator.is_main_process and args.NAME
    if use_wandb:
        import wandb
        config = dict(args)
        #config['params'] = utils.n_params(model)
        wandb.init(project=args.NAME, config=config, save_code=True)

    if use_wandb:
        wandb.watch(model)

    step = 0
    epoch = 0

    try:
        while step < args.MAX_STEPS:
            for batch in tqdm(train, disable=not accelerator.is_main_process):
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

                    if step % args.VAL_EVERY == 0:
                        model_unwrap.validation_epoch_end([output])
                        p.tick("Demo")

                # if step > 0 and step % args.CHECKPOINT_EVERY == 0:
                #     save()
                #print(p)

                step += 1
            epoch += 1
    except RuntimeError as err:
            import requests
            import datetime
            ts = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            resp = requests.get('http://169.254.169.254/latest/meta-data/instance-id')
            print(f'ERROR at {ts} on {resp.text} {device}: {type(err).__name__}: {err}', flush=True)
            raise err
    except KeyboardInterrupt:
        pass