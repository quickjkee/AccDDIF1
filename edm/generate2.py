# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import sys
import os

SOURCE_CODE_PATH = os.environ['SOURCE_CODE_PATH']
INPUT_PATH = os.environ['INPUT_PATH']

sys.path.append(f'{SOURCE_CODE_PATH}/EffDiff/ClipModel')
sys.path.append(f'{SOURCE_CODE_PATH}/EffDiff/ClipModel/clip')
sys.path.append(f'{SOURCE_CODE_PATH}/EffDiff/edm')
sys.path.append(f'{SOURCE_CODE_PATH}/EffDiff/guided-diffusion-main')
sys.path.append(f'{SOURCE_CODE_PATH}/EffDiff')
sys.path.append(f'{SOURCE_CODE_PATH}/EffDiff/consistency_models_main/')
sys.path.append(f'{SOURCE_CODE_PATH}/EffDiff/consistency_models_main/cm/')

import os
import re
import click
import tqdm
import pickle
import numpy as np
import subprocess
import torch
import PIL.Image
import dnnlib
from torch_utils import misc
from torch_utils import distributed as dist
from torch.utils.data import DistributedSampler
import torchvision.transforms as T

from consistency_models_main.cm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

#----------------------------------------------------------------------------
@torch.no_grad()
def stochastic_iterative_sampler(
    diffusion, model,
    x,
    generator,
    ts,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
):
    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        _, x0 = diffusion.denoise(model, x, t * s_in)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x
#----------------------------------------------------------------------------


#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).

def edm_sampler(
    diffusion, model, latents, class_labels=None, randn_like=torch.randn_like,
    x_init=0.0, num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    second_ord=False, correction=False,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = sigma_min
    sigma_max = sigma_max

    if correction is not False:
        w2 = np.array([1] + [0] * (num_steps))
        w1 = 1 - w2

        step_indices = torch.arange(num_steps, dtype=torch.float64).to(latents.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

        t_steps = torch.cat([torch.ones_like(t_steps[:1]) * 80.0, net.round_sigma(t_steps)])
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0
    else:
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

        w1 = [1] * num_steps
        w2 = [0] * num_steps

    # Main sampling loop.
    x_next = latents * t_steps[0]
    x0s = []
    ones = x_next.new_ones([x_next.shape[0]])
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = t_cur + gamma * t_cur
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        _, denoised = diffusion.denoise(model, x_hat, ones * t_hat)
        denoised = w1[i] * denoised + w2[i] * correction
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if second_ord and i < num_steps - 1:
            t_next = ones * t_next
            _, denoised = diffusion.denoise(model, x_next, t_next)  # net(x_next, t_next).to(torch.float64)
            denoised = w1[i] * denoised + w2[i] * correction
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        x0s.append(denoised.cpu())

    return x_next, x0s
#----------------------------------------------------------------------------

# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.
class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def prepare(rank, world_size, dataset, batch_size, pin_memory=False, num_workers=0):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory,
                                             num_workers=num_workers,
                                             drop_last=False, shuffle=False, sampler=sampler)

    return iter(dataloader)

@click.command()
@click.option('--edm_path', 'edm_path',  help='Network pickle filename', metavar='PATH|URL',                        type=str, required=True)
@click.option('--cons_path', 'cons_path',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--sigma_max', 'sigma_max',  help='Network pickle filename', metavar='PATH|URL',                      type=float, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--path', 'path',            help='Where to save the output images', metavar='DIR',                   type=str, required=False)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=32, show_default=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)

@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))

def main(edm_path, cons_path, num_steps, sigma_max, outdir, subdirs, seeds, class_idx, max_batch_size, path=None, device=torch.device('cuda'), **sampler_kwargs):
    os.makedirs(outdir, exist_ok=True)

    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # NETWORKS LOADING
    ##########################################################################

    # Consistency distillation
    cons_net, cons_diff = create_model_and_diffusion(
        attention_resolutions="32, 16, 8",
        class_cond=False,
        use_scale_shift_norm=False,
        dropout=0.0,
        image_size=256,
        num_channels=256,
        num_res_blocks=2,
        num_head_channels=64,
        resblock_updown=True,
        channel_mult="",
        learn_sigma=False,
        num_heads=4,
        num_heads_upsample=1,
        use_checkpoint=False,
        use_new_attention_order=False,
        use_fp16=True,
        weight_schedule='uniform',
        distillation=True,
    )
    cons_net.load_state_dict(torch.load(cons_path))
    cons_net.to(device)

    cons_net.convert_to_fp16()
    cons_net.eval()

    # EDM
    edm_net, edm_diff = create_model_and_diffusion(
        attention_resolutions="32, 16, 8",
        class_cond=False,
        use_scale_shift_norm=False,
        dropout=0.1,
        image_size=256,
        num_channels=256,
        num_res_blocks=2,
        num_head_channels=64,
        resblock_updown=True,
        channel_mult="",
        learn_sigma=False,
        num_heads=4,
        num_heads_upsample=1,
        use_checkpoint=False,
        use_new_attention_order=False,
        use_fp16=True,
        weight_schedule='karras',
        distillation=False,
    )
    edm_net.load_state_dict(torch.load(edm_path))
    edm_net.to(device)

    edm_net.convert_to_fp16()
    edm_net.eval()
    ##########################################################################

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    dist.print0(f'Batch size {len(rank_batches[0])}')

    all_images = []

    # GENERATION
    ##########################################################################
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, 3, 256, 256], device=device)

        # Generate images.
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}

        images_distill = stochastic_iterative_sampler(diffusion=cons_diff, model=cons_net, x=latents * 80.0,
                                                      t_max=80.0, steps=151, ts=[0, 62, 150], generator=rnd)

        latents = rnd.randn([batch_size, 3, 256, 256], device=device)
        #images, x0_images = edm_sampler(diffusion=edm_diff, model=edm_net,
        #                                sigma_max=sigma_max, correction=images_distill,
        #                                num_steps=num_steps, second_ord=True,
                                       #S_churn=40, S_min=0.05, S_max=50, S_noise=1.003,
        #                                latents=latents, randn_like=rnd.randn_like)

        # Save images.
        images = (images_distill * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        gathered_samples = [torch.zeros_like(images) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, images)
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        # Save estimates
        #for i, x0 in enumerate(x0_images):
        #    continue
        ##########################################################################

    # Saving.
    arr = np.concatenate(all_images, axis=0)
    np.savez(f'{outdir}/array', arr)

    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
