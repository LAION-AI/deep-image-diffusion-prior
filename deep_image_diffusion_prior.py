from functools import lru_cache
from typing import List
import os
import gc
import math
import random

import clip
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from dalle2_pytorch import (
    DiffusionPrior,
    OpenAIClipAdapter,
)
from deep_image_prior.models import *
from deep_image_prior.utils.sr_utils import *
from util import *
from madgrad import MADGRAD
from torch import optim
from tqdm import tqdm, trange

normalize = T.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
)

CLIP_CHOICE = os.environ.get("CLIP_CHOICE", "ViT-L/14")
INPUT_DEPTH = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OPTIMIZER_TYPE = "MADGRAD"

# prior: DiffusionPrior,
# return prior.sample(
#     tokenized_text,
#     num_samples_per_batch=num_samples_per_batch,
#     cond_scale=cond_scale,
# )
# prompt: str = "",

def predict(
    clip_model: OpenAIClipAdapter,
    make_cutouts: MakeCutouts,
    target_embed: torch.Tensor = None,
    offset_type: str = "none",
    num_scales: int = 6,
    size: List[int] = [256, 256],
    input_noise_strength: float = 0.0,
    lr: float = 1e-3,
    offset_lr_fac: float = 1.0,
    lr_decay: float = 0.995,
    param_noise_strength: float = 0.0,
    display_freq: int = 25,
    iterations: int = 250,
    num_samples_per_batch: int = 2,
    cond_scale: float = 1.0,
    seed: int = -1,
) -> "List[str]":
    print("Using device:", DEVICE)

    dip_net = load_dip(
        input_depth=INPUT_DEPTH,
        num_scales=num_scales,
        offset_type=offset_type,
        device=DEVICE,
    )
    sideX, sideY = size  # Resolution

    # Seed
    if seed == -1:
        seed = random.randint(0, 100000) * random.randint(0, 1000)
    print("Seed:", seed)
    torch.manual_seed(seed)

    # Constants
    input_scale = 0.1
    net_input = torch.randn([1, INPUT_DEPTH, sideY, sideX], device=DEVICE)
    noise = torch.randn((1, 512))
    # t = torch.linspace(1, 0, 1000 + 1)[:-1]

    prompts = [Prompt(target_embed)]

    params = [
        {"params": get_non_offset_params(dip_net), "lr": lr},
        {"params": get_offset_params(dip_net), "lr": lr * offset_lr_fac},
    ]

    if OPTIMIZER_TYPE == "Adam":
        opt = optim.Adam(params, lr)
    elif OPTIMIZER_TYPE == "MADGRAD":
        opt = MADGRAD(params, lr, momentum=0.9)
    scaler = torch.cuda.amp.GradScaler()
    image = None
    try:
        itt = 0
        for _ in trange(iterations, leave=True, position=0):
            opt.zero_grad(set_to_none=True)

            noise_ramp = 1 - min(1, itt / iterations)
            net_input_noised = net_input
            if input_noise_strength:
                phi = min(1, noise_ramp * input_noise_strength) * math.pi / 2
                noise = torch.randn_like(net_input)
                net_input_noised = net_input * math.cos(phi) + noise * math.sin(phi)

            with torch.cuda.amp.autocast():
                out = dip_net(net_input_noised * input_scale).float()

            losses = []
            # for i, clip_model in enumerate(clip_models):
            cutouts = normalize(make_cutouts(out))
            with torch.cuda.amp.autocast(False):
                image_embeds = clip_model.encode_image(cutouts).float()
            for prompt in prompts:
                losses.append(prompt(image_embeds))  # * clip_model.weight)

            loss = sum(losses, out.new_zeros([]))

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if param_noise_strength:
                with torch.no_grad():
                    noise_ramp = 1 - min(1, itt / iterations)
                    for group in opt.param_groups:
                        for param in group["params"]:
                            param += (
                                torch.randn_like(param)
                                * group["lr"]
                                * param_noise_strength
                                * noise_ramp
                            )

            itt += 1
            import datetime
            output_folder_w_timestamp = os.path.join(
                "output",
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            )
            if not os.path.exists(output_folder_w_timestamp):
                os.makedirs(output_folder_w_timestamp)

            if itt % display_freq == 0:
                with torch.inference_mode():
                    image = TF.to_pil_image(out[0].clamp(0, 1))
                    if itt % display_freq == 0:
                        losses_str = ", ".join([f"{loss.item():g}" for loss in losses])
                        tqdm.write(
                            f"i: {itt}, loss: {loss.item():g}, losses: {losses_str}"
                        )
                        current_image_output_path = os.path.join(
                            output_folder_w_timestamp, f"out_{itt:05}.png"
                        )
                        image.save(current_image_output_path)
                        tqdm.write(f"Saved image to {current_image_output_path}")
                        # display(image, display_id=1)
                        yield current_image_output_path
            for group in opt.param_groups:
                group["lr"] = lr_decay * group["lr"]

    except KeyboardInterrupt:
        torch.cuda.empty_cache()
        gc.collect()
        pass


def inference(cutn=16):
    dpiror = "prior_L.pth" if CLIP_CHOICE == "ViT-L/14" else "prior_B.pth"
    prior = (
        load_diffusion_model(dprior_path=dpiror, clip_choice=CLIP_CHOICE)
        .to(DEVICE)
        .eval()
        .requires_grad_(False)
    )
    print("loaded model!")
    clip_model = prior.diffusion_prior.clip.clip
    clip_size = clip_model.visual.input_resolution

    make_cutouts = MakeCutouts(
        clip_size,
        cutn,
    )

    for image in predict(
        prior,
        clip_model,
        make_cutouts,
        prompt="",
        offset_type="none",
        num_scales=6,
        size=[256, 256],
        input_noise_strength=0.0,
        lr=1e-3,
        offset_lr_fac=1.0,
        lr_decay=0.995,
        param_noise_strength=0.0,
        display_freq=25,
        iterations=250,
        num_samples_per_batch=2,
        cond_scale=1.0,
        seed=-1,
    ):
        pass

if __name__ == "__main__":
    inference()
