# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from typing import List
import os
from deep_image_diffusion_prior import predict

import torch
from deep_image_prior.models import *
from deep_image_prior.utils.sr_utils import *
from util import *
from tqdm import tqdm, trange

from typing import List
from cog import BasePredictor, Input, Path
import torch

CLIP_CHOICE = os.environ.get("CLIP_CHOICE", "ViT-L/14")
INPUT_DEPTH = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OPTIMIZER_TYPE = "MADGRAD"

def load_prior(model_path):
    """
    Loads the prior model and returns it.
    **Note** - this is a modified version of the original function to allow for the use of slim fp16 checkpoints.
    """
    prior_network = DiffusionPriorNetwork(
        dim=768,
        depth=24,
        dim_head=64,
        heads=32,
        normformer=True,
        attn_dropout=5e-2,
        ff_dropout=5e-2,
        num_time_embeds=1,
        num_image_embeds=1,
        num_text_embeds=1,
        num_timesteps=1000,
        ff_mult=4,
    )

    diffusion_prior = DiffusionPrior(
        net=prior_network,
        clip=OpenAIClipAdapter("ViT-L/14"),
        image_embed_dim=768,
        timesteps=1000,
        cond_drop_prob=0.1,
        loss_type="l2",
        condition_on_text_encodings=True,
    )
    state_dict = torch.load(model_path, map_location="cpu")
    diffusion_prior.load_state_dict(state_dict, strict=True)
    diffusion_prior.eval()
    diffusion_prior.to(DEVICE)
    return diffusion_prior

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # prior_model_path = "prior_L.pth" if CLIP_CHOICE == "ViT-L/14" else "prior_B.pth"
        prior_model_path = "prior_L_fp16.pth"
        # self.prior = ( load_diffusion_model(dprior_path=prior_model_path, clip_choice=CLIP_CHOICE) .to(DEVICE) .eval() .requires_grad_(False))
        self.prior = load_prior(prior_model_path).to(DEVICE).eval().requires_grad_(False)
        print("loaded model!")
        self.clip_model = self.prior.clip.clip
        self.clip_size = self.clip_model.visual.input_resolution

        
    def predict(
        self,
        prompt: str = Input(description="Prompt to generate", default=""),
        offset_type: str = Input(
            description="Offset type",
            default="none",
            choices=["none", "random", "random_offset", "random_offset_scale"],
        ),
        num_scales: int = Input(
            description="Number of scales", ge=1, le=10, default=6
        ),
        size_x: int = Input(
            description="Resolution of input image",
            default=256,
            choices=[64, 128, 256, 512],
        ),
        size_y: int = Input(
            description="Resolution of input image",
            default=256,
            choices=[64, 128, 256, 512],
        ),
        input_noise_strength: float = Input(
            description="Strength of input noise", ge=0, le=1, default=0.0
        ),
        lr: float = Input(
            description="Learning rate", ge=0, le=10, default=1e-3
        ),
        offset_lr_fac: float = Input(
            description="Learning rate factor for offset", ge=0, le=10, default=1.0
        ),
        lr_decay: float = Input(
            description="Learning rate decay", ge=0, le=1, default=0.995
        ),
        param_noise_strength: float = Input(
            description="Strength of parameter noise", ge=0, le=1, default=0.0
        ),
        display_freq: int = Input(
            description="Display frequency", ge=0, le=100, default=25
        ),
        iterations: int = Input(
            description="Number of iterations", ge=0, le=1000, default=250
        ),
        num_samples_per_batch: int = Input(
            description="Number of samples per batch", ge=1, le=10, default=2
        ),
        num_cutouts: int = Input(
            description="Number of cutouts", ge=8, le=64, default=16
        ),
        cond_scale: float = Input(
            description="Scale of conditioning", ge=0, le=10, default=1.0
        ),
        seed: int = Input(
            description="Random seed", ge=-1, le=100000, default=-1
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        make_cutouts = MakeCutouts(
            self.clip_size,
            num_cutouts,
        )
        for image in tqdm(predict(
            self.prior,
            self.clip_model,
            make_cutouts,
            prompt=prompt,
            offset_type=offset_type,
            num_scales=num_scales,
            size=(size_x, size_y),
            input_noise_strength=input_noise_strength,
            lr=lr,
            offset_lr_fac=offset_lr_fac,
            lr_decay=lr_decay,
            param_noise_strength=param_noise_strength,
            display_freq=display_freq,
            iterations=iterations,
            num_samples_per_batch=num_samples_per_batch,
            cond_scale=cond_scale,
            seed=seed,
        )):
            yield Path(image)