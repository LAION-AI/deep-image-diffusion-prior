import kornia.augmentation as K
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from dalle2_pytorch import (
    DiffusionPrior,
    DiffusionPriorNetwork,
    DiffusionPriorTrainer,
    OpenAIClipAdapter,
)
from deep_image_prior.models import *
from deep_image_prior.utils.sr_utils import *
from resize_right import resize
from torch import nn
from torch.nn import functional as F

# @title
class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


replace_grad = ReplaceGrad.apply


def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        # self.cut_pow = cut_pow
        self.augs = T.Compose(
            [
                K.RandomHorizontalFlip(p=0.5),
                K.RandomAffine(
                    degrees=15,
                    translate=0.1,
                    p=0.8,
                    padding_mode="border",
                    resample="bilinear",
                ),
                K.RandomPerspective(0.4, p=0.7, resample="bilinear"),
                K.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.7
                ),
                K.RandomGrayscale(p=0.15),
            ]
        )

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        long_size, short_size = max(sideX, sideY), min(sideX, sideY)
        min_size = min(short_size, self.cut_size)
        pad_x, pad_y = long_size - sideX, long_size - sideY
        input_zero_padded = F.pad(input, (pad_x, pad_x, pad_y, pad_y), "constant")
        input_mask = F.pad(
            torch.zeros_like(input), (pad_x, pad_x, pad_y, pad_y), "constant", 1.0
        )
        input_padded = input_zero_padded + input_mask * input.mean(
            dim=[2, 3], keepdim=True
        )
        cutouts = []
        for cn in range(self.cutn):
            if cn >= self.cutn - self.cutn // 4:
                size = long_size
            else:
                size = clamp(
                    int(short_size * torch.zeros([]).normal_(mean=0.8, std=0.3)),
                    min_size,
                    long_size,
                )
            # size = int(torch.rand([])**self.cut_pow * (short_size - min_size) + min_size)
            offsetx = (
                torch.randint(min(0, sideX - size), abs(sideX - size) + 1, ()) + pad_x
            )
            offsety = (
                torch.randint(min(0, sideY - size), abs(sideY - size) + 1, ()) + pad_y
            )
            cutout = input_padded[
                :, :, offsety : offsety + size, offsetx : offsetx + size
            ]
            # cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
            cutouts.append(
                resize(
                    cutout,
                    out_shape=(self.cut_size, self.cut_size),
                    by_convs=True,
                    pad_mode="reflect",
                )
            )
        return self.augs(torch.cat(cutouts))


class Prompt(nn.Module):
    def __init__(self, embed, weight=1.0, stop=float("-inf")):
        super().__init__()
        self.register_buffer("embed", embed)
        self.register_buffer("weight", torch.as_tensor(weight))
        self.register_buffer("stop", torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return (
            self.weight.abs()
            * replace_grad(dists, torch.maximum(dists, self.stop)).mean()
        )


class CaptureOutput:
    """Captures a layer's output activations using a forward hook."""

    def __init__(self, module):
        self.output = None
        self.handle = module.register_forward_hook(self)

    def __call__(self, module, input, output):
        self.output = output

    def __del__(self):
        self.handle.remove()

    def get_output(self):
        return self.output


def load_dip(input_depth, num_scales, offset_type, device):
    dip_net = get_hq_skip_net(
        input_depth,
        skip_n33d=192,
        skip_n33u=192,
        skip_n11=4,
        num_scales=num_scales,
        offset_type=offset_type,
    ).to(device)

    return dip_net


def load_diffusion_model(dprior_path, clip_choice="ViT-B/32", device="cuda"):

    clip_model = OpenAIClipAdapter(clip_choice)

    if clip_choice == "ViT-B/32":
        print("loading ViT-B/32...", end="")
        prior_network = DiffusionPriorNetwork(
            dim=512,
            depth=12,
            dim_head=64,
            heads=16,
            normformer=True,
            attn_dropout=5e-2,
            ff_dropout=5e-2,
            num_time_embeds=1,
            num_image_embeds=1,
            num_text_embeds=1,
            num_timesteps=100,
            ff_mult=4,
        )

        diffusion_prior = DiffusionPrior(
            net=prior_network,
            clip=clip_model,
            image_embed_dim=512,
            timesteps=100,
            cond_drop_prob=0.1,
            loss_type="l2",
            condition_on_text_encodings=True,
        )

    elif clip_choice == "ViT-L/14":
        print("loading ViT-L/14...hang tight...", end="")
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
            clip=clip_model,
            image_embed_dim=768,
            timesteps=1000,
            cond_drop_prob=0.1,
            loss_type="l2",
            condition_on_text_encodings=True,
        )

    # this will load the entire trainer
    # If you only want EMA weights for inference you will need to extract them yourself for now
    # (if you beat me to writing a nice function for that please make a PR on Github!)
    trainer = DiffusionPriorTrainer(
        diffusion_prior=diffusion_prior,
        lr=1.1e-4,
        wd=6.02e-2,
        max_grad_norm=0.5,
        amp=False,
        group_wd_params=True,
        use_ema=True,
        device=device,
        accelerator=None,
    )

    trainer.load(dprior_path)

    return trainer
