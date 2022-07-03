# Deep Image Diffusion Prior (WIP)

Invert CLIP text embeds to image embeds and visualize with deep-image-prior from Katherine Crowson.


<a href="https://replicate.com/laion-ai/deep-image-diffusion-prior" target="_blank"><img src="https://img.shields.io/static/v1?label=Replicate&message=Demo&color=blue"></a>

<img src="/example.png" width="512"></img>

## Quick start (docker required)

- Install [docker](https://docs.docker.com/get-docker/)
- Install [cog](https://github.com/replicate/cog/)

The following command will download all weights and run a prediction with your inputs inside a proper docker container.

```sh
cog predict r8.im/laion-ai/deep-image-diffusion-prior \
  -i prompt=... \
  -i offset_type=... \
  -i num_scales=... \
  -i input_noise_strength=... \
  -i lr=... \
  -i offset_lr_fac=... \
  -i lr_decay=... \
  -i param_noise_strength=... \
  -i display_freq=... \
  -i iterations=... \
  -i num_samples_per_batch=... \
  -i num_cutouts=... \
  -i guidance_scale=... \
  -i seed=... 
```

## Intended use

See the world "through CLIP's eyes" by taking advantage of the `diffusion prior` as replicated by Laion to invert CLIP "ViT-L/14" text embeds to image embeds (as in unCLIP/DALLE2). After, a process known as `deep-image-prior` developed by Katherine Crowson is run to visualize the features in CLIP's weights corresponding to activations from your prompt.  

## Ethical considerations

Just to avoid any confusion, this research is a recreation of (one part of) OpenAI's DALLE2 paper. It is _not_, "DALLE2", the product/service from OpenAI you may have seen on the web. 

## Caveats and recommendations

These visualizations can be quite abstract compared to other text-2-image models. However, you can often find a sort of dream like quality due to this. Many outputs are artistically _fantastic_ because of this, but whether or not the visual matches your prompt as often is another matter.