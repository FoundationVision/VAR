# VAR: a new visual generation method elevates GPT-style models beyond diffusion🚀 & Scaling laws observed📈

<div align="center">

[![demo platform](https://img.shields.io/badge/Play%20with%20VAR%21-VAR%20demo%20platform-lightblue)](https://var.vision/demo)&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2404.02905-b31b1b.svg)](https://arxiv.org/abs/2404.02905)&nbsp;
[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-FoundationVision/var-yellow)](https://huggingface.co/FoundationVision/var)&nbsp;
[![SOTA](https://img.shields.io/badge/State%20of%20the%20Art-Image%20Generation%20on%20ImageNet%20%28AR%29-32B1B4?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iNjA2IiBoZWlnaHQ9IjYwNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgb3ZlcmZsb3c9ImhpZGRlbiI%2BPGRlZnM%2BPGNsaXBQYXRoIGlkPSJjbGlwMCI%2BPHJlY3QgeD0iLTEiIHk9Ii0xIiB3aWR0aD0iNjA2IiBoZWlnaHQ9IjYwNiIvPjwvY2xpcFBhdGg%2BPC9kZWZzPjxnIGNsaXAtcGF0aD0idXJsKCNjbGlwMCkiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEgMSkiPjxyZWN0IHg9IjUyOSIgeT0iNjYiIHdpZHRoPSI1NiIgaGVpZ2h0PSI0NzMiIGZpbGw9IiM0NEYyRjYiLz48cmVjdCB4PSIxOSIgeT0iNjYiIHdpZHRoPSI1NyIgaGVpZ2h0PSI0NzMiIGZpbGw9IiM0NEYyRjYiLz48cmVjdCB4PSIyNzQiIHk9IjE1MSIgd2lkdGg9IjU3IiBoZWlnaHQ9IjMwMiIgZmlsbD0iIzQ0RjJGNiIvPjxyZWN0IHg9IjEwNCIgeT0iMTUxIiB3aWR0aD0iNTciIGhlaWdodD0iMzAyIiBmaWxsPSIjNDRGMkY2Ii8%2BPHJlY3QgeD0iNDQ0IiB5PSIxNTEiIHdpZHRoPSI1NyIgaGVpZ2h0PSIzMDIiIGZpbGw9IiM0NEYyRjYiLz48cmVjdCB4PSIzNTkiIHk9IjE3MCIgd2lkdGg9IjU2IiBoZWlnaHQ9IjI2NCIgZmlsbD0iIzQ0RjJGNiIvPjxyZWN0IHg9IjE4OCIgeT0iMTcwIiB3aWR0aD0iNTciIGhlaWdodD0iMjY0IiBmaWxsPSIjNDRGMkY2Ii8%2BPHJlY3QgeD0iNzYiIHk9IjY2IiB3aWR0aD0iNDciIGhlaWdodD0iNTciIGZpbGw9IiM0NEYyRjYiLz48cmVjdCB4PSI0ODIiIHk9IjY2IiB3aWR0aD0iNDciIGhlaWdodD0iNTciIGZpbGw9IiM0NEYyRjYiLz48cmVjdCB4PSI3NiIgeT0iNDgyIiB3aWR0aD0iNDciIGhlaWdodD0iNTciIGZpbGw9IiM0NEYyRjYiLz48cmVjdCB4PSI0ODIiIHk9IjQ4MiIgd2lkdGg9IjQ3IiBoZWlnaHQ9IjU3IiBmaWxsPSIjNDRGMkY2Ii8%2BPC9nPjwvc3ZnPg%3D%3D)](https://paperswithcode.com/sota/image-generation-on-imagenet-256x256?tag_filter=485&p=visual-autoregressive-modeling-scalable-image)


</div>
<p align="center" style="font-size: larger;">
  <a href="https://arxiv.org/abs/2404.02905">Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction</a>
</p>

<div>
  <p align="center" style="font-size: larger;">
    <strong>NeurIPS 2024 Oral</strong>
  </p>
</div>

<p align="center">
<img src="https://github.com/FoundationVision/VAR/assets/39692511/9850df90-20b1-4f29-8592-e3526d16d755" width=95%>
<p>

<br>

## News

* **2024-12:** We Release our Text-to-Image research based on VAR, please check [Infinity](https://arxiv.org/abs/2412.04431).
* **2024-09:** VAR is accepted as **NeurIPS 2024 Oral** Presentation.
* **2024-04:** [Visual AutoRegressive modeling](https://github.com/FoundationVision/VAR) is released.

## 🕹️ Try and Play with VAR!

We provide a [demo website](https://var.vision/demo) for you to play with VAR models and generate images interactively. Enjoy the fun of visual autoregressive modeling!

We also provide [demo_sample.ipynb](demo_sample.ipynb) for you to see more technical details about VAR.

[//]: # (<p align="center">)
[//]: # (<img src="https://user-images.githubusercontent.com/39692511/226376648-3f28a1a6-275d-4f88-8f3e-cd1219882488.png" width=50%)
[//]: # (<p>)


## What's New?

### 🔥 Introducing VAR: a new paradigm in autoregressive visual generation✨:

Visual Autoregressive Modeling (VAR) redefines the autoregressive learning on images as coarse-to-fine "next-scale prediction" or "next-resolution prediction", diverging from the standard raster-scan "next-token prediction".

<p align="center">
<img src="https://github.com/FoundationVision/VAR/assets/39692511/3e12655c-37dc-4528-b923-ec6c4cfef178" width=93%>
<p>

### 🔥 For the first time, GPT-style autoregressive models surpass diffusion models🚀:
<p align="center">
<img src="https://github.com/FoundationVision/VAR/assets/39692511/cc30b043-fa4e-4d01-a9b1-e50650d5675d" width=55%>
<p>


### 🔥 Discovering power-law Scaling Laws in VAR transformers📈:


<p align="center">
<img src="https://github.com/FoundationVision/VAR/assets/39692511/c35fb56e-896e-4e4b-9fb9-7a1c38513804" width=85%>
<p>
<p align="center">
<img src="https://github.com/FoundationVision/VAR/assets/39692511/91d7b92c-8fc3-44d9-8fb4-73d6cdb8ec1e" width=85%>
<p>


### 🔥 Zero-shot generalizability🛠️:

<p align="center">
<img src="https://github.com/FoundationVision/VAR/assets/39692511/a54a4e52-6793-4130-bae2-9e459a08e96a" width=70%>
<p>

#### For a deep dive into our analyses, discussions, and evaluations, check out our [paper](https://arxiv.org/abs/2404.02905).


## VAR zoo
We provide VAR models for you to play with, which are on <a href='https://huggingface.co/FoundationVision/var'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-FoundationVision/var-yellow'></a> or can be downloaded from the following links:

|   model    | reso. |   FID    | rel. cost | #params | HF weights🤗                                                                        |
|:----------:|:-----:|:--------:|:---------:|:-------:|:------------------------------------------------------------------------------------|
|  VAR-d16   |  256  |   3.55   |    0.4    |  310M   | [var_d16.pth](https://huggingface.co/FoundationVision/var/resolve/main/var_d16.pth) |
|  VAR-d20   |  256  |   2.95   |    0.5    |  600M   | [var_d20.pth](https://huggingface.co/FoundationVision/var/resolve/main/var_d20.pth) |
|  VAR-d24   |  256  |   2.33   |    0.6    |  1.0B   | [var_d24.pth](https://huggingface.co/FoundationVision/var/resolve/main/var_d24.pth) |
|  VAR-d30   |  256  |   1.97   |     1     |  2.0B   | [var_d30.pth](https://huggingface.co/FoundationVision/var/resolve/main/var_d30.pth) |
| VAR-d30-re |  256  | **1.80** |     1     |  2.0B   | [var_d30.pth](https://huggingface.co/FoundationVision/var/resolve/main/var_d30.pth) |

You can load these models to generate images via the codes in [demo_sample.ipynb](demo_sample.ipynb). Note: you need to download [vae_ch160v4096z32.pth](https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth) first.


## Installation

1. Install `torch>=2.0.0`.
2. Install other pip packages via `pip3 install -r requirements.txt`.
3. Prepare the [ImageNet](http://image-net.org/) dataset
    <details>
    <summary> assume the ImageNet is in `/path/to/imagenet`. It should be like this:</summary>

    ```
    /path/to/imagenet/:
        train/:
            n01440764: 
                many_images.JPEG ...
            n01443537:
                many_images.JPEG ...
        val/:
            n01440764:
                ILSVRC2012_val_00000293.JPEG ...
            n01443537:
                ILSVRC2012_val_00000236.JPEG ...
    ```
   **NOTE: The arg `--data_path=/path/to/imagenet` should be passed to the training script.**
    </details>

5. (Optional) install and compile `flash-attn` and `xformers` for faster attention computation. Our code will automatically use them if installed. See [models/basic_var.py#L15-L30](models/basic_var.py#L15-L30).


## Training Scripts

To train VAR-{d16, d20, d24, d30, d36-s} on ImageNet 256x256 or 512x512, you can run the following command:
```shell
# d16, 256x256
torchrun --nproc_per_node=8 --nnodes=... --node_rank=... --master_addr=... --master_port=... train.py \
  --depth=16 --bs=768 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1
# d20, 256x256
torchrun --nproc_per_node=8 --nnodes=... --node_rank=... --master_addr=... --master_port=... train.py \
  --depth=20 --bs=768 --ep=250 --fp16=1 --alng=1e-3 --wpe=0.1
# d24, 256x256
torchrun --nproc_per_node=8 --nnodes=... --node_rank=... --master_addr=... --master_port=... train.py \
  --depth=24 --bs=768 --ep=350 --tblr=8e-5 --fp16=1 --alng=1e-4 --wpe=0.01
# d30, 256x256
torchrun --nproc_per_node=8 --nnodes=... --node_rank=... --master_addr=... --master_port=... train.py \
  --depth=30 --bs=1024 --ep=350 --tblr=8e-5 --fp16=1 --alng=1e-5 --wpe=0.01 --twde=0.08
# d36-s, 512x512 (-s means saln=1, shared AdaLN)
torchrun --nproc_per_node=8 --nnodes=... --node_rank=... --master_addr=... --master_port=... train.py \
  --depth=36 --saln=1 --pn=512 --bs=768 --ep=350 --tblr=8e-5 --fp16=1 --alng=5e-6 --wpe=0.01 --twde=0.08
```
A folder named `local_output` will be created to save the checkpoints and logs.
You can monitor the training process by checking the logs in `local_output/log.txt` and `local_output/stdout.txt`, or using `tensorboard --logdir=local_output/`.

If your experiment is interrupted, just rerun the command, and the training will **automatically resume** from the last checkpoint in `local_output/ckpt*.pth` (see [utils/misc.py#L344-L357](utils/misc.py#L344-L357)).

## Sampling & Zero-shot Inference

For FID evaluation, use `var.autoregressive_infer_cfg(..., cfg=1.5, top_p=0.96, top_k=900, more_smooth=False)` to sample 50,000 images (50 per class) and save them as PNG (not JPEG) files in a folder. Pack them into a `.npz` file via `create_npz_from_sample_folder(sample_folder)` in [utils/misc.py#L344](utils/misc.py#L360).
Then use the [OpenAI's FID evaluation toolkit](https://github.com/openai/guided-diffusion/tree/main/evaluations) and reference ground truth npz file of [256x256](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz) or [512x512](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/VIRTUAL_imagenet512.npz) to evaluate FID, IS, precision, and recall.

Note a relatively small `cfg=1.5` is used for trade-off between image quality and diversity. You can adjust it to `cfg=5.0`, or sample with `autoregressive_infer_cfg(..., more_smooth=True)` for **better visual quality**.
We'll provide the sampling script later.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Citation
If our work assists your research, feel free to give us a star ⭐ or cite us using:
```
@Article{VAR,
      title={Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction}, 
      author={Keyu Tian and Yi Jiang and Zehuan Yuan and Bingyue Peng and Liwei Wang},
      year={2024},
      eprint={2404.02905},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
