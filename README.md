# VAR: a new visual generation method elevates GPT-style models beyond diffusion🚀 & Scaling laws observed📈

<div align="center">

[![SOTA](https://img.shields.io/badge/State%20of%20the%20Art-Image%20Generation%20on%20ImageNet%20256x256%20%28Autoregressive%29-32B1B4?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iNjA2IiBoZWlnaHQ9IjYwNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgb3ZlcmZsb3c9ImhpZGRlbiI%2BPGRlZnM%2BPGNsaXBQYXRoIGlkPSJjbGlwMCI%2BPHJlY3QgeD0iLTEiIHk9Ii0xIiB3aWR0aD0iNjA2IiBoZWlnaHQ9IjYwNiIvPjwvY2xpcFBhdGg%2BPC9kZWZzPjxnIGNsaXAtcGF0aD0idXJsKCNjbGlwMCkiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEgMSkiPjxyZWN0IHg9IjUyOSIgeT0iNjYiIHdpZHRoPSI1NiIgaGVpZ2h0PSI0NzMiIGZpbGw9IiM0NEYyRjYiLz48cmVjdCB4PSIxOSIgeT0iNjYiIHdpZHRoPSI1NyIgaGVpZ2h0PSI0NzMiIGZpbGw9IiM0NEYyRjYiLz48cmVjdCB4PSIyNzQiIHk9IjE1MSIgd2lkdGg9IjU3IiBoZWlnaHQ9IjMwMiIgZmlsbD0iIzQ0RjJGNiIvPjxyZWN0IHg9IjEwNCIgeT0iMTUxIiB3aWR0aD0iNTciIGhlaWdodD0iMzAyIiBmaWxsPSIjNDRGMkY2Ii8%2BPHJlY3QgeD0iNDQ0IiB5PSIxNTEiIHdpZHRoPSI1NyIgaGVpZ2h0PSIzMDIiIGZpbGw9IiM0NEYyRjYiLz48cmVjdCB4PSIzNTkiIHk9IjE3MCIgd2lkdGg9IjU2IiBoZWlnaHQ9IjI2NCIgZmlsbD0iIzQ0RjJGNiIvPjxyZWN0IHg9IjE4OCIgeT0iMTcwIiB3aWR0aD0iNTciIGhlaWdodD0iMjY0IiBmaWxsPSIjNDRGMkY2Ii8%2BPHJlY3QgeD0iNzYiIHk9IjY2IiB3aWR0aD0iNDciIGhlaWdodD0iNTciIGZpbGw9IiM0NEYyRjYiLz48cmVjdCB4PSI0ODIiIHk9IjY2IiB3aWR0aD0iNDciIGhlaWdodD0iNTciIGZpbGw9IiM0NEYyRjYiLz48cmVjdCB4PSI3NiIgeT0iNDgyIiB3aWR0aD0iNDciIGhlaWdodD0iNTciIGZpbGw9IiM0NEYyRjYiLz48cmVjdCB4PSI0ODIiIHk9IjQ4MiIgd2lkdGg9IjQ3IiBoZWlnaHQ9IjU3IiBmaWxsPSIjNDRGMkY2Ii8%2BPC9nPjwvc3ZnPg%3D%3D)](https://paperswithcode.com/sota/image-generation-on-imagenet-256x256?tag_filter=485&p=visual-autoregressive-modeling-scalable-image)&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv-2404.02905-b31b1b.svg)](https://arxiv.org/abs/2404.02905)

</div>

This is the official PyTorch implementation of [Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction](https://arxiv.org/abs/2404.02905).

**NOTE: Mark your calendars📅! Our code will be ready before 9:00 AM UTC on 4/4/2024.** Feel free to star ⭐ or watch 👓 for the latest updates🤗!

![abs](https://github.com/FoundationVision/VAR/assets/39692511/9850df90-20b1-4f29-8592-e3526d16d755)

<br>

## What's New?

### 🔥 Introducing VAR: a new paradigm in autoregressive visual generation✨:

Visual Autoregressive Modeling (VAR) redefines the autoregressive learning on images as coarse-to-fine "next-scale prediction" or "next-resolution prediction", diverging from the standard raster-scan "next-token prediction".

<p align="center">
<img src="https://github.com/FoundationVision/VAR/assets/39692511/3e12655c-37dc-4528-b923-ec6c4cfef178" width=95%>
<p>

### 🔥 For the first time, GPT-style autoregressive models surpass diffusion models🚀:
<p align="center">
<img src="https://github.com/FoundationVision/VAR/assets/39692511/cc30b043-fa4e-4d01-a9b1-e50650d5675d" width=60%>
<p>


### 🔥 Discovering power-law Scaling Laws in VAR transformers📈:


<p align="center">
<img src="https://github.com/FoundationVision/VAR/assets/39692511/c35fb56e-896e-4e4b-9fb9-7a1c38513804" width=88%>
<p>
<p align="center">
<img src="https://github.com/FoundationVision/VAR/assets/39692511/91d7b92c-8fc3-44d9-8fb4-73d6cdb8ec1e" width=88%>
<p>


### 🔥 Zero-shot generalizability🛠️:

<p align="center">
<img src="https://github.com/FoundationVision/VAR/assets/39692511/a54a4e52-6793-4130-bae2-9e459a08e96a" width=80%>
<p>

#### For a deep dive into our analyses, discussions, and evaluations, check out our [paper](https://arxiv.org/abs/2404.02905).


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
