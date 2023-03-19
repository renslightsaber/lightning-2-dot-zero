# 'Compatibility with `torch.compile()`' Test in Colab 

<img src="/compile_test/imgs/스크린샷 2023-03-19 오후 2.28.18.png" width="99%"></img>

## Brief Test Info
 - Reference: [Training Compiled PyTorch 2.0 with PyTorch Lightning](https://lightning.ai/pages/blog/training-compiled-pytorch-2.0-with-pytorch-lightning/)
 - DATA : [`CIFAR10`](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html)
 - TASK: `Classification`
 - Evaluation Metric: `Accuracy`, `F1-Score`
 - Environment: Colab
 - Pytorch Version: `2.0.0`
 - Lightning Version: `2.0.0`
 - model: `resnet50`, `resnet150`
 - Batch_Size: `32`
 - Epochs: `7`

 
## How to train in CLI? 
- [pip install lightning ...](https://github.com/renslightsaber/lightning-2-dot-zero/blob/main/needs_to_install.md)
- [⚡train in cli](https://github.com/renslightsaber/Kaggle_FB2/blob/main/lightning) 
- Check [![wandb](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/wako/FB_TWO?workspace=user-wako)
