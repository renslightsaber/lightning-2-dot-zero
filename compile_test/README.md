# 'Compatibility with `torch.compile()`' test in Colab 

<img src="/compile_test/imgs/스크린샷 2023-03-19 오후 2.28.18.png" width="99%"></img>

## Brief Test Info
 - Reference: [Training Compiled PyTorch 2.0 with PyTorch Lightning](https://lightning.ai/pages/blog/training-compiled-pytorch-2.0-with-pytorch-lightning/)
 - DATA: [`CIFAR10`](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html)
 - TASK: `Classification`
 - Evaluation Metric: `Accuracy`, `F1-Score`
 - Environment: Colab
 - Pytorch Version: `2.0.0`
 - Lightning Version: `2.0.0`
 - model: `resnet50`, `resnet101`, `resnet152` from `timm` (more models will be added soon)
 - Batch_Size: `32`
 - Epochs: `7`

## How to train in CLI? 
- [pip install lightning ...](https://github.com/renslightsaber/lightning-2-dot-zero/blob/main/needs_to_install.md)
- [⚡train in cli](https://github.com/renslightsaber/lightning-2-dot-zero/blob/main/compile_test/how_to_train.md) 


## Check [![wandb](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/wako/Lightning_2_dot_zero?workspace=user-wako)

`resnet50`
<img src="/compile_test/imgs/스크린샷 2023-03-19 오후 8.32.34.png" width="99%"></img>

`resnet101`
<img src="/compile_test/imgs/스크린샷 2023-03-19 오후 8.33.12.png" width="99%"></img>

`resnet152`
<img src="/compile_test/imgs/스크린샷 2023-03-19 오후 8.32.11.png" width="99%"></img>

## References
- [Training Compiled PyTorch 2.0 with PyTorch Lightning](https://lightning.ai/pages/blog/training-compiled-pytorch-2.0-with-pytorch-lightning/)
- [WHAT IS A STRATEGY?](https://lightning.ai/docs/pytorch/stable/extensions/strategy.html#what-is-a-strategy)
- [WANDB (Lightning 2.0)](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#wandb)
- [TORCHMETRICS IN PYTORCH LIGHTNING](https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html#torchmetrics-in-pytorch-lightning)
