

# `fabric` test

<img src="/compile_test/imgs/스크린샷 2023-03-19 오후 2.28.18.png" width="99%"></img>

## Why `Fabric`?
 I didn't know the existence of `fabric` before `Lightning 2.0` launched on March. `Lightning` and many people seems to look forward to experiencing `fabric`. This makes me curious of `fabric`.
 `fabric` can make things easier, and convert(?) `lightning` code into `pytorch` style. This looks interesting, convinced me that `fabric` could be important in the future of my career and decided to test. 
 

## Brief Test Info
 - Reference: 
 - DATA: [`CIFAR10`](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html)
 - TASK: `Classification`
 - Evaluation Metric: `F1-Score`
 - Environment: Colab 
    - `!python --version`: `3.9.16`
    - `Pytorch`: `2.0.0+cu118`
 - Pytorch Version: `2.0.0`
 - Lightning Version: `2.0.0`
 - `torch.compile()`
    - `mode` : `default` 
 - model: `resne180`
 - Batch_Size: `16`
 - Epochs: `10`

## Codes (Colab) [![wandb](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/wako/fabric_test?workspace=user-wako)

 [[only_torch] CIFAR10_Classification.ipynb](https://colab.research.google.com/drive/1-FhXe-IHWEaimXhM94Qx71idIcJP6uHr?usp=share_link) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-FhXe-IHWEaimXhM94Qx71idIcJP6uHr?usp=share_link)  
 
  `torch.compile()` vs `Not torch.compile()`
<img src="/compile_test/imgs/스크린샷 2023-03-19 오후 2.28.18.png" width="99%"></img>


 [[torch_with_fabric] CIFAR10_Classification_fabric.ipynb](https://colab.research.google.com/drive/14ALviftPRqwQdlHELyzhDnVtKqcSd90G?usp=share_link) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14ALviftPRqwQdlHELyzhDnVtKqcSd90G?usp=share_link)  
 
  `torch.compile()` vs `Not torch.compile()`
<img src="/compile_test/imgs/스크린샷 2023-03-19 오후 2.28.18.png" width="99%"></img>


 [[lightning] CIFAR10_Classification.ipynb](https://colab.research.google.com/drive/1cxIZ9m_8_nM2nkqLqRcAPv2kIB8wmI2J?usp=share_link) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cxIZ9m_8_nM2nkqLqRcAPv2kIB8wmI2J?usp=share_link)  
 
  `torch.compile()` vs `Not torch.compile()`
<img src="/compile_test/imgs/스크린샷 2023-03-19 오후 2.28.18.png" width="99%"></img>


 [[from_lightning_fabric] CIFAR10_Classification_fabric.ipynb](https://colab.research.google.com/drive/1lhrB6LBLUY_djSuQ66o47w3bt4h23Tsl?usp=share_link) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lhrB6LBLUY_djSuQ66o47w3bt4h23Tsl?usp=share_link)  
 
  `torch.compile()` vs `Not torch.compile()`
<img src="/compile_test/imgs/스크린샷 2023-03-19 오후 2.28.18.png" width="99%"></img>



## References
- [Training Compiled PyTorch 2.0 with PyTorch Lightning](https://lightning.ai/pages/blog/training-compiled-pytorch-2.0-with-pytorch-lightning/)
- [CONVERT PYTORCH CODE TO FABRIC](https://lightning.ai/docs/fabric/stable/fundamentals/convert.html#convert-pytorch-code-to-fabric)
- [ACCELERATE YOUR CODE WITH FABRIC](https://lightning.ai/docs/fabric/stable/fundamentals/accelerators.html#accelerate-your-code-with-fabric)
- [HOW TO STRUCTURE YOUR CODE WITH FABRIC](https://lightning.ai/docs/fabric/stable/fundamentals/code_structure.html#how-to-structure-your-code-with-fabric)
- [FABRIC IN NOTEBOOKS](https://lightning.ai/docs/fabric/stable/fundamentals/notebooks.html#fabric-in-notebooks)
- [ORGANIZE YOUR CODE](https://lightning.ai/docs/fabric/stable/guide/lightning_module.html#organize-your-code)
- [TORCHMETRICS IN PYTORCH LIGHTNING](https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html#torchmetrics-in-pytorch-lightning)
