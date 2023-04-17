

# `fabric` test

<img src="/fabric_test/imgs/스크린샷 2023-04-17 오전 9.26.12.png" width="99%"></img>

## Why `Fabric`?
 I didn't know the existence of `fabric` before `Lightning 2.0` launched on March. `Lightning` and many people seems to look forward to experiencing `fabric`. This makes me curious of `fabric`.
 `fabric` can make things easier, and convert(?) `lightning` code into `torch` style. This looks interesting, convinced me that `fabric` could be important in the future of my career and decided to test. 
 
<img src="/fabric_test/imgs/스크린샷 2023-04-17 오전 10.02.12.png" width="99%"></img>

## Brief Test Info
 - Reference: 
 - DATA: [`CIFAR10`](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html)
 - TASK: `Classification`
 - Evaluation Metric: `F1-Score`
 - Environment: Colab 
    - `python`: `3.9.16`
    - `torch`: `2.0.0+cu118`
    - `lightning`: `2.0.1.post0`
 - `torch.compile()`
    - `mode` : `default` 
 - model: `resne18`
 - Batch_Size: `16`
 - Epochs: `10`

## Codes (Colab) and Wandb [![wandb](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/wako/fabric_test?workspace=user-wako)
 Codes are shared in `Colab` link and also shared as 'Jupyter Notebook(.ipynb)' file in `ipynbs` folder.

 ### [[only_torch] CIFAR10_Classification.ipynb](https://colab.research.google.com/drive/1-FhXe-IHWEaimXhM94Qx71idIcJP6uHr?usp=share_link) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-FhXe-IHWEaimXhM94Qx71idIcJP6uHr?usp=share_link)  
  Classification Code with only `torch`.
 
  `torch.compile()` vs `Not torch.compile()`
<img src="/fabric_test/imgs/스크린샷 2023-04-17 오전 9.33.51.png" width="99%"></img>
-----------

 ### [[torch_with_fabric] CIFAR10_Classification_fabric.ipynb](https://colab.research.google.com/drive/14ALviftPRqwQdlHELyzhDnVtKqcSd90G?usp=share_link) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14ALviftPRqwQdlHELyzhDnVtKqcSd90G?usp=share_link)  
 Classification Code with `torch` and `fabric`.  
 Tried:
 - `fabric = Fabric(accelerator= "auto", devices= "auto", strategy="auto")`
   `fabric.launch()`
 - `train_loader = fabric.setup_dataloaders(train_loader)`
 - `fabric.backward(loss)`

  `torch.compile()` vs `Not torch.compile()`
<img src="/fabric_test/imgs/스크린샷 2023-04-17 오전 9.36.02.png" width="99%"></img>
-----------

 ### [[lightning] CIFAR10_Classification.ipynb](https://colab.research.google.com/drive/1cxIZ9m_8_nM2nkqLqRcAPv2kIB8wmI2J?usp=share_link) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cxIZ9m_8_nM2nkqLqRcAPv2kIB8wmI2J?usp=share_link)  
 Classification Code with `lightning`. 
 - `fabric = Fabric(accelerator= "auto", devices= "auto", strategy="auto")`
   `fabric.launch()`
 - `train_loader = fabric.setup_dataloaders(train_loader)`
 - `model, optimizer = fabric.setup(model, optimizer)`
 - `fabric.backward(loss)`

  `torch.compile()` vs `Not torch.compile()`
<img src="/fabric_test/imgs/스크린샷 2023-04-17 오전 9.36.27.png" width="99%"></img>
-----------

 ### [[from_lightning_fabric] CIFAR10_Classification_fabric.ipynb](https://colab.research.google.com/drive/1lhrB6LBLUY_djSuQ66o47w3bt4h23Tsl?usp=share_link) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lhrB6LBLUY_djSuQ66o47w3bt4h23Tsl?usp=share_link)  
 Converted classification Code with `lightning` to `torch` style code.
 - `training_epoch_end` is reaplaced to `on_train_epoch_end`
 - `validation_epoch_end` is reaplaced to `on_validation_epoch_end`
 - `torchmetrics`'s `metric.compute()`  
     : Got Errors because of `metric.compute()` but the next day it was solved. (still don't know how it could be solved.) 
 - `batch = model.training_step(data, step)`
 - `outputs = model.on_train_epoch_end()`
 
  `torch.compile()` vs `Not torch.compile()`
<img src="/fabric_test/imgs/스크린샷 2023-04-17 오전 9.36.40.png" width="99%"></img>
-----------


## References
- [Training Compiled PyTorch 2.0 with PyTorch Lightning](https://lightning.ai/pages/blog/training-compiled-pytorch-2.0-with-pytorch-lightning/)
- [CONVERT PYTORCH CODE TO FABRIC](https://lightning.ai/docs/fabric/stable/fundamentals/convert.html#convert-pytorch-code-to-fabric)
- [ACCELERATE YOUR CODE WITH FABRIC](https://lightning.ai/docs/fabric/stable/fundamentals/accelerators.html#accelerate-your-code-with-fabric)
- [HOW TO STRUCTURE YOUR CODE WITH FABRIC](https://lightning.ai/docs/fabric/stable/fundamentals/code_structure.html#how-to-structure-your-code-with-fabric)
- [FABRIC IN NOTEBOOKS](https://lightning.ai/docs/fabric/stable/fundamentals/notebooks.html#fabric-in-notebooks)
- [ORGANIZE YOUR CODE](https://lightning.ai/docs/fabric/stable/guide/lightning_module.html#organize-your-code)
- [TORCHMETRICS IN PYTORCH LIGHTNING](https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html#torchmetrics-in-pytorch-lightning)
- [Remove memory-retaining epoch-end hooks](https://github.com/Lightning-AI/lightning/pull/16520)
