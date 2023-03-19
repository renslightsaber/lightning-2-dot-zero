# [Lightning⚡] How to train in CLI? [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HuqztVxmbk5MnL7sSVMoOAyGSB6v8LbE?usp=sharing) 


#### Check Jupyter Notebook Version at [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WQHt9KhTavEHC6CDkYSuBvoe8zgr4bk2?usp=sharing) 
#### Check wandb at [![wandb](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/wako/Lightning_2_dot_zero?workspace=user-wako)
 : Just grouped by `compiled` or `Not compiled`

<img src="/imgs/스크린샷 2023-03-16 오전 8.15.17.png" width="85%"></img>

 
## [wandb login in CLI interface](https://docs.wandb.ai/ref/cli/wandb-login)
```python
$ wandb login --relogin '######### your API token ###########'                  
``` 

## Train 
```bash
$ python train.py --project_name 'Lightning_2_dot_zero' \
                  --is_compiled 'compiled' \
                  --model_name 'resnet152' \
                  --strategy 'auto' \
                  --n_epochs 7 \
                  --bs 32
```

- `project_name`: Project Name for `wandb`
- `model_name`: `timm`'s pretrained model 
- `is_compiled`: `torch.compile()` or not; `compiled` or `Not compiled`
- `mode`: three options for `torch.compile()`; `default`, `reduce-overhead`, and `max-autotune`.
- `strategy` : [Trainer strategy](https://lightning.ai/docs/pytorch/stable/extensions/strategy.html) (Default: `auto`)
- `n_epochs` : Epochs
- `bs` : Batch Size (Default: 32)
- [`train.py`](https://github.com/renslightsaber/lightning-2-dot-zero/blob/main/compile_test/train.py) 참고!   







