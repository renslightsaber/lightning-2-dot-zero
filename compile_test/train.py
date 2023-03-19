import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import wandb
import torchmetrics

# timm 
import timm
from timm import create_model

# import Pytorch Lightning 2.0 
import lightning as L
from lightning.pytorch.loggers import WandbLogger



################## load_data ######################
def load_data(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    num_workers = os.cpu_count()
    # batch_size = 32

    train_set = datasets.CIFAR10(
        root="~/data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers= num_workers
    )

    val_set = datasets.CIFAR10(
        root="~/data", train=False, download=True, transform=transform
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers= num_workers
    )
    print("DataLoader Completed")
    return train_loader, val_loader


######################## Lightning 2.0 Model ############################################
##### code from: https://lightning.ai/pages/blog/training-compiled-pytorch-2.0-with-pytorch-lightning/
##### just added torchmetrices modules
########################################################################################
class LitModel(L.LightningModule):
    def __init__(self, model_name):
        super().__init__()
        self.model = create_model(model_name, num_classes=10)
        self.criterion = nn.CrossEntropyLoss()

        # torchmetrics modules
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes = 10)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes = 10)

        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes = 10)
        self.valid_f1 = torchmetrics.F1Score(task="multiclass", num_classes = 10)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Accuracy 
        self.train_acc.update(logits, y)

        # F1 Score
        self.train_f1.update(logits, y)
        
        self.log(f'train/loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log(f'train/acc', self.train_acc, on_epoch=True, on_step=True, prog_bar=True)
        self.log(f'train/f1', self.train_f1, on_epoch=True, on_step=True, prog_bar=True)
        
        return {'loss': loss, "acc": self.train_acc, "f1": self.train_f1}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Accuracy 
        self.valid_acc.update(logits, y)

        # F1 Score
        self.valid_f1.update(logits, y)
        
        self.log(f'valid/loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log(f'valid/acc', self.valid_acc, on_epoch=True, on_step=True, prog_bar=True)
        self.log(f'valid/f1', self.valid_f1, on_epoch=True, on_step=True, prog_bar=True)
        
        return {'loss': loss, "acc": self.valid_acc, "f1": self.valid_f1}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)




######################## torch.compile() or not ############################################
def get_model(model_name ,
              is_compiled,
              mode,
              ):
    
    model = LitModel(model_name = model_name)

    if is_compiled == "compiled":
        print(f"model_name: {model_name}, is_compiled?: {is_compiled}, and mode is {mode}")
        return torch.compile(model, mode = mode)
    else:
        print(f"model_name: {model_name}")
        return model


############################### define() - argparse ######################################
def define():
    p = argparse.ArgumentParser()
    
    p.add_argument('--project_name', type = str, default = "lightning 2.0 with torch.compile()", help="wandb")
    p.add_argument('--model_name', type = str, default = "resnet18", help="timm's image pretrained model")
    p.add_argument('--is_compiled', type = str, default = "compiled", help="torch.compile() option: 'compiled' or 'not compiled'")
    p.add_argument('--mode', type = str, default = "default", help="torch.compile()'s mode: 'default', 'reduce-overhead', 'max-autotune' ")
    p.add_argument('--strategy', type = str, default = "auto", help="Lightning's Trainer Strategy: 'auto', 'ddp', 'fsdp', 'deepspeed', ... ")
    p.add_argument('--n_epochs', type = int, default = 15, help="Epochs")
    p.add_argument('--bs', type = int, default = 32, help="Batch Size")

    config = p.parse_args()
    return config
  
  
################################## main() ####################################################
def main(config):
    
    # Library Import 
    print("Version of Pytorch: ", torch.__version__)
    print("Version of Lightning: ", L.__version__)
    # https://lightning.ai/pages/blog/training-compiled-pytorch-2.0-with-pytorch-lightning/

    # DataLoader
    train_loader, val_loader = load_data(batch_size = config.bs)

    ## check batch(=data) shape
    data =next(iter(train_loader))
    print("Check Shape of Batch: ", data[0].shape, data[1].shape )
    
    ## get model 
    model = get_model(model_name = config.model_name, 
                      is_compiled = config.is_compiled, 
                      mode = config.mode)
    
    ## wandb_logger
    wandb_logger = WandbLogger( project= config.project_name, 
                                config = config,
                                job_type='Train',
                                group= config.is_compiled,
                                tags=['Lightning 2.0', 'torch.compile', config.is_compiled, config.mode],
                                name= f"{config.model_name}" + f"_{config.is_compiled}" + f"_{config.mode}",
                                anonymous='must')
    
    ## Trainer
    trainer = L.Trainer(accelerator = "auto", 
                        devices = -1, 
                        max_epochs= config.n_epochs,
                        logger = wandb_logger,
                        strategy = config.strategy
                        )
    
    ## train!
    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loader,)
    
    ## wandb
    wandb.finish()
    print("Train Completed")

if __name__ == '__main__':
    config = define()
    main(config)
    
    
    
    
    
