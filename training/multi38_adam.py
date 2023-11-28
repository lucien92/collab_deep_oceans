import os
from pathlib import Path

import numpy as np
import pandas as pd
import csv

import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import Dataset
import hydra
from omegaconf import DictConfig
from malpolon.models.utils import check_loss, check_model, check_optimizer
import pytorch_lightning as pl
import torchmetrics.functional as Fmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms
import wandb
from pytorch_lightning.loggers import WandbLogger


from malpolon.data.data_module import BaseDataModule
from malpolon.models import GenericPredictionSystem
from malpolon.logging import Summary


from transforms import RGBDataTransform

class Multi38Dataset(Dataset):
    """Pytorch dataset handler for a subset of GeoLifeCLEF 2022 dataset.
    It consists in a restriction to France and to the 100 most present plant species.

    Parameters
    ----------
    root : string or pathlib.Path
        Root directory of dataset.
    subset : string, either "train", "val", "train+val" or "test"
        Use the given subset ("train+val" is the complete training data).
    transform : callable (optional)
        A function/transform that takes a list of arrays and returns a transformed version.
    target_transform : callable (optional)
        A function/transform that takes in the target and transforms it.
    """

    def __init__(
        self,
        root,
        dataset_name,
        subset,
        transform=None,
        target_transform=None,
        ignore_indices=[],
    ):
        
        print("Initializing Multi38Dataset...")  # Debug print
        print("Subset: ", subset)  # Debug print
        print("Ignore indices: ", ignore_indices)  # Debug print
        print("Root: ", root)  # Debug print
        root = Path(root)

        self.root = root
        self.subset = subset
        self.transform = transform
        self.target_transform = target_transform
        self.ignore_indices = ignore_indices
        self.dataset_name = dataset_name


        print(f"Loading data from {root / dataset_name}")  # Debug print
        df = pd.read_csv(
            root / dataset_name,
            index_col="index",
        )

        # species = df['taxon']
        # self.species_index = {x['id']:int(x['index']) for x in species}
        
        if subset != "train+val":
            ind = df.index[df["subset"] == subset]
        else:
            ind = df.index[np.isin(df["subset"], ["train", "val"])]
        df = df.loc[ind]

        taxons = [
            "Dinophysis acuminata",
            "Karenia mikimotoi",
            "Chaetoceros",
            "Dinophysis", 
            "Alexandrium minutum",
            "Pseudo-nitzschia"
        ]

        # GOOD
        self.observation_ids = df.index
        # self.targets = df[taxons].values
        # Simpler task : 
        self.targets = df['total plankton'].values

        print("Shape of target:", self.targets.shape)  # Debug print

        print(f"Dataset initialized with {len(self.observation_ids)} observations.")  # Debug print


    def __len__(self):
        """Returns the number of observations in the dataset."""
        return len(self.observation_ids)

    def __getitem__(self, index):
        observation_id = self.observation_ids[index]
        
        species = self.targets[index]

        # Might need to be changed
        patches = self.load_patch(observation_id, self.root / 'npy' / 'plankton_med-npy-norm')

        if self.transform:
            patches = self.transform(patches)

        assert not(torch.isnan(patches).any())

        # Modified for regression
        species_density = self.targets[index]
        target = torch.tensor([species_density]).float()
        # target = torch.tensor(species_density).float()

        # species_target = self.targets[index]
        # zeros = [0]*38
        
        # if species_target in self.species_index:
        #     zeros[self.species_index[species_target]] = 1
            
        # target = torch.tensor(zeros).float()

        if self.target_transform:
            target = self.target_transform(target)

        return patches, target


    def load_patch(self, observation_id, patches_path):
        """Loads the patch data associated to an observation id.

        Parameters
        ----------
        observation_id : integer / string
            Identifier of the observation.
        patches_path : string / pathlib.Path
            Path to the folder containing all the patches.

        Returns
        -------
        patches : dict containing 2d array-like objects
            Returns a dict containing the requested patches.
        """
        # print(f"Loading patch for observation ID: {observation_id}")  # Debug print
        filename = Path(patches_path) / str(observation_id)

        patches = {}

        patch25_filename = filename.with_name(filename.stem + ".npy")
        # print(f"Loading patch file from {patch25_filename}")  # Debug print
        patch25 = np.load(patch25_filename)
        

        for i in self.ignore_indices:
            patch25[...,i] = 0   

        # # Test with summary values
        # av, std = np.average(patch25, (0,1)), np.std(patch25, (0,1))
        # mi, ma = np.min(patch25, (0,1)), np.max(patch25, (0,1))

        # patch25 = np.zeros_like(patch25)
        # patch25[0,0] = av
        # patch25[0,1] = std
        # patch25[0,2] = mi
        # patch25[0,3] = ma
        # # End of test code


        if(np.isnan(patch25).any()):
            print(patch25_filename, patch25.shape,np.isnan(patch25).sum())

        patches["25"] = patch25

        return patches


class Multi38DataModule(BaseDataModule):
    r"""
    Data module for MicroGeoLifeCLEF 2022.

    Parameters
    ----------
        dataset_path: Path to dataset
        train_batch_size: Size of batch for training
        inference_batch_size: Size of batch for inference (validation, testing, prediction)
        num_workers: Number of workers to use for data loading
    """
    def __init__(
        self,
        dataset_path: str,
        dataset_name: str = None,
        train_batch_size: int = 32,
        inference_batch_size: int = 256,
        num_workers: int = 8,
        ignore_indices: list = [],
        pin_memory: bool = True,
    ):
        super().__init__(pin_memory)
        self.dataset_path = dataset_path
        self.train_batch_size = train_batch_size
        self.inference_batch_size = inference_batch_size
        self.num_workers = num_workers
        self.ignore_indices = ignore_indices
        self.dataset_name = dataset_name
        

    @property
    def train_transform(self):
        return transforms.Compose(
            [
                lambda data: RGBDataTransform()(data["25"])
            ]
        )

    @property
    def test_transform(self):
        return transforms.Compose(
            [
                lambda data: RGBDataTransform()(data["25"])
            ]
        )
    

    def get_dataset(self, split, transform, **kwargs):
        dataset = Multi38Dataset(
            self.dataset_path,
            dataset_name=self.dataset_name,
            subset=split,
            transform=transform,
            target_transform= None,
            ignore_indices=self.ignore_indices,
            **kwargs
        )
        return dataset

class MeanLogarithmicError(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("sum_log_errors", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Ensure that predictions and targets have the same shape
        preds, target = self._input_format(preds, target)
        
        # Calculate the absolute errors
        absolute_errors = torch.abs(preds - target)
        
        # Apply the max operation to ensure no log(0) occurs
        max_values = torch.maximum(absolute_errors, torch.tensor(1.0))
        
        # Calculate the log of max values using log base 10
        log_errors = torch.log10(max_values)
        
        # Update metric states
        self.sum_log_errors += torch.sum(log_errors)
        self.total += target.numel()

    def compute(self):
        # Compute the final mean logarithmic error
        return self.sum_log_errors / self.total

    def _input_format(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.ndim > target.ndim:
            target = target.view_as(preds)
        elif target.ndim > preds.ndim:
            preds = preds.view_as(target)
        
        return preds, target

class MeanLogarithmicErrorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        absolute_errors = torch.abs(y_pred - y_true)
        max_values = torch.maximum(absolute_errors, torch.tensor(1.0))
        log_errors = torch.log10(max_values)
        mean_log_error = torch.mean(log_errors)
        
        return mean_log_error



# class Multi38ClassificationSystem(GenericPredictionSystem):
class Multi38RegressionSystem(GenericPredictionSystem):
    r"""
    Basic finetuning regression system.

    Parameters
    ----------
        model: model to use
        lr: learning rate
        weight_decay: weight decay value
        momentum: value of momentum
        nesterov: if True, uses Nesterov's momentum
        metrics: dictionnary containing the metrics to compute
        binary: if True, uses binary classification loss instead of multi-class one
    """

    def __init__(
        self,
        model,
        lr: float = 1e-2,
        weight_decay: float = 0,
        momentum: float = 0.9,
        nesterov: bool = True,
        metrics = None,
        weight: torch.Tensor = None,
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov    

        model = check_model(model)

        # optimizer = torch.optim.SGD(
        #     model.parameters(),
        #     lr=self.lr,
        #     weight_decay=self.weight_decay,
        #     momentum=self.momentum,
        #     nesterov=self.nesterov,
        # )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        if metrics is None:
            metrics = {
                "mae": Fmetrics.mean_absolute_error,
                "rmse": lambda preds, target: torch.sqrt(Fmetrics.mean_squared_error(preds, target)),
            }


        # Custom loss
        custom_loss = MeanLogarithmicErrorLoss()

        # Cast the custom loss to nn.modules.loss._Loss
        loss = nn.modules.loss._Loss()
        loss.__dict__ = custom_loss.__dict__
        # loss = nn.BCELoss(weight=weight)
        # loss = nn.BCELoss()

        # MSE loss : 
        loss = nn.MSELoss()

        

        super().__init__(model, loss, optimizer, metrics)
        self.relu = nn.ReLU()  # Define ReLU here

    # Override forward to pass through a reLU layer
    def forward(self, x):
        x = self.model(x)
        x = self.relu(x)
        return x
    


# class ClassificationSystem(Multi38ClassificationSystem):
class RegressionSystem(Multi38RegressionSystem):
    def __init__(
        self,
        model: dict,
        lr: float,
        weight_decay: float,
        momentum: float,
        nesterov: bool
    ):
        # metrics = {
        #     "acc": Fmetrics.classification.binary_accuracy,
        #     "f1": Fmetrics.classification.binary_f1_score,
        #     #"cm": Fmetrics.classification.binary_confusion_matrix,
        #     "jac": Fmetrics.classification.binary_jaccard_index,
        # }

        metrics = {
            "mse": Fmetrics.regression.mean_squared_error,
            "mle": MeanLogarithmicError().to('cuda'),
        }

        # To change with provided loss
        # loss = nn.MSELoss()
        # loss = MeanLogarithmicErrorLoss()

        super().__init__(
            model,
            lr,
            weight_decay,
            momentum,
            nesterov,
            metrics
        )
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs)
        self.log('train_mse', self.metrics['mse'](predictions, targets))
        self.log('train_mle', self.metrics['mle'](predictions, targets))
        # print(f"Training Step - Batch {batch_idx}: Predictions: {predictions}, Targets: {targets}")
        # Check if the prediction are all positive (true or false) : 
        # print(f"Training Step - Batch {batch_idx}, positive predictions: {torch.sum(predictions >= 0)}, negative predictions: {torch.sum(predictions < 0)}")
        # Check if some prediction are > 1 : 
        print(f"Training Step - Batch {batch_idx}, > 1 predictions: {torch.sum(predictions > 1)}")

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs)
        self.log('val_mse', self.metrics['mse'](predictions, targets))
        self.log('val_mle', self.metrics['mle'](predictions, targets))
        # print(f"Validation Step - Batch {batch_idx}: Predictions: {predictions}, Targets: {targets}")
        # print(f"Validation Step - Batch {batch_idx}, positive predictions: {torch.sum(predictions >= 0)}, negative predictions: {torch.sum(predictions < 0)}")
        print(f"Validation Step - Batch {batch_idx}, > 1 predictions: {torch.sum(predictions > 1)}")


@hydra.main(version_base="1.1", config_path="conf", config_name="multi38_config")
def main(cfg: DictConfig) -> None:

    print("Setting number of threads...")  # Debug print    
    torch.set_num_threads(32)
    
    print("Determining run path...")  # Debug print
    run_path = Path.cwd()
    print(f"Run path: {run_path}")  # Debug print

    print("Setting up logger...")  # Debug print
    logger = pl.loggers.TensorBoardLogger(
        save_dir=run_path.parent, 
        name='', 
        version = run_path.stem,
        sub_dir='logs', 
        default_hp_metric = False
        )
    
    print(f"TensorBoard logs will be saved in: {logger.log_dir}")
    
    logger.log_hyperparams(cfg)

    print("Initializing data module...")  # Debug print
    # Enumerate the args **cfg.data
    print("Data module args: ", cfg.data)  # Debug print
    datamodule = Multi38DataModule(**cfg.data)
    
    if cfg.other.train_from_checkpoint:
        print("Loading model from checkpoint...")  # Debug print
        ckpt_path = cfg.other.ckpt_path + cfg.other.ckpt_name
        model = RegressionSystem.load_from_checkpoint(ckpt_path, model=cfg.model, **cfg.optimizer)
    else:
        print("Initializing model...")  # Debug print
        model = RegressionSystem(cfg.model, **cfg.optimizer)

    print("Setting up callbacks...")  # Debug print
    callbacks = [
        Summary(),
        ModelCheckpoint(
            dirpath=os.getcwd(),
            filename="checkpoint-{epoch:02d}--{val_f1:.4f}",
            monitor="val_mse",
            mode="max",
        ),
    ]

    
    wandb.init(project="deep-ocean", name = 'Adam - Target : total plankton (with RELU)')
    print("Initializing trainer...")  # Debug print

    logger = WandbLogger(name="Target : total plankton", project="deep-ocean")

    trainer = pl.Trainer(logger=logger, callbacks=callbacks, **cfg.trainer)
    print("")
    print("Starting training...")  # Debug print
    print("")
    trainer.fit(model, datamodule=datamodule)
    print("")
    print("Starting validation...")  # Debug print
    print("")
    trainer.validate(model, datamodule=datamodule)

    wandb.finish()
    


def predict(cfg: DictConfig) -> list:

    datamodule = Multi38DataModule(**cfg.data)
    model = RegressionSystem(cfg.model, **cfg.optimizer)
    trainer = pl.Trainer(**cfg.trainer)

    ckpt_path = cfg.other.ckpt_path + cfg.other.ckpt_name
    predictions = trainer.predict(model, datamodule=datamodule, ckpt_path=ckpt_path)
  
    return(nn.cat(predictions).numpy())



def test(cfg: DictConfig) -> list:

    datamodule = Multi38DataModule(**cfg.data)
    model = RegressionSystem(cfg.model, **cfg.optimizer)
    trainer = pl.Trainer(**cfg.trainer)

    ckpt_path = cfg.other.ckpt_path + cfg.other.ckpt_name
    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


def last_checkpoint() -> str:
    
    cur_path = os.getcwd()
    os.chdir('/home/gaetan/multi38/outputs/multi38')
    avail = [str(p) for p in Path('.').glob('*/*.ckpt')]
    avail.sort()
    os.chdir(cur_path)
    
    return(avail[-1])
    
    
if __name__ == "__main__":
    main()
