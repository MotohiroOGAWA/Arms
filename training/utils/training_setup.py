import os
from typing import Dict, Tuple, Union, Optional
import torch
from torch.utils.data import DataLoader, Subset, random_split
import pandas as pd
import dill
import re
import yaml
from collections import defaultdict

from arms.cores.TorchUtils.ModelBase import ModelBase
from .BsDataset import BsDataset
from .optim_sheduler.MovingWindowReduceLROnPlateau import MovingWindowReduceLROnPlateau

def get_optimizer(model: torch.nn.Module, optimizer_info: dict, is_return_scheduler: bool = False) -> Union[torch.optim.Optimizer, Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]]:
    """
    Create and return a PyTorch optimizer based on the given configuration.

    Args:
        model (torch.nn.Module): The model whose parameters will be optimized.
        optimizer_info (dict): Dictionary containing optimizer configuration.
            Must include:
                - "name" (str): Name of the optimizer. Supported options are:
                    ["sgd", "adagrad", "rmsprop", "adadelta", "adam", "adamw"].
                - "lr" (float): Learning rate.
            May include (depending on optimizer):
                - "momentum" (float): Momentum factor (used in SGD, RMSprop).
                - "eps" (float): Term added to denominator for numerical stability.
        if is_return_scheduler:
            scheduler = get_scheduler(optimizer, optimizer_info.get('scheduler', None))
            return optimizer, scheduler

    Returns:
        torch.optim.Optimizer: The initialized optimizer.

    Raises:
        SystemExit: If the optimizer name is not recognized.

    Example:
        >>> model = MyModel()
        >>> optimizer_info = {
        ...     "name": "adam",
        ...     "lr": 1e-3,
        ...     "eps": 1e-8
        ... }
        >>> optimizer = get_optimizer(model, optimizer_info)
    """
    optimizer_info = optimizer_info.copy()
    name = optimizer_info.pop('name')
    scheduler_info = optimizer_info.pop('scheduler', None)
    if name.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=optimizer_info['lr'], momentum=optimizer_info['momentum'])
    elif name.lower() == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=optimizer_info['lr'], eps=optimizer_info['eps'])
    elif name.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=optimizer_info['lr'], momentum=optimizer_info['momentum'], eps=optimizer_info['eps'])
    elif name.lower() == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=optimizer_info['lr'], eps=optimizer_info['eps'])
    elif name.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_info)
    elif name.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_info)
    else:
        print("Error: optimizer not recognized")
        quit()

    if not is_return_scheduler:
        return optimizer
    else:
        scheduler = get_scheduler(optimizer, scheduler_info)
        return optimizer, scheduler

def get_scheduler(optimizer: torch.optim.Optimizer, scheduler_info: dict) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create and return a PyTorch learning rate scheduler based on the given configuration.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer whose learning rate will be scheduled.
        scheduler_info (dict): Dictionary containing scheduler configuration.
            Must include:
                - "name" (str): Name of the scheduler. Supported options are:
                    ["steplr", "multisteplr", "exponentiallr", "cosineannealinglr", 
                     "reducelronplateau", "onecyclelr"]
            Additional keys depend on the scheduler type:
                - StepLR: step_size (int), gamma (float)
                - MultiStepLR: milestones (list of int), gamma (float)
                - ExponentialLR: gamma (float)
                - CosineAnnealingLR: T_max (int), eta_min (float, optional)
                - ReduceLROnPlateau: mode (str), factor (float), patience (int)
                - OneCycleLR: max_lr (float), epochs (int), steps_per_epoch (int)

    Returns:
        torch.optim.lr_scheduler._LRScheduler or ReduceLROnPlateau

    Raises:
        ValueError: If the scheduler name is not recognized.
    """
    if scheduler_info is None:
        return None
    if scheduler_info['name'] is None:
        return None

    scheduler_info = scheduler_info.copy()
    name = scheduler_info.pop('name').lower()
    scheduler_info['optimizer'] = optimizer

    if name == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_info.get('step_size', 10),
            gamma=scheduler_info.get('gamma', 0.1)
        )
    elif name == 'multisteplr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=scheduler_info.get('milestones', [30, 80]),
            gamma=scheduler_info.get('gamma', 0.1)
        )
    elif name == 'exponentiallr':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=scheduler_info.get('gamma', 0.95)
        )
    elif name == 'cosineannealinglr':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_info.get('T_max', 50),
            eta_min=scheduler_info.get('eta_min', 0.0)
        )
    elif name == 'reducelronplateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_info.get('mode', 'min'),
            factor=scheduler_info.get('factor', 0.1),
            patience=scheduler_info.get('patience', 10)
        )
    elif name == 'onecyclelr':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scheduler_info['max_lr'],
            epochs=scheduler_info['epochs'],
            steps_per_epoch=scheduler_info['steps_per_epoch']
        )
    elif name == 'movingwindowreducelronplateau':
        scheduler = MovingWindowReduceLROnPlateau(**scheduler_info)
    else:
        raise ValueError(f"Error: scheduler '{name}' not recognized")

    return scheduler


def get_criterion(name: str):
    name = name.lower()
    if name == 'crossentropy': # CrossEntropy
        return torch.nn.CrossEntropyLoss()
    elif name == 'bce': # Binary Cross Entropy
        return torch.nn.BCELoss()
    elif name == 'bcewithlogits': # Binary Cross Entropy with Logits
        return torch.nn.BCEWithLogitsLoss()
    elif name == 'mse': # Mean Squared Error
        return torch.nn.MSELoss()
    elif name == 'l1': # L1 Loss
        return torch.nn.L1Loss()
    elif name == 'smoothl1': # Smooth L1 Loss
        return torch.nn.SmoothL1Loss()
    elif name == 'hinge': # Hinge Loss
        return torch.nn.HingeEmbeddingLoss()
    elif name == 'kldiv': # KL Divergence
        return torch.nn.KLDivLoss()
    elif name == 'nll': # Negative Log Likelihood
        return torch.nn.NLLLoss()
    elif name == 'poissonnll': # Poisson Negative Log Likelihood
        return torch.nn.PoissonNLLLoss()
    elif name == 'cosineembedding': # Cosine Embedding
        return torch.nn.CosineEmbeddingLoss()
    elif name == 'huber': # Huber Loss
        return torch.nn.HuberLoss()
    elif name == 'multilabelmargin': # Multi Label Margin
        return torch.nn.MultiLabelMarginLoss()
    elif name == 'multilabelsoftmargin': # Multi Label Soft Margin
        return torch.nn.MultiLabelSoftMarginLoss()
    elif name == 'multimargin': # Multi Margin
        return torch.nn.MultiMarginLoss()
    elif name == 'marginranking': # Margin Ranking
        return torch.nn.MarginRankingLoss()
    elif name == 'ctc': # Connectionist Temporal Classification
        return torch.nn.CTCLoss()
    else:
        print("Error: loss function not recognized")
        quit()

def save_dataset(
        save_dir, dataset, train_loader, val_dataloader, test_dataloader, extra_data:dict=None, name=None):
    # Save the dataset to the specified save_dir
    if name is None:
        save_dir = os.path.join(save_dir, "ds")
    else:
        save_dir = os.path.join(save_dir, "ds", name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dill.dump(dataset, open(os.path.join(save_dir, 'dataset.dill'), 'wb'))
    dill.dump(train_loader, open(os.path.join(save_dir, 'train_loader.dill'), 'wb'))
    dill.dump(val_dataloader, open(os.path.join(save_dir, 'val_loader.dill'), 'wb'))
    dill.dump(test_dataloader, open(os.path.join(save_dir, 'test_loader.dill'), 'wb'))
    dill.dump(extra_data, open(os.path.join(save_dir, 'extra_data.dill'), 'wb'))
    # if extra_data is not None:
    #     for key, value in extra_data.items():
    #         dill.dump(value, open(os.path.join(save_dir, key + '.dill'), 'wb'))
    return save_dir

def load_dataset(load_dir, batch_size=None, name=None, load_dataset=True, load_train_loader=True, load_val_loader=True, load_test_loader=True, load_extra_data=True) -> Tuple[BsDataset, DataLoader, DataLoader, DataLoader, Dict]:
    # Load the dataset from the specified save_dir
    if name is None:
        load_dir = os.path.join(load_dir, "ds")
    else:
        load_dir = os.path.join(load_dir, "ds", name)

    if load_dataset:
        dataset = dill.load(open(os.path.join(load_dir, 'dataset.dill'), 'rb'))
    else:
        dataset = None

    if load_train_loader:
        train_loader = dill.load(open(os.path.join(load_dir, 'train_loader.dill'), 'rb'))
        if batch_size is not None:
            train_loader = DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=True)
    else:
        train_loader = None

    if load_val_loader:
        val_dataloader = dill.load(open(os.path.join(load_dir, 'val_loader.dill'), 'rb'))
        if batch_size is not None:
            val_dataloader = DataLoader(val_dataloader.dataset, batch_size=batch_size, shuffle=False)
    else:
        val_dataloader = None

    if load_test_loader:
        test_dataloader = dill.load(open(os.path.join(load_dir, 'test_loader.dill'), 'rb'))
        if batch_size is not None:
            test_dataloader = DataLoader(test_dataloader.dataset, batch_size=batch_size, shuffle=False)
    else:
        test_dataloader = None

    if load_extra_data:
        extra_data = dill.load(open(os.path.join(load_dir, 'extra_data.dill'), 'rb')) if os.path.exists(os.path.join(load_dir, 'extra_data.dill')) else {}
        # if extra_data_keys is not None:
        #     for key in extra_data_keys:
        #         extra_data[key] = dill.load(open(os.path.join(load_dir, key + '.dill'), 'rb'))


    return dataset, train_loader, val_dataloader, test_dataloader, extra_data

def get_ds(
        variables: Dict[str, torch.Tensor], mode: str, 
        train_size: float = 0.8, val_size: float = 0.1, test_size: float = None, 
        batch_size: int = 8, device: torch.device = torch.device('cpu'), set_index=True,
        split_indices: Optional[Dict[str, list]] = None
        ):
    """
    Function to create datasets and dataloaders for train, validation, and test modes.
    
    Args:
        variables (Dict[str, torch.Tensor]): Input dataset variables.
        mode (str): Mode of operation ('train' or 'test').
        train_size (float): Proportion of dataset for training.
        val_size (float): Proportion of dataset for validation.
        test_size (float): Proportion of dataset for testing.
        batch_size (int): Size of batches for DataLoader.
        device (torch.device): Device to move tensors.
        set_index (bool): Whether to set index for dataset.
        split_indices (dict): Optional, explicit indices for {"train", "val", "test"}.
                              If provided, overrides random_split.

    Returns:
        tuple: Returns dataset and corresponding DataLoaders based on mode.
    """
    # Initialize dataset using the provided variables
    variables = {key: tensor.to(device) for key, tensor in variables.items()}
    dataset = BsDataset(**variables)
    if set_index:
        dataset.set_index()

    if mode == 'train':
        # Calculate sizes for train, validation, and test datasets
        if test_size is None:
            test_ds_size = 0
            val_ds_size = max(int(val_size * len(dataset)), 1 if val_size > 0 else 0)
            train_ds_size = len(dataset) - val_ds_size - test_ds_size
        else:
            test_ds_size = max(int(test_size * len(dataset)), 1 if test_size > 0 else 0)
            val_ds_size = max(int((len(dataset)-test_ds_size)*val_size/(train_size+val_size)), 1 if val_size > 0 else 0)
            train_ds_size = len(dataset) - val_ds_size - test_ds_size
        assert train_ds_size > 0, "Train dataset size is too small. Please adjust the proportions."

        if split_indices is None:
            # Split the dataset into train, validation, and test datasets
            train_ds, val_ds, test_ds = random_split(dataset, [train_ds_size, val_ds_size, test_ds_size])
        else:
            # Use manually provided indices
            train_ds = Subset(dataset, split_indices.get("train", []))
            val_ds   = Subset(dataset, split_indices.get("val", []))
            test_ds  = Subset(dataset, split_indices.get("test", []))

        # Create DataLoaders for each dataset
        train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        return dataset, train_dataloader, val_dataloader, test_dataloader
    
    elif mode == 'test':
        # Only create a DataLoader for the full dataset
        test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataset, test_dataloader