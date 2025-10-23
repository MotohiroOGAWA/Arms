import torch
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset
from typing import Dict


class BsDataset(TensorDataset):
    def __init__(self, **tensors):
        assert all(tensor.size(0) == next(iter(tensors.values())).size(0) for tensor in tensors.values()), \
            "All tensors must have the same size in the first dimension"
        self.tensors: Dict[str, torch.Tensor] = tensors

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensors.items()}

    def __len__(self):
        return next(iter(self.tensors.values())).size(0)
    
    def set_index(self):
        self.tensors['index'] = torch.arange(len(self))