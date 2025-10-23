import os
import torch
import torch.nn as nn
import inspect
import yaml

class ModelBase(nn.Module):
    def __init__(self, ignore_config_keys=None, **kwargs):
        super(ModelBase, self).__init__()
        self._ignore_config_keys = ['self', '_ignore_config_keys', '__class__']
        if ignore_config_keys is not None:
            self._ignore_config_keys.extend(ignore_config_keys)
        for key, value in kwargs.items():
            if key not in self._ignore_config_keys:
                setattr(self, key, value)


    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward method must be implemented by the subclass.")
    
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_params(self):
        """
        Get model configuration automatically.

        Returns:
            dict: Model configuration.
        """
        # Get all constructor parameter names dynamically
        constructor_params = inspect.signature(self.__init__).parameters
        config_keys = [param for param in constructor_params if param not in self._ignore_config_keys]

        # Extract only the required parameters from instance attributes
        config = {key: getattr(self, key) for key in config_keys if hasattr(self, key)}
        
        return config
    
    @classmethod
    def from_params(cls, config_param):
        """
        Create model from configuration parameters.

        Args:
            config_param (dict): Model configuration parameters.

        Returns:
            BaseModel: Model instance.
        """
        if config_param is None:
            config_param = {}
        return cls(**config_param)
    
    @staticmethod
    def write_config(file_path: str, config_dict: dict):
        """
        Write a config.yaml file.
        """
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f, sort_keys=False, allow_unicode=True)

        print(f"Default config written to {file_path}")

    def log_grad_norms(self):
        """
        Log the average gradient norm of all parameters in the model.
        """
        grad_norms = []
        for name, p in self.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().norm(2).item()
                grad_norms.append(param_norm)

        if len(grad_norms) == 0:
            return None

        avg_norm = sum(grad_norms) / len(grad_norms)
        return avg_norm