import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self, input_dim:int, hidden_dims:list[int], output_dim:int, dropout_rate:float=0.5):
        super(SiameseNetwork, self).__init__()
        constructor_params = inspect.signature(self.__init__).parameters
        config_keys = [param for param in constructor_params if param != "self"]
        for key in config_keys:
            setattr(self, key, locals()[key])

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.fc = nn.Sequential(*layers)
    
    def forward_once(self, x):
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2
    
    def get_config_param(self):
        """
        Get model configuration automatically.

        Returns:
            dict: Model configuration.
        """
        # Get all constructor parameter names dynamically
        constructor_params = inspect.signature(self.__init__).parameters
        ignore_config_keys = ['self']
        config_keys = [param for param in constructor_params if param not in ignore_config_keys]

        # Extract only the required parameters from instance attributes
        config = {key: getattr(self, key) for key in config_keys if hasattr(self, key)}
        
        return config

    @staticmethod
    def from_config_param(config_param):
        """
        Create model from configuration parameters.

        Args:
            config_param (dict): Model configuration parameters.

        Returns:
            Ms2z: Model instance.
        """
        return SiameseNetwork(**config_param)
    
# class ContrastiveLoss(nn.Module):
#     def __init__(self, margin=1.0):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin

#     def forward(self, output1, output2, label):
#         distance = F.pairwise_distance(output1, output2, keepdim=True)
#         loss = (1 - label) * torch.pow(distance, 2) + label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
#         return loss.mean()
    
def tanimoto_similarity(x1:torch.Tensor, x2:torch.Tensor, eps=1e-7):
    dot_product = (x1 * x2).sum(dim=1)
    norm_x1 = (x1 ** 2).sum(dim=1)
    norm_x2 = (x2 ** 2).sum(dim=1)
    denominator = norm_x1 + norm_x2 - (x1 * x2).sum(dim=1) + eps

    return dot_product / denominator

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        """
        Simple Contrastive Loss function.

        Parameters:
        - margin: The minimum distance for dissimilar pairs.
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """
        Compute contrastive loss.

        Parameters:
        - output1: Feature vector of first input
        - output2: Feature vector of second input
        - label: 1 if similar (same class), 0 if dissimilar (different class)

        Returns:
        - Contrastive loss value
        """
        # ユークリッド距離を計算
        euclidean_distance = F.pairwise_distance(output1, output2)

        # Contrastive Loss の計算
        loss = (label * euclidean_distance.pow(2) +  # 同じクラスなら距離を小さく
                (1 - label) * F.relu(self.margin - euclidean_distance).pow(2))  # 異なるクラスなら margin 以上に

        return loss.mean()


class CenterDistanceLoss(nn.Module):
    def __init__(self, close_threshold=2.0, far_threshold=5.0):
        """
        Loss function based on distance from the class center.

        Parameters:
        - close_threshold: If a point is within this distance from the center, its loss is 0.
        - far_threshold: If a point is beyond this distance from the center, its loss is 0.
        """
        super(CenterDistanceLoss, self).__init__()
        self.close_threshold = close_threshold
        self.far_threshold = far_threshold

    def forward(self, output, label):
        """
        Compute loss based on distance from the class center.

        Parameters:
        - output: Feature vector of a data point
        - label: Value between 0 (should be far) and 1 (should be close)

        Returns:
        - Loss value
        """
        # **中心との距離を計算**
        distance = torch.norm(output, dim=1)

        # **ラベル 1（中心に近づくべき） → close_threshold 未満ならロス 0**
        close_loss = F.relu(distance - self.close_threshold).pow(2)

        # **ラベル 0（中心から離れるべき） → far_threshold 以上ならロス 0**
        far_loss = F.relu(self.far_threshold - distance).pow(2)

        # **ラベルを考慮して重みを適用**
        loss = label * close_loss + (1 - label) * far_loss

        return loss.mean()
