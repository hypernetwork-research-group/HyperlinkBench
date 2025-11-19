from abc import ABC
import torch
from torch import Tensor

class HyperlinkPredictionResult(ABC):

    def __init__(self, 
                 edge_index: Tensor, 
                 device="cpu"):
        self.device = device
        self.__edge_index = edge_index.to(device)

        _, self.__edge_index[1] = torch.unique(self.__edge_index[1], return_inverse=True)

    @property
    def edge_index(self) -> Tensor:
        return self.__edge_index

    @property
    def num_edges(self):
        return torch.unique(self.__edge_index[1]).shape[0]

    @property
    def y(self) -> Tensor:
        return torch.ones((self.edge_index.size(1), 1), device=self.device)

    def __repr__(self):
        return self.edge_index.__repr__()
