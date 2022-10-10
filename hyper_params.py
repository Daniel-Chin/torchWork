import torch
from typing import Type

from torchWork.loss_weight_tree import LossWeightTree

class BaseHyperParams:
    def __init__(self) -> None:
        self.lossWeightTree: LossWeightTree = None
        self.OptimClass: Type[torch.optim.Optimizer] = None
        self.lr: float = None
