import torch
from typing import Type

from torchWork.loss_weight_tree import LossWeightTree

class BaseHyperParams:
    def __init__(self) -> None:
        self.lossWeightTree: LossWeightTree = None
        self.OptimClass: Type[torch.optim.Optimizer] = None
        self.lr: float = None
    
    def print(self, depth=0):
        for k, v in self.__dict__.items():
            if not k.startswith('_') and not callable(v):
                print(' ' * depth, k, ': ', sep='', end=' ')
                if isinstance(v, LossWeightTree):
                    print('{')
                    v.print(depth + 1)
                    print(' ' * depth, '}', sep='')
                else:
                    print(v)
