import torch
from typing import Type
from copy import deepcopy

from torchWork.loss_weight_tree import LossWeightTree

class BaseHyperParams:
    def __init__(self) -> None:
        self.lossWeightTree: LossWeightTree = None
        self.OptimClass: Type[torch.optim.Optimizer] = None
        self.lr: float = None
        self.weight_decay: float = None
    
    def print(self, depth=0, exclude=None):
        exclude = exclude or []
        for k, v in self.__dict__.items():
            if (
                not k.startswith('_') and not callable(v)
                and k not in exclude
            ):
                print(' ' * depth, k, ': ', sep='', end=' ')
                if isinstance(v, LossWeightTree):
                    print('{')
                    v.print(depth + 1)
                    print(' ' * depth, '}', sep='')
                else:
                    print(v)
    
    def __deepcopy__(self, memo):
        other = self.__class__()
        for k, v in self.__dict__.items():
            do_copy, v_copy = self.copyOneParam(k, v, memo)
            if do_copy:
                other.__setattr__(k, v_copy)
        return other

    def copyOneParam(self, k: str, v, memo):
        '''
        returns `(do_copy, value_copied)`  
        '''
        if k.startswith('_') or callable(v):
            return False, None
        if k == 'OptimClass':
            return True, v
        return True, deepcopy(v, memo)
