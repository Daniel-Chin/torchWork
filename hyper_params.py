import torch
from typing import Type

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
    
    def copy(self):
        other = self.__class__()
        for k, v in self.__dict__.items():
            do_copy, v_copy = self.copyOneParam(k, v)
            if do_copy:
                other.__setattr__(k, v_copy)
        return other

    TYPES_IMMUTABLE = [
        float, int, bool, str, type(None), 
    ]
    TYPES_CALL_COPY = [LossWeightTree]
    def copyOneParam(self, k: str, v):
        '''
        returns `(do_copy, value_copied)`  
        '''
        if v is None:
            return True, None
        if k.startswith('_') or callable(v):
            return False, None
        _type = type(v)
        if _type in __class__.TYPES_IMMUTABLE:
            return True, v
        if _type in __class__.TYPES_CALL_COPY:
            return True, v.copy()
        if k == 'OptimClass':
            return True, v
        if _type in (list, tuple):
            if all([
                type(x) in __class__.TYPES_IMMUTABLE for x in v
            ]):
                if _type is list:
                    return True, v.copy()
                if _type is tuple:
                    return True, v
            print('Debug trg3t3sff4af3eh983')
            from console import console
            console({**globals(), **locals()})
        raise TypeError(
            f'Don\'t know whether/how to copy "{k}": {v}', 
        )
