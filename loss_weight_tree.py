from __future__ import annotations

from typing import Callable, List, Optional, Union
from copy import deepcopy

import torch

class LossWeightTree:
    def __init__(
        self, name: str, 
        weight: Union[float, Callable[[int], float]], 
        children: Optional[List[LossWeightTree]], 
    ) -> None:
        self.name = name
        self.weight = weight
        self.children = children
    
    def __repr__(self):
        # if self.children is None:
        #     return f'{self.weight}_{self.name}'
        # else:
        #     return f'''{self.weight}x({"+".join([
        #         repr(x) for x in self.children
        #     ])})'''
        return f'<LossWeightTree {self.name} weight={repr(self.weight)}>'
    
    def __getitem__(self, key):
        for child in self.children:
            if child.name == key:
                return child
        raise KeyError(f'{key} not found.')
    
    def __setitem__(self, key, value):
        if isinstance(value, __class__):
            assert value.name == key
            for child in self.children:
                if child.name == key:
                    raise ValueError(f'{key} already exists.')
            self.children.append(value)
        else:
            raise TypeError(f'{__class__.__name__} can only setitem {__class__.__name__}.')

    def __contains__(self, key):
        return key in [x.name for x in self.children]
    
    def getWeight(self, epoch: int):
        try:
            return self.weight(epoch)
        except TypeError:
            return self.weight
    
    def to(self, device):
        if self.children is None:
            children = None
        else:
            children = [x.to(device) for x in self.children]
        return __class__(
            self.name, 
            torch.tensor(self.weight, device=device), 
            children, 
        )
    
    def print(self, depth=0):
        print(' ' * depth, self.name, ': ', self.weight, sep='')
        for child in self.children or []:
            child.print(depth + 1)

    def __deepcopy__(self, memo):
        if self.children is None:
            children = None
        else:
            children = [deepcopy(x, memo) for x in self.children]
        return __class__(self.name, self.weight, children)
    
    def errorAboutWeight(*_):
        raise TypeError('You are probably forgetting `.weight`')
    def __init_class__(cls):
        for operator in (
            '__bool__', '__eq__', '__ne__', 
            '__lt__', '__le__', 
            '__gt__', '__ge__', 
        ):
            setattr(cls, operator, cls.errorAboutWeight)

LossWeightTree.__init_class__(LossWeightTree)
