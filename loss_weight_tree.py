from __future__ import annotations

from typing import Callable, List, Optional, Union

import torch
from torchWork import DEVICE

class LossWeightTree:
    def __init__(
        self, name: str, 
        weight: Union[float, Callable[[int], float]], 
        children: Optional[List[LossWeightTree]], 
    ) -> None:
        self.name = name
        self.weight = weight
        self.children = children
    
    def __getitem__(self, key):
        for child in self.children:
            if child.name == key:
                return child
        raise KeyError(f'{key} not found.')
    
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
