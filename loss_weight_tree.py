from __future__ import annotations

from typing import Callable, List, Optional, Union

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
