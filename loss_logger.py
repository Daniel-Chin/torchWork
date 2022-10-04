import os
import sys
from typing import Union, List, Tuple

from indentprinter import IndentPrinter
from torchWork.loss_weight_tree import LossWeightTree
from torchWork.loss_tree import Loss

class LossLogger:
    def __init__(
        self, filename=None, print_every: int = 1, 
    ) -> None:
        if filename is None:
            self.filename = None
        else:
            self.filename = os.path.abspath(filename)
        self.print_every = print_every

    def __write(
        self, file, epoch_i, lossRoot: Loss, 
        lossWeightTree, extras, 
    ):
        def p(*a, **kw):
            print(*a, file=file, **kw)
        p('Finished epoch', epoch_i, ':',)
        with IndentPrinter(p, 2 * ' ') as p:
            self.dfs(p, lossRoot, lossWeightTree)
            if extras is not None:
                for key, value in extras:
                    p(key, '=', value)
    
    def eat(
        self, epoch_i: int, 
        lossRoot: Loss, lossWeightTree: LossWeightTree, 
        extras: List[Tuple[str, float]]=None, 
        verbose=True, 
    ):
        if self.filename is not None:
            with open(self.filename, 'a') as f:
                self.__write(
                    f, epoch_i, lossRoot, 
                    lossWeightTree, extras, 
                )
        if verbose and epoch_i % self.print_every == 0:
            self.__write(
                sys.stdout, epoch_i, lossRoot, 
                lossWeightTree, extras, 
            )
            sys.stdout.flush()

    def dfs(self, p, loss: Loss, lossWeightTree: LossWeightTree):
        p(loss.name, '=', loss.sum(lossWeightTree))
        with IndentPrinter(p, 2 * ' ') as p:
            for lossWeightNode in lossWeightTree.children:
                name = lossWeightNode.name
                lossChild: Union[
                    Loss, float, 
                ] = loss.__getattribute__(name)
                if lossWeightNode.children is None:
                    p(name, '=', lossChild)
                else:
                    self.dfs(p, lossChild, lossWeightNode)

    def clearFile(self):
        with open(self.filename, 'w'):
            pass
