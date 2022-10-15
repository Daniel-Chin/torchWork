from os import path
from typing import Callable, List, Optional
import itertools

from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from tqdm import tqdm

from torchWork.experiment_control import loadExperiment, getGroupPath
from torchWork.loss_logger import Decompressor, LOSS_FILE_NAME

class LossType:
    def __init__(
        self, train_or_validate: str, loss_name: str, 
    ) -> None:
        self.train_or_validate = train_or_validate
        self.loss_name = loss_name

        self.display_name = train_or_validate + '_' + loss_name
    
    def __hash__(self):
        return hash(self.display_name)
    
    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, __class__):
            return self.display_name == __o.display_name
        return False

class LossAcc:
    def __init__(self, average_over) -> None:
        self.average_over = average_over
        self.__losses = []

        self.group_acc = 0
        self.group_size = 0
        self.n_groups = 0

        self.batch_acc = 0
        self.batch_size = 0
    
    def eat(self, x, /):
        self.batch_acc += x
        self.batch_size += 1
    
    def endBatch(self, epoch_i=None):
        if self.batch_size == 0:
            return
        x = self.batch_acc / self.batch_size
        self.batch_acc = 0
        self.batch_size = 0
        self.group_acc += x
        self.group_size += 1
        if self.group_size == self.average_over:
            self.pop()

    def pop(self):
        self.__losses.append(self.group_acc / self.group_size)
        self.group_acc = 0
        self.group_size = 0
        self.n_groups += 1
    
    def getHistory(self):
        assert self.batch_acc == 0
        return self.__losses

def plotLosses(
    experiment_py_path: str, lossTypes: List[LossType], 
    average_over: int, epoch_start: int, 
    epoch_stop: Optional[int] = None, 
):
    fig, axes = plt.subplots(1, len(lossTypes))
    (
        experiment_name, n_rand_inits, groups, 
    ) = loadExperiment(experiment_py_path)
    print(f'{experiment_name = }')
    group_start = epoch_start // average_over
    if epoch_stop is None:
        group_stop = None
    else:
        group_stop = epoch_stop // average_over
    for (group_i, group), rand_init_i in tqdm([*itertools.product(
        enumerate(groups), range(n_rand_inits), 
    )]):
        lossAccs = {x: LossAcc(average_over) for x in lossTypes}
        with OnChangeOrEnd(*[
            x.endBatch for x in lossAccs.values()
        ]) as oCoE:
            for (
                epoch_i, batch_i, train_or_validate, _, entries, 
            ) in Decompressor(path.join(getGroupPath(
                path.dirname(experiment_py_path), 
                group.name(), rand_init_i, 
            ), LOSS_FILE_NAME)):
                oCoE.eat(epoch_i)
                for loss_name, value in entries.items():
                    lossType = LossType(
                        'train' if train_or_validate else 'validate', 
                        loss_name, 
                    )
                    try:
                        lossAcc = lossAccs[lossType]
                    except KeyError:
                        pass
                    else:
                        lossAcc.eat(value)
        epochs = [(i + 1) * average_over for i in range(
            next(iter(lossAccs.values())).n_groups, 
        )]
        if rand_init_i == 0:
            kw = dict(label=group.name())
        else:
            kw = dict()
        for ax, (lossType, lossAcc) in zip(
            axes, lossAccs.items(), 
        ):
            ax.plot(
                epochs[group_start:group_stop], 
                lossAcc.getHistory()[group_start:group_stop], 
                c=hsv_to_rgb((group_i / len(groups), 1, .8)), 
                **kw, 
            )
            ax.set_title(lossType.display_name)
    for ax in axes:
        ax.axhline(y=0, color='k')
        ax.set_xlabel('epoch')
    axes[-1].legend()
    fig.tight_layout()
    return fig

class OnChangeOrEnd:
    def __init__(self, *callbacks: Callable) -> None:
        self.callbacks = callbacks

        self.started = False
        self.value = None

    def __enter__(self):
        return self
    
    def callback(self, x):
        for callback in self.callbacks:
            callback(x)
    
    def eat(self, x):
        if self.started:
            if self.value != x:
                self.callback(self.value)
                self.value = x
        else:
            self.started = True
            self.value = x
    
    def __exit__(self, *_, **__):
        if self.started:
            self.callback(self.value)
        return False
