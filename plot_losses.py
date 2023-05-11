from os import path
from typing import Callable, List, Optional, Tuple, Dict
import itertools

from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from tqdm import tqdm

try:
    from graphic_terminal import rollText
except ImportError as e:
    module_name = str(e).split('No module named ', 1)[1].strip().strip('"\'')
    if module_name in (
        'graphic_terminal', 
    ):
        print(f'Missing module {module_name}. Please download at')
        print(f'https://github.com/Daniel-Chin/Python_Lib')
        input('Press Enter to quit...')
    raise e

from torchWork.experiment_control import loadExperiment, getTrainerPath, BaseExperimentGroup
from torchWork.loss_logger import Decompressor, LOSS_FILE_NAME

class LossType:
    def __init__(
        self, train_or_validate: str, loss_name: str, 
    ) -> None:
        self.train_or_validate = train_or_validate
        self.loss_name = loss_name

        self.display_name = self.prettify(
            train_or_validate + '_' + loss_name, 
        )
    
    def prettify(self, x: str, /):
        x = x.replace('_', ' ').replace('.', ' ')
        return '\n'.join(rollText(x, 17))
    
    def __hash__(self):
        return hash(self.display_name)
    
    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, __class__):
            return self.display_name == __o.display_name
        return False
    
    def __repr__(self):
        return f'<LossType "{self.train_or_validate}_{self.loss_name}">'

class LossAcc:
    def __init__(self, average_over) -> None:
        self.average_over = average_over
        self.__losses = []
        self.__time = []

        self.group_acc = 0
        self.group_size = 0
        self.n_groups = 0

        self.batch_acc = 0
        self.batch_size = 0
        self.next_t_to_pop = average_over

    def eat(self, x, /):
        self.batch_acc += x
        self.batch_size += 1
    
    def endBatch(self, t, is_final):
        if self.batch_size != 0:
            x = self.batch_acc / self.batch_size
            self.batch_acc = 0
            self.batch_size = 0
            self.group_acc += x
            self.group_size += 1
        if is_final or t >= self.next_t_to_pop:
            self.next_t_to_pop += self.average_over
            self.pop(t)

    def pop(self, t):
        if self.group_size:
            self.__losses.append(self.group_acc / self.group_size)
            self.__time.append(t)
            self.group_acc = 0
            self.group_size = 0
            self.n_groups += 1
    
    def getHistory(self):
        assert self.batch_size == 0
        assert self.group_size == 0
        return self.__time, self.__losses

def PlotLosses(
    experiment_py_path: str, lossTypes: List[LossType], /, 
    using_epoch_not_batch: bool, 
    average_over: int, 
    start: Optional[int] = None, 
    stop:  Optional[int] = None, 
    which_legend: int = -1, **style_kw, 
):
    (
        experiment_name, n_rand_inits, groups, _, 
    ) = loadExperiment(experiment_py_path)
    print(f'{experiment_name = }')
    group_start = start // average_over
    if stop is None:
        group_stop = None
    else:
        group_stop = stop // average_over
    data: List[Tuple[int, BaseExperimentGroup, int, Dict[LossType, LossAcc]]] = []
    for (group_i, group), rand_init_i in tqdm([*itertools.product(
        enumerate(groups), range(n_rand_inits), 
    )]):
        lossAccs = {x: LossAcc(average_over) for x in lossTypes}
        with OnChangeOrEnd(*[
            x.endBatch for x in lossAccs.values()
        ]) as oCoE:
            total_batch_i = 0
            for (
                epoch_i, batch_i, train_or_validate, _, entries, 
            ) in Decompressor(path.join(getTrainerPath(
                path.dirname(experiment_py_path), 
                group.pathName(), rand_init_i, 
            ), LOSS_FILE_NAME)):
                if using_epoch_not_batch:
                    oCoE.eat(epoch_i)
                else:
                    oCoE.eat(total_batch_i)
                if train_or_validate:
                    total_batch_i += 1
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
        for lossType, lossAcc in lossAccs.items():
            if lossAcc.n_groups == 0:
                raise ValueError('Did not get any', lossType)
        data.append((
            group_i, group, rand_init_i, lossAccs, 
        ))

    for rand_init_i_to_plot in (None, *range(n_rand_inits)):
        fig, axes = plt.subplots(len(lossTypes), 1, sharex=True)
        if len(lossTypes) == 1:
            axes = [axes]   # crazy matplotlib
        for (
            group_i, group, rand_init_i, lossAccs, 
        ) in data:
            if rand_init_i_to_plot is not None and rand_init_i != rand_init_i_to_plot:
                continue
            if rand_init_i == 0 or rand_init_i_to_plot is not None:
                kw = dict(label=group.name())
            else:
                kw = dict()
            for ax, (lossType, lossAcc) in zip(
                axes, lossAccs.items(), 
            ):
                time, losses = lossAcc.getHistory()
                ax.plot(
                    time[group_start:group_stop], 
                    losses[group_start:group_stop], 
                    c=hsv_to_rgb((group_i / len(groups) * .8, 1, .8)), 
                    **kw, **style_kw, 
                )
                ax.set_ylabel(lossType.display_name)
        # for ax in axes:
            # ax.axhline(y=0, color='k')
            # ax.set_ylim(bottom=0)
            # Both will hide a loss=0 curve. 
        axes[which_legend].legend()
        axes[-1].set_xlabel(
            'epoch' if using_epoch_not_batch else 'batch'
        )
        fig.suptitle(experiment_name)
        fig.tight_layout()
        yield fig

class OnChangeOrEnd:
    def __init__(self, *callbacks: Callable) -> None:
        self.callbacks = callbacks

        self.started = False
        self.value = None

    def __enter__(self):
        return self
    
    def callback(self, *a):
        for callback in self.callbacks:
            callback(*a)
    
    def eat(self, x):
        if self.started:
            if self.value != x:
                self.callback(self.value, False)
                self.value = x
        else:
            self.started = True
            self.value = x
    
    def __exit__(self, *_, **__):
        if self.started:
            self.callback(self.value, True)
        return False
