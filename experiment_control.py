import os
from os import path
from typing import Any, Callable, Dict, Optional, Tuple, Type, List
from datetime import datetime
from time import perf_counter
import importlib.util
import shutil
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.utils.data
from torch import nn
from torchWork import BaseHyperParams, LossLogger, Profiler, DEVICE, HAS_CUDA
import git

EXPERIMENT_PY_FILENAME = 'experiment.py'

class ExperimentGroup(ABC):
    def __init__(self, hyperParams: BaseHyperParams) -> None:
        self.hyperParams = hyperParams
    
    @abstractmethod
    def name(self):
        # Override this method!
        raise NotImplemented
    
class Trainer:
    def __init__(
        self, hyperParams: BaseHyperParams, 
        models: Dict[str, nn.Module], save_path, name, 
    ) -> None:
        self.hyperParams = hyperParams
        self.models = models
        self.save_path = save_path
        self.name = name

        self.epoch = 0
        all_params = []
        for model in models.values():
            all_params.extend(model.parameters())
        self.optim = hyperParams.OptimClass(
            all_params, lr=hyperParams.lr, 
        )
        os.mkdir(save_path)
        self.lossLogger = LossLogger(save_path)
        self.lossLogger.clearFile()
    
def roundRobinSched(n_workers):
    ages = np.zeros((n_workers, ))
    while True:
        elected = ages.argmin()
        start = perf_counter()
        yield elected
        end   = perf_counter()
        ages[elected] += end - start

def loadExperiment(experiment_py_path) -> Tuple[
    str, int, List[ExperimentGroup], Any, 
]:
    spec = importlib.util.spec_from_file_location(
        "experiment", experiment_py_path, 
    )
    experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment)
    return (
        experiment.EXP_NAME, 
        experiment.N_RAND_INITS, 
        experiment.GROUPS, 
        experiment, 
    )

def getCommitHash():
    repo = git.Repo('.', search_parent_directories=True)
    return next(repo.iter_commits()).hexsha

def runExperiment(
    current_experiment_path: str, 
    oneEpoch: Callable[[Any], bool], 
    modelClasses: Dict[str, Type[nn.Module]], 
    trainSet   : torch.utils.data.Dataset, 
    validateSet: torch.utils.data.Dataset, 
    save_path: str = './experiments', 
):
    print('Loading experiment...', flush=True)
    (
        experiment_name, n_rand_inits, groups, experiment, 
    ) = loadExperiment(current_experiment_path)

    names = set()
    for group in groups:
        name = group.name()
        if name in names:
            raise ValueError(
                f'Multiple experiment groups named "{name}"', 
            )
        names.add(name)
    exp_path = path.join(
        path.abspath(save_path), 
        datetime.now().strftime(
            '%Y_m%m_d%d@%H_%M_%S', 
        ) + '_' + experiment_name, 
    )
    os.makedirs(exp_path)
    shutil.copy(current_experiment_path, path.join(
        exp_path, EXPERIMENT_PY_FILENAME, 
    ))
    with open(path.join(
        exp_path, 'commit_hash.txt', 
    ), 'w') as f:
        print(getCommitHash(), file=f)
    
    print('Initing trainers...', flush=True)
    trainers = []
    for group in groups:
        for rand_init_i in range(n_rand_inits):
            models = {}
            for name, ModelClass in modelClasses.items():
                models[name] = ModelClass(group.hyperParams).to(DEVICE)
            group_path = getGroupPath(
                exp_path, group.name(), rand_init_i, 
            )
            trainer_name = path.split(group_path)[-1]
            trainer = Trainer(
                group.hyperParams, models, 
                group_path, trainer_name, 
            )
            trainers.append(trainer)

    print('Syncing GPU...', flush=True)
    if HAS_CUDA:
        a = torch.zeros((3, ), device=DEVICE, requires_grad=True)
        b = a + 3
        b.sum().backward()
        torch.cuda.synchronize()    # just for profiling
    print('Training starts...', flush=True)
    profiler = Profiler()
    while trainers:
        for trainer_i in roundRobinSched(len(trainers)):
            trainer: Trainer = trainers[trainer_i]
            with profiler('oneEpoch'):
                do_continue = oneEpoch(
                    trainer.name, trainer.epoch, 
                    experiment, trainer.hyperParams, 
                    trainer.models, trainer.optim, 
                    trainSet, validateSet, 
                    trainer.lossLogger, profiler, 
                    trainer.save_path, trainer_i, 
                )
                if not do_continue:
                    trainers.pop(trainer_i)
                    break
            trainer.epoch += 1
    print('All trainers stopped.', flush=True)

def getGroupPath(
    experiment_path: str, group_name: str, rand_init_i: int, 
):
    return path.join(
        experiment_path, 
        group_name + f'_rand_{rand_init_i}', 
    )

def saveModels(models: Dict[str, nn.Module], epoch, save_path):
    for key, model in models.items():
        torch.save(model.state_dict(), path.join(
            save_path, f'{key}_epoch_{epoch}.pt', 
        ))

def loadLatestModels(
    experiment_path: str, group: ExperimentGroup, 
    rand_init_i: int, 
    modelClasses: Dict[str, Type[nn.Module]], 
    lock_epoch: Optional[int]=None, 
):
    models: Dict[str, nn.Module] = {}
    for name, ModelClass in modelClasses.items():
        models[name] = ModelClass(group.hyperParams).to(DEVICE)
    
    group_path = getGroupPath(experiment_path, group.name(), rand_init_i)
    if lock_epoch is None:
        max_epoch = 0
        for filename in os.listdir(group_path):
            try:
                x = filename.split('_epoch_')[1]
                x = x.split('.pt')[0]
                epoch = int(x)
            except (ValueError, IndexError):
                continue
            else:
                max_epoch = max(max_epoch, epoch)
        epoch = max_epoch
    else:
        epoch = lock_epoch
    print('taking epoch', epoch)
    for name, model in models.items():
        model.load_state_dict(torch.load(path.join(
            group_path, f'{name}_epoch_{epoch}.pt', 
        ), map_location=DEVICE))
    return models
