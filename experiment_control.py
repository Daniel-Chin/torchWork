import os
from os import path
from typing import Callable, Dict, Tuple, Type, List
from datetime import datetime
from time import perf_counter
import importlib.util
import shutil
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.utils.data
from torch import nn
from torchWork import BaseHyperParams, LossLogger, Profiler, DEVICE
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
    str, int, List[ExperimentGroup], 
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
    )

def getCommitHash():
    repo = git.Repo('.', search_parent_directories=True)
    return next(repo.iter_commits()).hexsha

def runExperiment(
    current_experiment_path: str, 
    oneEpoch: Callable, 
    modelClasses: Dict[str, Type[nn.Module]], 
    trainSet   : torch.utils.data.Dataset, 
    validateSet: torch.utils.data.Dataset, 
    save_path: str = './experiments', 
):
    (
        experiment_name, n_rand_inits, groups, 
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
        experiment_name + '_' + datetime.now().strftime(
            '%Y_%b_%d_%H;%M;%S', 
        ), 
    )
    os.makedirs(exp_path)
    shutil.copy(current_experiment_path, path.join(
        exp_path, EXPERIMENT_PY_FILENAME, 
    ))
    with open(path.join(
        exp_path, 'commit_hash.txt', 
    ), 'w') as f:
        print(getCommitHash(), file=f)
    
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

    print('Training starts...', flush=True)
    profiler = Profiler()
    for trainer_i in roundRobinSched(len(trainers)):
        trainer: Trainer = trainers[trainer_i]
        with profiler('oneEpoch'):
            oneEpoch(
                trainer.name, trainer.epoch, trainer.hyperParams, 
                models, trainer.optim, 
                trainSet, validateSet, 
                trainer.lossLogger, profiler, 
                trainer.save_path, 
            )
        trainer.epoch += 1

def getGroupPath(
    experiment_path: str, group_name: str, rand_init_i: int, 
):
    return path.join(
        experiment_path, 
        group_name + f'_rand_{rand_init_i}', 
    )
