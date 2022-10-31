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
    def name(self) -> str:
        # Override this method!
        raise NotImplemented
    
    def pathName(self):
        return self.name() \
            .replace(':', '-') \
            .replace('<', '(') \
            .replace('>', ')') \
            .replace(' ', '') \
    
class Trainer:
    def __init__(
        self, hyperParams: BaseHyperParams, 
        models: Dict[str, nn.Module], save_path, name, 
        do_clear_log=True, do_mkdir=True, epoch=0, 
    ) -> None:
        self.hyperParams = hyperParams
        self.models = models
        self.save_path = save_path
        self.name = name
        self.epoch = epoch

        all_params = []
        for model in models.values():
            all_params.extend(model.parameters())
        self.optim = hyperParams.OptimClass(
            all_params, lr=hyperParams.lr, 
        )
        if do_mkdir:
            os.mkdir(save_path)
        self.lossLogger = LossLogger(save_path)
        if do_clear_log:
            self.lossLogger.clearFile()
    
    @staticmethod
    def loadFromDisk(
        group: ExperimentGroup, 
        modelClasses: Dict[str, Type[nn.Module]], 
        trainer_path, trainer_name, rand_init_i, 
        lock_epoch: Optional[int] = None, 
    ):
        # Doesn't preserve the optim.  
        experiment_path = path.abspath(path.join(
            trainer_path, '..', 
        ))
        epoch, models = loadLatestModels(
            experiment_path, group, rand_init_i, 
            modelClasses, lock_epoch, 
        )

        return __class__(
            group.hyperParams, models, trainer_path, 
            trainer_name, False, False, epoch, 
        )
    
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
    continue_from: Optional[str] = None, 
):
    print('Loading experiment...', flush=True)
    (
        experiment_name, n_rand_inits, groups, experiment, 
    ) = loadExperiment(current_experiment_path)

    # check for collision
    path_names = set()
    for group in groups:
        path_name = group.pathName()
        if path_name in path_names:
            raise ValueError(
                f'Multiple experiment groups named "{path_name}"', 
            )
        path_names.add(path_name)
    del path_names

    if continue_from is None:
        exp_dir = datetime.now().strftime(
            '%Y_m%m_d%d@%H_%M_%S', 
        ) + '_' + experiment_name
    else:
        exp_dir = continue_from
    exp_path = path.join(path.abspath(save_path), exp_dir)
    if continue_from is None:
        os.makedirs(exp_path)
        shutil.copy(current_experiment_path, path.join(
            exp_path, EXPERIMENT_PY_FILENAME, 
        ))
    with open(path.join(
        exp_path, 'commit_hash.txt', 
    ), 'a') as f:
        print(getCommitHash(), file=f)
    
    print('Initing trainers...', flush=True)
    trainers = []
    for group in groups:
        for rand_init_i in range(n_rand_inits):
            models = instantiateModels(
                modelClasses, group.hyperParams, 
            )
            trainer_path = getTrainerPath(
                exp_path, group.pathName(), rand_init_i, 
            )
            trainer_name = path.split(trainer_path)[-1]
            if continue_from is None:
                trainer = Trainer(
                    group.hyperParams, models, 
                    trainer_path, trainer_name, 
                )
            else:
                trainer = Trainer.loadFromDisk(
                    group, modelClasses, 
                    trainer_path, trainer_name, rand_init_i, 
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

def getTrainerPath(
    experiment_path: str, group_path_name: str, rand_init_i: int, 
):
    return path.join(
        experiment_path, 
        group_path_name + f'_rand_{rand_init_i}', 
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
    models = instantiateModels(
        modelClasses, group.hyperParams, 
    )
    
    trainer_path = getTrainerPath(
        experiment_path, group.pathName(), rand_init_i, 
    )
    if lock_epoch is None:
        max_epoch = 0
        for filename in os.listdir(trainer_path):
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
            trainer_path, f'{name}_epoch_{epoch}.pt', 
        ), map_location=DEVICE))
    return epoch, models

def instantiateModels(
    modelClasses: Dict[str, Type[nn.Module]], 
    hyperParams: BaseHyperParams, 
):
    models: Dict[str, nn.Module] = {}
    for name, ModelClass in modelClasses.items():
        models[name] = ModelClass(hyperParams).to(DEVICE)
    return models
