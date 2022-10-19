import sys
from os import path
from typing import Optional, Union, List, Tuple, TextIO
import struct
import pickle
from io import BytesIO

import torch
from torchWork.loss_weight_tree import LossWeightTree
from torchWork.loss_tree import LossTree

LOSS_FILE_NAME = 'losses.torchworklosslog'

class LossLogger:
    def __init__(
        self, path_name, print_every: int = 1, 
    ) -> None:
        self.filename = path.abspath(path.join(
            path_name, LOSS_FILE_NAME, 
        ))
        self.print_every = print_every
        self.compressor = Compressor(self.filename)
    
    def eat(
        self, epoch_i: int, batch_i: int, 
        train_or_validate: bool, 
        profiler, 
        lossRoot: LossTree, lossWeightTree: LossWeightTree, 
        extras: List[Tuple[str, torch.Tensor]]=None, 
        flush=True, 
    ):
        self.compressor.newBatch(
            epoch_i, batch_i, train_or_validate, 
        )
        self.dfs(lossRoot, lossWeightTree, epoch_i, 1)
        if extras is not None:
            for key, value in extras:
                self.compressor.write(key, value.item(), 1)
        with profiler('log.mesaFlush'):
            self.compressor.mesaFlush(profiler)
        if flush:
            with profiler('log.flush'):
                self.compressor.flush()

    def dfs(
        self, loss: LossTree, lossWeightTree: LossWeightTree, 
        epoch_i: int, depth: int, 
    ):
        _sum = loss.sum(lossWeightTree, epoch_i).item()
        self.compressor.write(
            loss.name, _sum, depth, 
        )
        for lossWeightNode in lossWeightTree.children:
            name = lossWeightNode.name
            lossChild: Union[
                LossTree, Optional[torch.Tensor], 
            ] = loss.__getattribute__(name)
            if lossWeightNode.children is None:
                self.compressor.write(
                    name, 
                    0.0 if lossChild is None else lossChild.item(), 
                    depth + 1, 
                )
            else:
                self.dfs(
                    lossChild, lossWeightNode, 
                    epoch_i, depth + 1, 
                )

    def clearFile(self):
        with open(self.filename, 'wb'):
            pass

class Compressor:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.keys: Optional[List[str]] = None
        self.struct_format: Optional[str] = None

        self.buffered_keys = []
        self.buffered_values: List[float] = []
        self.now_batch = None
        self.io = BytesIO()

    def write(self, key: str, value: float, padding: int):
        self.buffered_keys.append(' ' * padding + key)
        self.buffered_values.append(value)
    
    def flush(self):
        self.io.flush()
        self.io.seek(0)
        with open(self.filename, 'ab') as f:
            f.write(self.io.read())
        self.io.seek(0)
        self.io.truncate()

    def mesaFlush(self, profiler):
        f = self.io
        if self.keys is None:
            self.keys = self.buffered_keys.copy()
            pickle.dump(self.keys, f)
            self.struct_format = '!II?' + 'f' * len(self.keys)
        assert self.keys == self.buffered_keys
        epoch_i, batch_i, train_or_validate = self.now_batch
        self.now_batch = None
        with profiler('mesa.pack'):
            data = struct.pack(
                self.struct_format, 
                epoch_i, batch_i, train_or_validate, 
                *self.buffered_values, 
            )
        with profiler('mesa.write'):
            f.write(data)
        self.buffered_keys  .clear()
        self.buffered_values.clear()
    
    def newBatch(
        self, epoch_i: int, batch_i: int, 
        train_or_validate: bool, 
    ):
        self.now_batch = (epoch_i, batch_i, train_or_validate)

def Decompressor(filename: str):
    with open(filename, 'rb') as f:
        pretty_keys: List[str] = pickle.load(f)
        stack = []
        tree_keys = []
        for pretty_key in pretty_keys:
            node_name = pretty_key.lstrip(' ')
            depth = len(pretty_key) - len(node_name) - 1
            for _ in range(len(stack) - depth):
                stack.pop(-1)
            stack.append(node_name)
            tree_keys.append('.'.join(stack))
        while True:
            data = f.read(4)
            if data == b'':
                break
            epoch_i          : int  = struct.unpack('!I', data)[0]
            batch_i          : int  = struct.unpack('!I', f.read(4))[0]
            train_not_validate: bool = struct.unpack('!?', f.read(1))[0]
            loss_values = [
                struct.unpack('!f', f.read(4))[0]
                for _ in pretty_keys
            ]
            pretty_entries = dict(zip(pretty_keys, loss_values))
            tree_entries   = dict(zip(tree_keys  , loss_values))

            yield (
                epoch_i, batch_i, train_not_validate, 
                pretty_entries, tree_entries, 
            )

def decompressToText(input_filename: str, output: TextIO):
    d = Decompressor(input_filename)
    for epoch_i, batch_i, train_or_validate, entries, _ in d:
        print(f'''epoch {epoch_i}, {
            'train' if train_or_validate else 'validate'
        }, batch {batch_i}:''', file=output)
        for key, value in entries.items():
            print(key, '=', value, file=output)

def previewLosses(filename):
    decompressToText(filename, sys.stdout)

def decompressLosses(filename):
    with open(filename + '.txt', 'w') as f:
        decompressToText(filename, f)
