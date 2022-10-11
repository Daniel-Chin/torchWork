import sys
import os
from typing import Dict, Optional, Union, List, Tuple, TextIO
import struct
import pickle
from io import BytesIO
from contextlib import nullcontext

from torchWork.loss_weight_tree import LossWeightTree
from torchWork.loss_tree import LossTree

class LossLogger:
    def __init__(
        self, filename, print_every: int = 1, 
    ) -> None:
        self.filename = os.path.abspath(filename)
        self.print_every = print_every
        self.compressor = Compressor(self.filename)
    
    def eat(
        self, epoch_i: int, batch_i: int, 
        train_or_validate: bool, 
        lossRoot: LossTree, lossWeightTree: LossWeightTree, 
        extras: List[Tuple[str, float]]=None, 
        profiler=None, 
    ):
        with (profiler or nullcontext)('log.newBatch'):
            self.compressor.newBatch(
                epoch_i, batch_i, train_or_validate, 
            )
        self.dfs(lossRoot, lossWeightTree, epoch_i, 1, profiler)
        with (profiler or nullcontext)('log.extras'):
            if extras is not None:
                for key, value in extras:
                    self.compressor.write(key, value, 1)
        with (profiler or nullcontext)('log.mesaFlush'):
            self.compressor.mesaFlush()
        with (profiler or nullcontext)('log.flush'):
            if epoch_i % 8 == 0:
                self.compressor.flush()

    def dfs(
        self, loss: LossTree, lossWeightTree: LossWeightTree, 
        epoch_i: int, depth: int, profiler, 
    ):
        with (profiler or nullcontext)('dfs.0'):
            self.compressor.write(
                loss.name, loss.sum(lossWeightTree, epoch_i), depth, 
            )
        for lossWeightNode in lossWeightTree.children:
            with (profiler or nullcontext)('dfs.1'):
                name = lossWeightNode.name
                lossChild: Union[
                    LossTree, float, 
                ] = loss.__getattribute__(name)
            if lossWeightNode.children is None:
                with (profiler or nullcontext)('dfs.2'):
                    self.compressor.write(name, lossChild, depth)
            else:
                self.dfs(
                    lossChild, lossWeightNode, 
                    epoch_i, depth + 1, profiler, 
                )

    def clearFile(self):
        with open(self.filename, 'wb'):
            pass

class Compressor:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.keys: Optional[List[str]] = None

        self.buffered_keys = []
        self.buffered_values = []
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

    def mesaFlush(self):
        f = self.io
        if self.keys is None:
            self.keys = self.buffered_keys.copy()
            pickle.dump(self.keys, f)
        assert self.keys == self.buffered_keys
        epoch_i, batch_i, train_or_validate = self.now_batch
        self.now_batch = None
        f.write(struct.pack('!I', epoch_i))
        f.write(struct.pack('!I', batch_i))
        f.write(struct.pack('!?', train_or_validate))
        for value in self.buffered_values:
            f.write(struct.pack('!f', value or 0.0))
        self.buffered_keys  .clear()
        self.buffered_values.clear()
    
    def newBatch(
        self, epoch_i: int, batch_i: int, 
        train_or_validate: bool, 
    ):
        self.now_batch = (epoch_i, batch_i, train_or_validate)

def Decompressor(filename: str):
    with open(filename, 'rb') as f:
        keys: List[str] = pickle.load(f)
        while True:
            data = f.read(4)
            if data == b'':
                break
            epoch_i           = struct.unpack('!I', data)[0]
            batch_i           = struct.unpack('!I', f.read(4))[0]
            train_or_validate = struct.unpack('!?', f.read(1))[0]
            entries: Dict[str, float] = {}
            for key in keys:
                entries[key] = struct.unpack('!f', f.read(4))[0]
            yield epoch_i, batch_i, train_or_validate, entries

def decompressToText(input_filename: str, output: TextIO):
    d = Decompressor(input_filename)
    for epoch_i, batch_i, train_or_validate, entries in d:
        print(f'''epoch {epoch_i}, {
            'train' if train_or_validate else 'validate'
        }, batch {batch_i}:''', file=output)
        for key, value in entries.items():
            print(key, '=', value, file=output)

def previewLosses(filename):
    decompressToText(filename, sys.stdout)
