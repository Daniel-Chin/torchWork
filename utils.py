from typing import List
import torch

def getParams(optim: torch.optim.Optimizer):
    s: List[torch.Tensor] = []
    for param in optim.param_groups[0]['params']:
        if param.grad is not None:
            s.append(param)
    return s

cached_sizes = []
def getGradNorm(params: List[torch.Tensor]):
    for p, size in cached_sizes:
        if p is params:
            buffer = torch.zeros((size, ))
            for i, param in enumerate(params):
                buffer[i] = param.grad.norm(2)
            return buffer.norm(2)
    else:
        acc = 0
        for i, param in enumerate(params):
            acc += param.grad.norm(2).item() ** 2
        cached_sizes.append((params, i + 1))
        return acc ** .5
