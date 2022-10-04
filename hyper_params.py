from torchWork.loss_weight_tree import LossWeightTree

class BaseHyperParams:
    def __init__(self) -> None:
        self.lossWeightTree: LossWeightTree = None
