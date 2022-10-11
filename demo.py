from time import sleep

import torch
import torch.nn.functional as F

from torchWork import *
AbstractLossNode = loss_tree.AbstractLossNode
try:
    from losses import Total_loss
except ImportError as e:
    print(e)

class HyperParams(BaseHyperParams):
    def __init__(self):
        super().__init__()
        self.rnn_width = None
        self.do_grad_clip = None
        self.grad_clip_ceil = None
        self.image_loss = None
    
    def validate(self):
        if self.do_grad_clip:
            assert self.grad_clip_ceil > 0
        else:
            assert self.grad_clip_ceil is None
    
    def expand(self):
        self.imgCriterion = {
            'bce': torch.nn.BCELoss(), 
            'mse': F.mse_loss, 
        }[self.image_loss]

def main():
    print('Press "1" to generate losses.py .')
    print('Press "2" to run demo.')
    if input('>') == '2':
        writeLosses()
    else:
        writeLosses()
        print('done')
        return

    hyperParams = HyperParams()
    hyperParams.loss_weight_tree = LossWeightTree('total', 1, [
        LossWeightTree('vae', .5, [
            LossWeightTree('reconstruct', .9, None), 
            LossWeightTree('kld', .1, None), 
        ]), 
        LossWeightTree('vrnn', .4, [
            LossWeightTree('predict', .9, [
                LossWeightTree('z', .5, None),
                LossWeightTree('image', .5, None),
            ]), 
            LossWeightTree('kld', .1, None), 
        ]), 
        LossWeightTree('weight_decay', .1, None), 
    ])
    hyperParams.rnn_width = 8
    hyperParams.do_grad_clip = True
    hyperParams.grad_clip_ceil = 1
    hyperParams.image_loss = 'mse'
    hyperParams.validate()
    hyperParams.expand()

    total_loss = Total_loss()
    total_loss.vae.reconstruct = 1
    total_loss.vae.kld = 2
    total_loss.vrnn.predict.z = 3
    total_loss.vrnn.predict.image = 4
    total_loss.vrnn.kld = 5
    total_loss.weight_decay = 6
    print('loss sum:', total_loss.sum(hyperParams.loss_weight_tree, epoch=0))

    lossLogger = LossLogger('losses.torchworklosslog')
    lossLogger.clearFile()
    lossLogger.eat(0, 0, True, total_loss, hyperParams.loss_weight_tree)
    lossLogger.eat(0, 1, True, total_loss + total_loss, hyperParams.loss_weight_tree)

    profiler = Profiler()
    for i in range(4):
        # dumb slow code
        sleep(.1)
        with profiler('good'):
            # backwards
            sleep(.3)
        profiler.report()

def writeLosses():
    absLossRoot = AbstractLossNode('total_loss', [
        AbstractLossNode('vae', ['reconstruct', 'kld']), 
        AbstractLossNode('vrnn', [
            AbstractLossNode('predict', ['z', 'image']), 
            'kld', 
        ]), 
        'weight_decay',
    ])
    with open('losses.py', 'w') as f:
        loss_tree.writeCode(f, absLossRoot)

if __name__ == '__main__':
    main()
    print('For a more complete demo, see https://github.com/Daniel-Chin/3S/tree/main/bounce_clean')
