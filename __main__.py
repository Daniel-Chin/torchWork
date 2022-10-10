import argparse

import torchWork.demo
import torchWork.loss_logger

parser = argparse.ArgumentParser()
parser.add_argument(
    '--preview-loss', dest='preview_loss', 
)
args = parser.parse_args()

if args.preview_loss is not None:
    torchWork.loss_logger.previewLosses(args.preview_loss)
else:
    torchWork.demo.main()
