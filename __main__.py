import argparse

import torchWork.loss_logger

parser = argparse.ArgumentParser()
parser.add_argument(
    '--preview-loss', dest='preview_loss', 
)
parser.add_argument(
    '--decompress-loss', dest='decompress_loss', 
)
parser.add_argument(
    '--open-loss-interactive', dest='open_loss_interactive', 
)
args = parser.parse_args()

if args.preview_loss is not None:
    torchWork.loss_logger.previewLosses(args.preview_loss)
elif args.decompress_loss is not None:
    torchWork.loss_logger.decompressLosses(args.decompress_loss)
elif args.open_loss_interactive is not None:
    filename = args.open_loss_interactive
    print(f'{filename = }')
    print('Please select:')
    print('1: Preview losses. ')
    print('2: Decompress losses to local file. ')
    op = input('>').strip()
    if op == '1':
        torchWork.loss_logger.previewLosses(filename)
    elif op == '2':
        torchWork.loss_logger.decompressLosses(filename)
    else:
        print(f'What is "{op}"?')
else:
    import torchWork.demo
    torchWork.demo.main()
