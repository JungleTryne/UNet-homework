import argparse
import click
import sys

from util.constants import DEFAULT_CHECKPOINT_PATH
from util.dataset_manager import is_dataset_downloaded, download_dataset
from util.checkpoint_manager import does_default_checkpoint_exist, download_checkpoint
from util.test import test
from util.train import train

import logging
from logging import StreamHandler, Formatter

logger = logging.getLogger('UNetExecutor')
logger.setLevel(logging.INFO)
handler = StreamHandler(stream=sys.stdout)
handler.setFormatter(Formatter(fmt='[%(asctime)s: %(levelname)s] %(message)s'))
logger.addHandler(handler)


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--mode', type=str, default='train')
    argument_parser.add_argument('--checkpoint_path', type=str, default="")
    argument_parser.add_argument('--device', type=str, default='cpu')

    args = argument_parser.parse_args()

    if not is_dataset_downloaded():
        if click.confirm('Do you want to download dataset? (2 GB)', default=True):
            download_dataset()
            logger.info("Dataset was downloaded successfully")
        else:
            logger.error("Cannot continue without dataset")
            return
    else:
        logger.info("Dataset found!")

    if args.mode == "train":
        ckpt_path = args.checkpoint_path or DEFAULT_CHECKPOINT_PATH
        logger.info("Starting train phase of the model")
        train(checkpoint_path=ckpt_path, device=args.device)
    elif args.mode == "test":
        if not args.checkpoint_path:
            if not does_default_checkpoint_exist():
                if click.confirm('Do you want to download checkpoint? (800 MB)', default=True):
                    download_checkpoint()
                else:
                    logger.error("Cannot continue testing without checkpoint")
                    return
            else:
                logger.info("Fount dataset!")
            ckpt_path = "./bin/checkpoint/unet.ckpt"
        else:
            ckpt_path = args.checkpoint_path
        logger.info("Starting test phase of the model")
        test(checkpoint_path=ckpt_path, device=args.device)
    else:
        logger.error("Unknown mode")


if __name__ == "__main__":
    main()
