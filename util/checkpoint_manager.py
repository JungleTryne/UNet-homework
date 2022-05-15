import os
import logging
import requests
import sys

from pathlib import Path
from tqdm import tqdm
from logging import StreamHandler, Formatter

from util.constants import CHECKPOINT_FOLDER, DEFAULT_CHECKPOINT_PATH, DEFAULT_CHECKPOINT_URL


logger = logging.getLogger('CheckpointManager')
logger.setLevel(logging.INFO)
handler = StreamHandler(stream=sys.stdout)
handler.setFormatter(Formatter(fmt='[%(asctime)s: %(levelname)s] %(message)s'))
logger.addHandler(handler)


def does_default_checkpoint_exist() -> bool:
    path = DEFAULT_CHECKPOINT_PATH
    return os.path.exists(path)


def download_checkpoint():
    path = CHECKPOINT_FOLDER
    Path(path).mkdir(parents=True, exist_ok=True)

    logger.info("Downloading the checkpoint")
    url = DEFAULT_CHECKPOINT_URL
    response = requests.get(url, stream=True)
    with open(os.path.join(path, "unet.ckpt"), "wb") as handle:
        for data in tqdm(response.iter_content(chunk_size=8192)):
            handle.write(data)
