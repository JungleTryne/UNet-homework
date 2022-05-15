import os
import requests
import logging
import zipfile
import sys

from pathlib import Path
from tqdm import tqdm
from logging import StreamHandler, Formatter

from util.constants import DATASET_URL, DATASET_PATH


logger = logging.getLogger('DatasetManager')
logger.setLevel(logging.INFO)
handler = StreamHandler(stream=sys.stdout)
handler.setFormatter(Formatter(fmt='[%(asctime)s: %(levelname)s] %(message)s'))
logger.addHandler(handler)


def is_dataset_downloaded() -> bool:
    path = DATASET_PATH
    return os.path.exists(path) and len(os.listdir(path)) > 0


def download_dataset():
    Path(DATASET_PATH).mkdir(parents=True, exist_ok=True)

    logger.info("Downloading the archive...")
    url = DATASET_URL
    response = requests.get(url, stream=True)
    with open(os.path.join(DATASET_PATH, "dataset.zip"), "wb") as handle:
        for data in tqdm(response.iter_content(chunk_size=8192)):
            handle.write(data)

    logger.info("Extracting data...")
    with zipfile.ZipFile(os.path.join(DATASET_PATH, "dataset.zip"), "r") as zip_ref:
        zip_ref.extractall(DATASET_PATH)
    os.remove(os.path.join(DATASET_PATH, "dataset.zip"))
