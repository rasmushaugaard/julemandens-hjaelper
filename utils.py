import yaml
from pathlib import Path

import pandas as pd
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm
import config
from PIL import Image
from pytorch_lightning.callbacks import TQDMProgressBar
import matplotlib.pyplot as plt


def img_id_from_fp(img_fp: Path):
    return img_fp.with_suffix('').name


def img_fp_from_id(img_id: str):
    return config.images_folder / f'{img_id}.jpg'


def original_img_fp_from_id(img_id: str):
    return config.images_folder_original / f'{img_id}.png'


def load_annotations(fp=config.anno_fp):
    if not fp.exists():
        fp.touch()
    with fp.open() as f:
        lines = f.readlines()
    lines = [l.strip().split(' ') for l in lines if len(l.strip())]
    lines = [(img_id, ok == 'True') for img_id, ok in lines]
    return lines


img_cache = dict()
def load_img(fp: Path, cache=True):
    if cache:
        if fp not in img_cache:
            img_cache[fp] = Image.open(fp)
        return img_cache[fp]
    return Image.open(fp)


class TQDMPBarNoVal(TQDMProgressBar):
    def init_validation_tqdm(self) -> Tqdm:
        return Tqdm(disable=True)
    

def plot_log(version: int):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f'version {version}')
    log_folder = Path(f'logs/lightning_logs/version_{version}')
    df = pd.read_csv(log_folder / 'metrics.csv')
    train_log = df[["train/loss", "train/acc", "step"]].dropna()
    valid_log = df[["valid/loss", "valid/acc", "step"]].dropna()
    for key, ax in ('loss', axs[0]), ('acc', axs[1]):
        ax.plot(train_log['step'], train_log[f'train/{key}'], label='train')
        ax.plot(valid_log['step'], valid_log[f'valid/{key}'], label='valid')
        ax.set_xlabel('step')
        ax.set_ylabel(key)
        if key == 'acc':
            ax.set_ylim(ax.get_ylim()[0], 1.05)
        ax.grid()
    with (log_folder / 'hparams.yaml').open() as f:
        hparams = yaml.safe_load(f)
    print(hparams)