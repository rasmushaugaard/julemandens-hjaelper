import yaml
from pathlib import Path

import numpy as np
import pandas as pd
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm
import config
from PIL import Image
import torch
import torch.utils.data
import torch.nn.functional as F
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
    lines = [(img_id, ok == 'True') for img_id, ok in lines if img_fp_from_id(img_id).exists()]
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
    x_key = 'epoch'
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f'version {version}')
    log_folder = Path(f'logs/lightning_logs/version_{version}')
    df = pd.read_csv(log_folder / 'metrics.csv')
    train_log = df[["train/loss", "train/acc", x_key]].dropna()
    valid_log = df[["valid/loss", "valid/acc", x_key]].dropna()
    for key, ax in ('loss', axs[0]), ('acc', axs[1]):
        scale = 100 if key == 'acc' else 1
        ax.plot(train_log[x_key], train_log[f'train/{key}'] * scale, label='træning')
        ax.plot(valid_log[x_key], valid_log[f'valid/{key}'] * scale, label='validering')
        ax.set_xlabel('epoke')
        ax.set_ylabel(key)
        if key == 'acc':
            ax.set_ylabel('Nøjagtighed [%]')
            ax.set_ylim(40, 105)
        ax.grid()
    axs[1].legend()
    with (log_folder / 'hparams.yaml').open() as f:
        hparams = yaml.safe_load(f)
    print(hparams)
    plt.show()
    print(f'træningsnøjagtighed: {list(df["train/acc"].dropna())[-1] * 100:.0f}%')
    print(f'valideringsnøjagtighed: {list(df["valid/acc"].dropna())[-1] * 100:.0f}%')

def vis_augmenteringseksempler(augmenteringer):
    from data import GaveDatasæt

    dataset = GaveDatasæt(load_annotations(), augs=augmenteringer)
    fig, axs = plt.subplots(2, 5, figsize=(12, 5))
    idxs = np.random.permutation(len(dataset))
    for i, ax in zip(idxs, axs.reshape(-1)):
        img, ok = dataset[i]
        ax.imshow(Image.fromarray(dataset.denormalize(img)))
        ax.axis('off')
        ax.set_title('ok' if ok else 'ikke ok')
    plt.show()



def vis_model_fejl(model, slice=slice(0, 100)):
    from data import GaveDatasæt

    dataset = GaveDatasæt(load_annotations()[slice])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=2)
    losses_per_image = []
    predictions_per_image = []

    device = 'cuda'
    model.eval()
    model.to(device)

    with torch.no_grad():
        for img, target in dataloader:
            img, target = img.to(device), target.to(device)
            lgts = model.model(img.permute(0, 3, 1, 2)).view(-1)
            predictions_per_image.append(torch.sigmoid(lgts))
            loss = F.binary_cross_entropy_with_logits(lgts, target, reduction='none')
            losses_per_image.append(loss)

    predictions_per_image = torch.cat(predictions_per_image).cpu().numpy()
    losses_per_image = torch.cat(losses_per_image).cpu().numpy()

    n_rows = 4
    n_cols = 5
    n = n_rows * n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2.5*n_rows))
    idxs = np.argsort(losses_per_image)[::-1]
    for i, ax in zip(idxs, axs.reshape(-1)):
        print(i, dataset.annotations[i][0])
        img, ok = dataset[i]
        ax.imshow(Image.fromarray(dataset.denormalize(img)))
        ax.axis('off')
        ax.set_title(('ok' if ok else 'ikke ok') + f' / model: {predictions_per_image[i]:.1f}')

    plt.tight_layout()
    plt.show()