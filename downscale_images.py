from pathlib import Path

import tqdm
from PIL import Image

import config


def downscale_images(res=config.img_res):
    folder = Path(f'images_{res}')

    folder.mkdir(exist_ok=True)
    for fp in folder.glob('*.jpg'):
        fp.unlink()
    folder.rmdir()
    folder.mkdir()

    for fp in tqdm.tqdm(list(Path('images').glob('*.png'))):
        img = Image.open(fp)
        img_ = img.resize((res, res), Image.Resampling.LANCZOS)
        img_.save(folder / fp.with_suffix('.jpg').name, optimize=True, quality=90)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--res', type=int, default=config.img_res)
    res = parser.parse_args().res
    downscale_images(res)