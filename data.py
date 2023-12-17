import torch.utils.data
import albumentations as A
import numpy as np

import utils


class GaveDatasæt(torch.utils.data.Dataset):
    def __init__(self, annotations, augs=[], cache=True):
        self.annotations = annotations
        self.n = len(annotations)
        self.augs = A.Compose([
            A.CenterCrop(300, 300),
            *augs,
            A.Resize(224, 224),
        ])
        self.cache = cache
        if cache:
            # make sure images are cached before multiprocessing
            for i in range(len(self)):
                self.load_anno(i)[0].load()

    def __len__(self):
        return self.n

    @staticmethod
    def normalize(img):
        return (img.astype(np.float32) / 255 - 0.5) / 0.2

    @staticmethod
    def denormalize(img):
        return (((img * 0.2) + 0.5) * 255).astype(np.uint8)
    
    def load_anno(self, i):
        img_id, er_ok = self.annotations[i]
        img = utils.load_img(utils.img_fp_from_id(img_id), cache=self.cache)
        return img, er_ok

    def __getitem__(self, i):
        img, er_ok = self.load_anno(i)
        img = self.augs(image=np.asarray(img))['image']
        img = self.normalize(img)
        return img, float(er_ok)