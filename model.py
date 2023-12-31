import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import pytorch_lightning as pl


_modeller = dict(
    lille=(torchvision.models.resnet18, torchvision.models.ResNet18_Weights.IMAGENET1K_V1, 512),
    mellem=(torchvision.models.resnet34, torchvision.models.ResNet34_Weights.IMAGENET1K_V1, 512),
    stor=(torchvision.models.resnet50, torchvision.models.ResNet50_Weights.IMAGENET1K_V2, 2048),
    større=(torchvision.models.resnet101, torchvision.models.ResNet101_Weights.IMAGENET1K_V2, 2048),
)

class GaveModel(pl.LightningModule):
    def __init__(self, lr=1e-3, pretrained=True, freeze=False, use_sched=True, størrelse='lille'):
        super().__init__()
        self.lr = lr
        self.use_sched = use_sched
        
        assert størrelse in _modeller, f"""
            størrelse skal være én af følgende: {list(_modeller.keys())}
        """

        model, weights, ch = _modeller[størrelse]
        self.model = model(
            weights=weights if pretrained else None,
        )
        self.model.fc = nn.Linear(ch, 1)
        if freeze:
            for module in list(self.model.children())[:6]:
                for param in module.parameters():
                    param.requires_grad = False

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        if not self.use_sched:
            return opt
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=opt,
            max_lr=self.lr,
            total_steps=self.trainer.max_steps,
        )
        return dict(
            optimizer=opt, 
            lr_scheduler=dict(
                scheduler=sched,
                interval='step',
            ),
        )
    
    def step(self, batch, name):
        imgs, targets = batch
        imgs = imgs.permute(0, 3, 1, 2)  # (b, h, w, 3) -> (b, 3, h, w)
        b = len(imgs)
        lgts = self.model(imgs).view(b)
        loss = F.binary_cross_entropy_with_logits(
            input=lgts,
            target=targets,
        )
        acc = ((lgts < 0) == (targets < 0.5)).float().mean()
        self.log(f'{name}/loss', loss, on_step=False, on_epoch=True)
        self.log(f'{name}/acc', acc, on_step=False, on_epoch=True)
        return loss
    
    def training_step(self, batch):
        return self.step(batch, 'train')

    def validation_step(self, batch, _):
        return self.step(batch, 'valid')