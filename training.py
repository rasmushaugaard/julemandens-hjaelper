import warnings

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor

import utils
from data import GaveDatasæt
from model import GaveModel


warnings.filterwarnings('ignore', category=pl.utilities.warnings.PossibleUserWarning)


def træn_model(
    epochs: int, lr: float, 
    start_fra_prætrænede_vægte: bool,
    frys_backbone: bool,
    augmenteringer: list, 
    n_train: int, n_val: int,
) -> (GaveModel, int):
    annotations = utils.load_annotations()
    assert n_train + n_val <= len(annotations), """
        n_train + n_val må ikke være større end antallet af annoteringer.
        Prøv enten: 
            - At annotér nogle flere billeder
            - Eller sæt 'n_train' og/eller 'n_val' mindre
    """

    data_train = GaveDatasæt(annotations[:n_train], augs=augmenteringer)
    data_valid = GaveDatasæt(annotations[n_train:n_train + n_val])
    dkwargs = dict(
        num_workers=2, 
        batch_size=4,
        persistent_workers=True,
    )
    loader_train = torch.utils.data.DataLoader(data_train, shuffle=True, **dkwargs)
    loader_valid = torch.utils.data.DataLoader(data_valid, shuffle=False, **dkwargs)

    model = GaveModel(lr=lr, pretrained=start_fra_prætrænede_vægte, freeze=frys_backbone)

    logger = CSVLogger('logs')

    logger.log_hyperparams(dict(
        epochs=epochs, lr=lr, start_fra_prætrænede_vægte=start_fra_prætrænede_vægte,
        frys_backbone=frys_backbone, augmenteringer=augmenteringer,
        n_train=n_train, n_val=n_val,
    ))

    har_gpu = torch.cuda.is_available()
    trainer = pl.Trainer(
        accelerator='gpu' if har_gpu else 'cpu',
        logger=logger,
        max_steps=len(loader_train) * epochs,
        enable_model_summary=False,
        enable_checkpointing=False,
        log_every_n_steps=len(loader_train),
        callbacks=[
            LearningRateMonitor(),
            utils.TQDMPBarNoVal(),
        ],
    )

    trainer.fit(model, loader_train, loader_valid)
    return model, logger.version