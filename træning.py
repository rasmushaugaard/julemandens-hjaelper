import warnings
import logging

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor

import utils
from data import GaveDatasæt
from model import GaveModel


logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=pl.utilities.warnings.PossibleUserWarning)


def træn_model(
    epoker: int,
    start_fra_prætrænede_vægte: bool,
    augmenteringer: list, 
    antal_træningsbilleder: int,
    model_størrelse: str, 
    frys_backbone: bool = False,
) -> (GaveModel, int):
    learning_rate = 0.0001
    antal_valideringsbilleder = 100
    annotations = utils.load_annotations()
    assert antal_træningsbilleder + antal_valideringsbilleder <= len(annotations), """
        Antal træningsbilleder + antal valideringsbilleder må ikke være større end antallet af annoterede billeder.
        Prøv enten: 
            - At annotér nogle flere billeder
            - Eller sæt 'antal_træningsbilleder' mindre
    """

    anno_valid = annotations[:antal_valideringsbilleder]
    anno_train = annotations[antal_valideringsbilleder:antal_valideringsbilleder + antal_træningsbilleder]

    data_train = GaveDatasæt(anno_train, augs=augmenteringer)
    data_valid = GaveDatasæt(anno_valid)
    dkwargs = dict(
        num_workers=2, 
        batch_size=4,
        persistent_workers=True,
    )
    loader_train = torch.utils.data.DataLoader(data_train, shuffle=True, **dkwargs)
    loader_valid = torch.utils.data.DataLoader(data_valid, shuffle=False, **dkwargs)

    model = GaveModel(lr=learning_rate, pretrained=start_fra_prætrænede_vægte, 
                      freeze=frys_backbone, størrelse=model_størrelse)

    logger = CSVLogger('logs')
    logger.log_hyperparams(dict(
        epoker=epoker, start_fra_prætrænede_vægte=start_fra_prætrænede_vægte,
        frys_backbone=frys_backbone, augmenteringer=list(map(str, augmenteringer)),
        antal_træningsbilleder=antal_træningsbilleder, model_størrelse=model_størrelse,
    ))

    har_gpu = torch.cuda.is_available()
    trainer = pl.Trainer(
        accelerator='gpu' if har_gpu else 'cpu',
        logger=logger,
        max_steps=len(loader_train) * epoker,
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