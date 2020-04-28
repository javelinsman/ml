from models import TTSModel
from callbacks.logging import LoggingCallback
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

def train(model: pl.LightningModule, epochs, experiment_name, log_dir='./tb_logs'):
    logger = pl.loggers.TensorBoardLogger(log_dir, name=experiment_name)
    trainer = pl.Trainer(
        max_nb_epochs=epochs,
        gpus='0',
        logger=logger,
        callbacks=[LoggingCallback()]
    )
    trainer.fit(model)
    return trainer, model