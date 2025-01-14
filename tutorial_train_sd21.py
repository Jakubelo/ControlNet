from share import *
import argparse
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch.utils.data import DataLoader
from s2s_dataset import S2sDataSet
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
import hydra
import torch


@hydra.main(
    config_path=str(Path.cwd()), config_name='config', version_base=None
)
def main(cfg):
    pl.seed_everything(cfg.seed)
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(cfg.model).cpu()
    if not Path(cfg.model_preparation.init_controlnet_model).is_file():
        raise FileNotFoundError(
            f'Please ensure valid path to model, {cfg.model_preparation.init_controlnet_model} not found.'
        )
    model.load_state_dict(
        load_state_dict(
            cfg.model_preparation.init_controlnet_model, location='cpu'
        )
    )
    model.learning_rate = cfg.train.learning_rate
    model.sd_locked = cfg.train.sd_locked
    model.only_mid_control = cfg.train.only_mid_control

    # Misc
    train_dataset = S2sDataSet(
        Path(cfg.train.train_input_dir), (cfg.train.size.x, cfg.train.size.y)
    )
    val_dataset = S2sDataSet(
        Path(cfg.train.val_input_dir), (cfg.train.size.x, cfg.train.size.y)
    )
    train_dataloader = DataLoader(
        train_dataset,
        num_workers=0,
        batch_size=cfg.train.batch_size,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        num_workers=0,
        batch_size=cfg.train.batch_size,
        shuffle=True,
    )
    logger = ImageLogger(batch_frequency=cfg.train.logger_freq)
    checkpoint_callback = ModelCheckpoint(every_n_epochs=2, save_top_k=-1)
    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        accelerator='gpu',
        precision=32,
        callbacks=[logger, checkpoint_callback],
        devices=torch.cuda.device_count(),
    )

    # Train!
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == '__main__':
    main()
