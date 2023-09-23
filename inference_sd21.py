from share import *

import pytorch_lightning as pl
import torch
import numpy as np
from torch.utils.data import DataLoader
from s2s_dataset import PredictS2sDataSet
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from cldm.cldm import ControlLDM
from pathlib import Path
from PIL import Image


# Configs
resume_path = '/home/jgalik/repos/S2S/lightning_logs/version_0/checkpoints/epoch=21-step=163724.ckpt'
input_directory = Path('/home/jgalik/repos/ControlNet/dataset/preprocessed/test2/')
output_path = Path('inference')
batch_size = 2
only_mid_control = False
size = 512, 512


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model: ControlLDM = create_model('src/ControlNet/models/cldm_v21.yaml').cuda()
model.load_state_dict(load_state_dict(resume_path, location='cuda'))
model.eval()
model.only_mid_control = only_mid_control


# Misc
dataset = PredictS2sDataSet(input_directory, size)
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
output_path.mkdir(exist_ok=True)
for batch in dataloader:
    images = model.log_images(batch, N=batch_size, n_row=1, ddim_steps=50)
    images = images['samples_cfg_scale_9.00'].detach().cpu()
    images = torch.clamp(images, -1., 1.)
    images = (images + 1.0) / 2.0
    for idx, img in enumerate(images):
        img = img.permute(1,2,0)
        img = (img.numpy() * 255).astype(np.uint8)
        Image.fromarray(img).save(output_path / batch['filename'][idx])
