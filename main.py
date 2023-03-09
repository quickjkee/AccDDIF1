import torch
import pickle

from fine_tune import FineTuner
from clip_model import DirectionLoss
from DiffModels.KarrasEDM import DiffModel
from edm import dnnlib
from data_utils import make_dataset, delete_and_create_dir

device = torch.device('cuda')

# Load the network
model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'
network_pkl = f'{model_root}/edm-ffhq-64x64-uncond-vp.pkl'

with dnnlib.util.open_url(network_pkl) as f:
    net = pickle.load(f)['ema'].to(device)

##############################
#
# CONFIGURATIONS
#
##############################

# Dataset
path_to_data = '../../Desktop/AccelerationDiff/datasets/ffhq-64x64.zip'
dataset = make_dataset(path_to_data, batch_size=64)

# Models
model = DiffModel(net=net,
                  num_steps=40,
                  device=device)
clip = DirectionLoss(device=device)
tuner = FineTuner(model=model,
                  clip=clip,
                  dataset=dataset,
                  device=device,
                  n_iters=50,
                  lr=9e-6)

# RUNNING
delete_and_create_dir('runs')

tuner.fine_tune()

