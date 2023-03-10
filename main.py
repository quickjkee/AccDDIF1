import sys
sys.path.append('$SOURCE_CODE_PATH:/EffDiff/ClipModel')
sys.path.append('$SOURCE_CODE_PATH:/EffDiff/ClipModel/clip')
sys.path.append('$SOURCE_CODE_PATH:/EffDiff/edm')
print(sys.path)

import torch
import pickle

from fine_tune import FineTuner
from clip_model import DirectionLoss
from DiffModels.KarrasEDM import DiffModel
from data_utils import make_dataset, delete_and_create_dir

devices = torch.device('cuda')
print(f'All devices {devices}')

device = torch.device('cuda:0')
print(f'Working device {device}')

# Load the network
network_pkl = '$INPUT_PATH/edm-ffhq-64x64-uncond-vp.pkl'

print(f'Loading network from "{network_pkl}"...')
with open(network_pkl, 'rb') as handle:
    net = pickle.load(handle)['ema'].to(device)

##############################
#
# CONFIGURATIONS
#
##############################

# Dataset
path_to_data = 'datasets/ffhq-64x64.zip'
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

