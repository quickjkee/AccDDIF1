import sys
import os

SOURCE_CODE_PATH = os.environ['SOURCE_CODE_PATH']
INPUT_PATH = os.environ['INPUT_PATH']

sys.path.append(f'{SOURCE_CODE_PATH}/EffDiff/ClipModel')
sys.path.append(f'{SOURCE_CODE_PATH}/EffDiff/ClipModel/clip')
sys.path.append(f'{SOURCE_CODE_PATH}/EffDiff/edm')
sys.path.append(f'{SOURCE_CODE_PATH}/EffDiff')
print(sys.path)

import torch
import pickle

from fine_tune import FineTuner
from clip_model import DirectionLoss
from DiffModels.KarrasEDM import DiffModel
from data_utils import make_dataset, delete_and_create_dir

devices = torch.device('cuda')
print(f'All devices {devices}')
print(torch.cuda.device_count())


device = torch.device('cuda:0')
print(f'Working device {device}')

# Load the network
network_pkl = f'{INPUT_PATH}/edm-ffhq-64x64-uncond-vp.pkl'

print(f'Loading network from "{network_pkl}"...')
with open(network_pkl, 'rb') as handle:
    net = pickle.load(handle)['ema'].to(device)

##############################
#
# CONFIGURATIONS
#
##############################

# Dataset
b_size = 64
path_to_data = 'datasets/ffhq-64x64.zip'
dataset = make_dataset(path_to_data, batch_size=b_size)

# Models
model = DiffModel(net=net,
                  num_steps=15,
                  device=device)
clip = DirectionLoss(device=device)
tuner = FineTuner(model=model,
                  clip=clip,
                  dataset=dataset,
                  device=device,
                  batch_size=b_size,
                  n_iters=2000,
                  lr=9e-6)

# RUNNING
delete_and_create_dir('runs')

tuner.fine_tune()


#WAS lr 9e-6, num_steps=40