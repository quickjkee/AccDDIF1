import subprocess
import shutil
import sys
import os

SOURCE_CODE_PATH = os.environ['SOURCE_CODE_PATH']
INPUT_PATH = os.environ['INPUT_PATH']


def run(path_to_model, path_to_copy, n_steps):

    path = f'{INPUT_PATH}/AccDDIF_sota_ffhq/ultramar_exp_estimate/data_cifar/out'
    steps = [10, 15, 20, 25, 30, 40]
    sigmas = [0.5, 0.6, 0.8, 1.0, 1.5]
    for n_steps in steps:
        for sigma in sigmas:
            print(n_steps)
            print('===============================================================================================================================')
            print(f'===================GENERATION STARTED using {path_to_model}===================')
            print(f'===================STEPS {n_steps}, SIGMA {sigma}===================')
            subprocess.call(f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 edm/generate.py --outdir=fid-tmp --seeds=150000-199999 --subdirs \
                --network={path_to_model} --network_copy={path_to_copy} --sigma_max={sigma} --steps={n_steps} \
                --path={path}", shell=True)

            print('===================FID CALCULATION===================')
            print('====================================')
            print('Final FID')
            subprocess.call(f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 edm/fid.py calc --images=fid-tmp/final \
                --ref=$INPUT_PATH/ffhq-64x64.npz \
                --ref_inc=edm/inception-2015-12-05.pkl", shell=True)
            print('====================================')

            for i in range(n_steps + 1):
                print('====================================')
                print(f'x0_{i} FID')
                subprocess.call(f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 edm/fid.py calc --images=fid-tmp/x0_{i} \
                    --ref=$INPUT_PATH/ffhq-64x64.npz \
                    --ref_inc=edm/inception-2015-12-05.pkl", shell=True)
                print('====================================')
            print(
                '===============================================================================================================================')


            shutil.rmtree('fid-tmp')