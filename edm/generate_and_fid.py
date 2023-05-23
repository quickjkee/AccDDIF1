import subprocess
import shutil


def run(path_to_model, path_to_copy, n_steps):

    steps = [3, 6, 8, 10, 15, 20, 25, 30, 37]
    sigmas = [0.13, 1.5, 1.8, 3.0, 3.0, 2.7, 3.0, 2.8, 2.8]
    for _ in [0]:
        for sigma, n_steps in zip(sigmas, steps):
            print(n_steps)
            print('===============================================================================================================================')
            print(f'===================GENERATION STARTED using {path_to_model}===================')
            print(f'===================STEPS {n_steps}, SIGMA {sigma}===================')
            subprocess.call(f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 edm/generate.py --outdir=fid-tmp --seeds=149999-199999 --subdirs \
                --network={path_to_model} --network_copy={path_to_copy} --sigma_max={sigma} --steps={n_steps}", shell=True)

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