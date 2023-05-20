import subprocess
import shutil


def run(path_to_model, path_to_copy, n_steps):

    steps = [10]
    sigmas = [1.5, 2.0, 3.0, 3.5]
    for sigma in sigmas:
        for n_steps in steps:
            print(n_steps)
            print('===============================================================================================================================')
            print(f'===================GENERATION STARTED using {path_to_model}===================')
            print(f'===================STEPS {n_steps}, SIGMA {sigma}===================')
            subprocess.call(f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=7 edm/generate.py --outdir=fid-tmp --seeds=50000-99999 --subdirs \
                --network={path_to_model} --network_copy={path_to_copy} --sigma_max={sigma} --steps={n_steps}", shell=True)

            print('===================FID CALCULATION===================')
            print('====================================')
            print('Final FID')
            subprocess.call(f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=7 edm/fid.py calc --images=fid-tmp/final \
                --ref=$INPUT_PATH/ffhq-64x64.npz \
                --ref_inc=edm/inception-2015-12-05.pkl", shell=True)
            print('====================================')

            for i in range(n_steps + 1):
                print('====================================')
                print(f'x0_{i} FID')
                subprocess.call(f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=7 edm/fid.py calc --images=fid-tmp/x0_{i} \
                    --ref=$INPUT_PATH/ffhq-64x64.npz \
                    --ref_inc=edm/inception-2015-12-05.pkl", shell=True)
                print('====================================')
            print(
                '===============================================================================================================================')


            shutil.rmtree('fid-tmp')