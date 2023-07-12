import subprocess
import shutil
import sys
import os

SOURCE_CODE_PATH = os.environ['SOURCE_CODE_PATH']
INPUT_PATH = os.environ['INPUT_PATH']

sys.path.append(f'{SOURCE_CODE_PATH}/EffDiff/ClipModel')
sys.path.append(f'{SOURCE_CODE_PATH}/EffDiff/ClipModel/clip')
sys.path.append(f'{SOURCE_CODE_PATH}/EffDiff/edm')
sys.path.append(f'{SOURCE_CODE_PATH}/EffDiff/guided-diffusion-main')
sys.path.append(f'{SOURCE_CODE_PATH}/EffDiff')
sys.path.append(f'{SOURCE_CODE_PATH}/EffDiff/consistency_models_main/')
sys.path.append(f'{SOURCE_CODE_PATH}/EffDiff/consistency_models_main/cm/')
print(sys.path)


def run(path_to_model, path_to_copy, n_steps):

    path = f'{INPUT_PATH}/AccDDIF_sota_ffhq/ultramar_exp_estimate/data_cifar/imagenet_2_cons.zip'

    steps = [80, 100, 126, 256]
    sigmas = [5.5]
    for n_steps in steps:
        for sigma in sigmas:
            print(n_steps)
            print('===============================================================================================================================')
            print(f'===================GENERATION STARTED using {path_to_model} with seed 50000-99999===================')
            print(f'===================STEPS {n_steps}, SIGMA {sigma}===================')
            subprocess.call(f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 edm/generate.py --outdir=fid-tmp --seeds=50000-99999 --subdirs \
                --network={path_to_model} --network_copy={path_to_copy} --sigma_max={sigma} --steps={n_steps} \
                --path={path}", shell=True)

            print('===================FID CALCULATION===================')
            print('====================================')
            print('Final FID')
            subprocess.call(f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 edm/fid.py calc --images=fid-tmp/final \
                --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-64x64.npz \
                --ref_inc=edm/inception-2015-12-05.pkl", shell=True)
            print('====================================')

            for i in [n_steps - 1]:
                print('====================================')
                print(f'x0_{i} FID')
                subprocess.call(f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 edm/fid.py calc --images=fid-tmp/x0_{i} \
                    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-64x64.npz \
                    --ref_inc=edm/inception-2015-12-05.pkl", shell=True)
                print('====================================')
                #print(f'guided FID')
                #subprocess.call(f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 guided-diffusion-main/evaluations/evaluator.py $INPUT_PATH/VIRTUAL_imagenet64_labeled.npz fid-tmp/array.npz", shell=True)
                #print('====================================')
            print(
                '===============================================================================================================================')
            shutil.rmtree('fid-tmp')

def run2(edm_path, cons_path, n_steps):

    path = f'{INPUT_PATH}/AccDDIF_sota_ffhq/ultramar_exp_estimate/data_cifar/imagenet_4_cd.zip'
    edm_path = f'{INPUT_PATH}/AccDDIF_sota_ffhq/ultramar_exp_estimate/data_cifar/edm_cat256_ema.pt'
    cons_path = f'{INPUT_PATH}/AccDDIF_sota_ffhq/ultramar_exp_estimate/data_cifar/cd_cat256_lpips.pt'
    path_to_ref = f'{INPUT_PATH}/AccDDIF_sota_ffhq/ultramar_exp_estimate/data_cifar/VIRTUAL_lsun_cat256.npz'

    steps = [40]
    sigmas = [5.0, 6.0, 7.0, 8.0, 9.0, 10, 11, 12, 13, 14, 15]
    for n_steps in steps:
        for sigma in sigmas:
            print(n_steps)
            print('===============================================================================================================================')
            print(f'===================GENERATION STARTED using {edm_path} with seed 0-49999===================')
            print(f'===================STEPS {n_steps}, SIGMA {sigma}===================')
            subprocess.call(f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 edm/generate2.py \
                            --outdir=fid-tmp --seeds=0-4999 --subdirs \
                            --edm_path={edm_path} --cons_path={cons_path} --sigma_max={sigma} --steps={n_steps} \
                            --path={path}", shell=True)

            print('===================FID CALCULATION===================')
            print('====================================')
            print('Final FID')
            subprocess.call(f"CUDA_VISIBLE_DEVICES=7 python3 guided-diffusion-main/evaluations/evaluator.py \
                            {path_to_ref} fid-tmp/array.npz", shell=True)
            print('====================================')

            #for _ in [n_steps - 1]:
            #    print('====================================')
            #    print(f'guided FID')
            #    subprocess.call(f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 guided-diffusion-main/evaluations/evaluator.py \
            #                    {path_to_ref} fid-tmp/array.npz", shell=True)
            #    print('====================================')

            print(
                '===============================================================================================================================')
            shutil.rmtree('fid-tmp')

run2(edm_path=f'{INPUT_PATH}/edm_cat256_ema.pt',
     cons_path=f'{INPUT_PATH}/cd_imagenet64_lpips.pt',
     n_steps=0)

#run(path_to_copy=f'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl',
#    n_steps=100500,
#    path_to_model=f'{INPUT_PATH}/AccDDIF_sota_ffhq/ultramar_exp_estimate/edm_5_steps_first_ord_hard.pkl')


