import subprocess


def run(path_to_model='$INPUT_PATH/edm-ffhq-64x64-uncond-vp.pkl'):

    print('===============================================================================================================================')
    print(f'===================GENERATION STARTED using {path_to_model}===================')
    subprocess.call(f"CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=7 generate.py --outdir=fid-tmp --seeds=100000-149999 --subdirs \
        --network={path_to_model}", shell=True)


    print('===================FID CALCULATION===================')
    print('====================================')
    print('Final FID')
    subprocess.call(f"CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=7 fid.py calc --images=fid-tmp/final \
        --ref=$INPUT_PATH/ffhq-64x64.npz \
        --ref_inc=inception-2015-12-05.pkl", shell=True)
    print('====================================')

    for i in range(40):
        print('====================================')
        print(f'x0_{i} FID')
        subprocess.call(f"CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=7 fid.py calc --images=fid-tmp/x0_{i} \
            --ref=$INPUT_PATH/ffhq-64x64.npz \
            --ref_inc=inception-2015-12-05.pkl", shell=True)
        print('====================================')
    print(
        '===============================================================================================================================')

run()
