import subprocess

print('===================GENERATION STARTED===================')
subprocess.call("torchrun --standalone --nproc_per_node=8 generate.py --outdir=fid-tmp --seeds=100000-149999 --subdirs \
    --network=$INPUT_PATH/edm-ffhq-64x64-uncond-vp.pkl", shell=True)


print('===================FID CALCULATION===================')
print('====================================')
print('Final FID')
subprocess.call(f"torchrun --standalone --nproc_per_node=8 fid.py calc --images=fid-tmp/final \
    --ref=$INPUT_PATH/ffhq-64x64.npz \
    --ref_inc=inception-2015-12-05.pkl", shell=True)
print('====================================')