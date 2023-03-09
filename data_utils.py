import torch
import PIL
import shutil
import os

from edm import dnnlib
from edm.torch_utils import misc
from edm.torch_utils import distributed as dist
from edm.fid import calc


# ----------------------------------------------------------------------------
def make_dataset(path,
                 batch_size=64):
    dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset',
                                     path=path,
                                     use_labels=False,
                                     xflip=False,
                                     cache=True)

    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)  # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj,
                                           rank=dist.get_rank(),
                                           num_replicas=dist.get_world_size())

    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj,
                                                        sampler=dataset_sampler,
                                                        batch_size=batch_size))

    return dataset_iterator
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
def calc_fid(path_to_images,
             ref_type,
             num_expected,
             batch,
             first):
    main_path = 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs'
    ref_statistics = {
        'cifar10': 'cifar10-32x32.npz',
        'ffhq': 'ffhq-64x64.npz',
        'imagenet': 'imagenet-64x64.npz',
        'afhq': 'afhqv2-64x64.npz'
    }

    ref_path = f'{main_path}/{ref_statistics[ref_type]}'
    fid = calc(path_to_images, ref_path, num_expected, 0, batch, first)

    return fid
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
@torch.no_grad()
def save_batch(images, path):
    """
    :param images: (Tensor), [b_size, C, W, H], tensor in range [-1, 1]
    :return:
    """
    b_size = images.size(0)
    img_res = images.size(2)
    img_c = images.size(1)
    grid_size = int(b_size ** 0.5)

    image = (images * 127.5 + 128).clip(0, 255).to(torch.uint8)
    image = image.reshape(grid_size, grid_size, *image.shape[1:]).permute(0, 3, 1, 4, 2)
    image = image.reshape(grid_size * img_res,
                          grid_size * img_res,
                          img_c)

    image = image.cpu().numpy()
    PIL.Image.fromarray(image, 'RGB').save(path)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def delete_and_create_dir(path):

    if os.path.isdir(path):
        shutil.rmtree(path)

    os.makedirs(path)
# ----------------------------------------------------------------------------
