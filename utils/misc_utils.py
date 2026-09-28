import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import shutil


def print_options(parser, opt, logger):
    """
    Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    taken from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/options/base_options.py
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    save_dir = Path(logger.log_dir)
    save_dir.mkdir(parents=True,exist_ok=True)
    file_name =  save_dir / '{}_opt.txt'.format(opt.mode)
    with open(str(file_name), 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')
    pass

class modReLU(nn.Module):
    def __init__(self, hidden_size, bounded=False):
        super(modReLU, self).__init__()
        self.relu = nn.ReLU()
        self.bias = torch.nn.Parameter(data=torch.full((1, hidden_size, 1, 1), -0.1), requires_grad=True)
        self.bounded = bool(bounded)

    def forward(self, x):
        if not self.bounded:
            return x * self.relu(torch.abs(x) + self.bias) / (torch.abs(x) + 1e-3)
        r = torch.abs(x)
        gain = (self.relu(r + self.bias) / (r + 1e-3)).clamp(max=1.0)
        return x * gain

def mriAdjointOp(rawdata, sens, mask):
    """ Adjoint operation that convert kspace to coil-combined under-sampled image """
    coil_sens = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(rawdata * mask), norm="ortho"))
    img = np.sum(coil_sens*np.conj(sens), axis=1)
    return img
    

def mri_adjoint_op(f, coil_sens):
    """ Adjoint operation that convert kspace to coil-combined under-sampled image """
    Finv = ifftc2d(f)  # NxVxCxTxDxHxW
    coil_sens = coil_sens.unsqueeze(1).unsqueeze(3)  # Nx1xCx1xDxHxW
    img = torch.sum(Finv * torch.conj(coil_sens), 2)  # NxVx1xTxDxHxW 
    return img


def mri_forward_op(u, coil_sens, sampling_mask): 
    """ Forward pass with kspace """
    coil_imgs = u.unsqueeze(2) * coil_sens.unsqueeze(1).unsqueeze(3)  # NxVxCxTxDxHxW
    Fu = fftc2d(coil_imgs)  # NxVxCxTxDxHxW
    kspace = sampling_mask.unsqueeze(2).unsqueeze(4) * Fu  # NxVxCxTxDxHxW
    return kspace

def fftc2d(x):
    x = torch.fft.ifftshift(x, dim=(-2,-1))
    x = torch.fft.fft2(x, dim=(-2, -1), norm="ortho") 
    x = torch.fft.fftshift(x, dim=[-1,-2])
    return x


def ifftc2d(x):
    x = torch.fft.ifftshift(x, dim=(-2,-1))
    x = torch.fft.ifft2(x, dim=(-2, -1), norm="ortho") 
    x = torch.fft.fftshift(x, dim=[-1,-2])
    return x

def simulate_undersampling(ky, kz, T, V, rng=None, target_acc=None):
    """
    Generate a Gaussian-weighted random undersampling mask for the ky-kz plane.
    Returns float32 mask of shape (V, T, ky, kz), same pattern broadcast across V and T.
    target_acc: R value (10/20/30/40/50); if None, randomly chosen.
    """
    if rng is None:
        rng = np.random.RandomState()
    if target_acc is None:
        target_acc = rng.choice([10, 20, 30, 40, 50])

    target_density = 1.0 / target_acc
    n_sample = max(1, int(round(ky * kz * target_density)))

    cy, cz = ky // 2, kz // 2
    acs_ky = max(1, ky // 16)
    acs_kz = max(1, kz // 16)
    acs_mask = np.zeros((ky, kz), dtype=bool)
    acs_mask[cy - acs_ky:cy + acs_ky + 1, cz - acs_kz:cz + acs_kz + 1] = True
    n_acs = int(np.sum(acs_mask))

    yy, zz = np.meshgrid(np.arange(ky) - cy, np.arange(kz) - cz, indexing='ij')
    prob = np.exp(-(yy ** 2 / (2 * (ky / 6.0) ** 2) + zz ** 2 / (2 * (kz / 6.0) ** 2)))
    prob[acs_mask] = 0.0
    prob_sum = prob.sum()

    mask_2d = acs_mask.astype(np.float32)
    n_random = max(0, n_sample - n_acs)
    if n_random > 0 and prob_sum > 0:
        prob /= prob_sum
        n_avail = int((prob > 0).sum())
        flat_idx = rng.choice(ky * kz, size=min(n_random, n_avail), replace=False, p=prob.ravel())
        mask_2d.ravel()[flat_idx] = 1.0

    return np.broadcast_to(mask_2d[np.newaxis, np.newaxis], (V, T, ky, kz)).copy()


def empty_or_create(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # If the folder exists, empty it
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        # If the folder does not exist, create it
        os.makedirs(folder_path)

def pad_array(img,pad_width,value=0):
    """
    Allows to pad or box-crop an array depending on pad_width (positive values zero-pad
    and negative values crop)

    Args:
        img (ndarray): image to zero-pad or crop.
        value (double): value to use for padding.
        pad_width (int or tuple): number of elements to pad or crop from.
            Examples of pad_width:
                pad_width = 6 --> pads array symmetrically along all directions with 2 x 6 elements with value
                pad_width = -6 --> crops array symmetrically along all directions by 2 x 6 elements
                pad_width = ((3,5),(7,7)) where len(pad_width) must be = img.ndim --> pads by 3 + 5 elements
                    along axis 1 and 7 + 7 elements along axis 2  

    Returns:
        out (ndarray): Padded or cropped matrix.
    """
    def duplicate_tuple_elements(input_tuple):
        duplicated_elements = []
        for element in input_tuple:
            duplicated_element = (element, element)
            duplicated_elements.append(duplicated_element)
        return tuple(duplicated_elements)

    if isinstance(pad_width,int):
        if pad_width>=0:
            return np.pad(img,pad_width)
        else:
            pad_width = duplicate_tuple_elements(np.tile(-pad_width,img.ndim))
            reversed_padding = tuple([slice(start_pad, dim - end_pad) for ((start_pad, end_pad), dim) in zip(pad_width, img.shape)])
            return img[reversed_padding]
    else:
        if not isinstance(pad_width,tuple):
            raise Exception('pad_width needs to be a tuple')

        if not len(pad_width)==img.ndim:
            raise Exception('pad_width needs to have same dimensions as image')

        if all(i >= 0 for i in pad_width):
            pad_width=duplicate_tuple_elements(pad_width)
            return np.pad(img,pad_width)
        elif all(i < 0 for i in pad_width):
            pad_width=duplicate_tuple_elements(tuple([-i for i in np.array(pad_width)]))
            reversed_padding = tuple([slice(start_pad, dim - end_pad) for ((start_pad, end_pad), dim) in zip(pad_width, img.shape)])
            return img[reversed_padding]
        else:
            raise Exception('All value element must be >0 OR <0')
        