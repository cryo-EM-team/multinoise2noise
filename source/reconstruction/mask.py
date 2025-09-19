import torch
from tqdm import tqdm
import numpy as np


def auto_mask(img_in: torch.Tensor, 
              ini_mask_density_threshold: float, 
              extend_ini_mask_list: list[int], 
              width_soft_mask_edge: int) -> torch.Tensor:
    """
    Function that computes mask for a given 3D electron map.
    
    Args:
        img_in (torch.Tensor): Input 3D tensor (image).
        ini_mask_density_threshold (float): Initial mask density threshold.
        extend_ini_mask (list[float]): Valuea to extend or shrink the mask one after another.
        width_soft_mask_edge (float): Width of the soft mask edge.
    
    Returns:
        torch.Tensor: Output mask.
    """
    # A. Calculate initial binary mask based on density threshold
    print("Thresholding ...")
    msk_out = (img_in >= ini_mask_density_threshold).to(img_in.dtype)
    print("... Done")

    # B. Extend/shrink initial binary mask using pooling
    print("Extending...")
    for extend_ini_mask in tqdm(extend_ini_mask_list):
        if extend_ini_mask != 0.0:
            if extend_ini_mask > 0.0:
                msk_out = expand_ones(msk_out, extend_ini_mask)
            else:
                # Min pooling to shrink the mask (invert, max pool, then invert back)
                msk_out = 1.0 - msk_out
                msk_out = expand_ones(msk_out, -extend_ini_mask)
                msk_out = 1.0 - msk_out

    print("... Done")
    # C. Make a soft edge to the mask (unchanged from the original implementation)
    print("Softening Edges...")
    if width_soft_mask_edge > 0.0:
        msk_out = expand_soft(msk_out, width_soft_mask_edge)
    print("... Done")
    return msk_out

def expand_ones(tensor: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Expands the values of 1 in a 3D tensor to their neighbors within a given radius.

    Args:
        tensor (torch.Tensor): A 3D tensor containing mostly 0s and some 1s.
        radius (int): The radius within which to expand the values of 1.

    Returns:
        torch.Tensor: A new tensor with the expanded values of 1.
    """
    spherical_kernel = torch.stack(torch.meshgrid(
        torch.arange(-radius, radius+1, dtype=tensor.dtype),
        torch.arange(-radius, radius+1, dtype=tensor.dtype),
        torch.arange(-radius, radius+1, dtype=tensor.dtype),
        indexing="ij"
    ), dim=-1)
    spherical_kernel = ((spherical_kernel ** 2).sum(-1) < (radius ** 2)).to(tensor.dtype)

    # Apply the kernel as a convolution
    spherical_kernel = spherical_kernel.to(tensor.device).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, D, H, W)
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # Perform 3D convolution
    expanded_tensor = torch.nn.functional.conv3d(tensor, spherical_kernel, padding=radius)
    expanded_tensor = (expanded_tensor > 0).to(tensor.dtype)  # Threshold to keep only 0s and 1s

    return expanded_tensor.squeeze(0).squeeze(0)

def padded_roll(data: torch.Tensor, shifts: tuple[int], value: int = 0) -> torch.Tensor:
    """
    Rolls a tensor along specified dimensions, padding with zeros instead of wrapping values.

    Args:
        data (torch.Tensor): The input tensor.
        shifts (tuple[int]): The number of positions to shift for each dimension. Can be positive or negative.

    Returns:
        torch.Tensor: The rolled tensor with zeros appended.
    """
    pad = [0] * (2 * data.ndim)

    # Apply padding based on shifts
    for dim, shift in enumerate(shifts):
        if shift > 0:
            pad[-(2 * dim + 1)] = shift  # Pad after
        elif shift < 0:
            pad[-(2 * dim + 2)] = -shift  # Pad before

    # Pad the data
    padded_data = torch.nn.functional.pad(data, pad, mode='constant', value=value)

    # Roll the padded data
    rolled_data = padded_data.roll(shifts, dims=tuple(range(len(shifts))))

    # Slice the rolled data to remove padding
    slices = []
    for dim, shift in enumerate(shifts):
        if shift > 0:
            slices.append(slice(0, -shift))
        elif shift < 0:
            slices.append(slice(-shift, None))
        else:
            slices.append(slice(None))  # No shift for this dimension

    # Add slices for remaining dimensions (if any)
    for _ in range(len(shifts), data.ndim):
        slices.append(slice(None))

    return rolled_data[tuple(slices)]

def expand_soft(tensor: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Computes the distance to the closest voxel with value 1 for each voxel in a 3D tensor within a given radius.
    Optimized by using kernel-based operations with strides.

    Args:
        tensor (torch.Tensor): A 3D tensor containing mostly 0s and some 1s.
        radius (int): The radius within which to search for the closest voxel with value 1.

    Returns:
        torch.Tensor: A 3D tensor where each voxel contains the distance to the closest voxel with value 1.
    """
    distances = torch.full_like(tensor, fill_value=float("inf"), dtype=tensor.dtype)

    spherical_kernel = torch.stack(torch.meshgrid(
        torch.arange(-radius, radius+1, dtype=tensor.dtype),
        torch.arange(-radius, radius+1, dtype=tensor.dtype),
        torch.arange(-radius, radius+1, dtype=tensor.dtype),
        indexing="ij"
    ), dim=-1)
    indices = spherical_kernel.flatten(start_dim=0, end_dim=-2).int()
    spherical_kernel = (spherical_kernel ** 2).sum(-1).to(tensor.dtype)
    spherical_mask = spherical_kernel < (radius**2)
    spherical_mask[radius, radius, radius] = False
    indices = indices[spherical_mask.flatten()]
    spherical_kernel = torch.sqrt(spherical_kernel).flatten()[spherical_mask.flatten()]
    tensor_inf = tensor + (tensor == 0) * (radius+1)
    for (dz, dy, dx), dist in tqdm(zip(indices, spherical_kernel), total=spherical_kernel.shape[0]):
        new_distances = padded_roll(dist * tensor_inf, shifts=(dz, dy, dx), value=float("inf"))
        distances = torch.minimum(distances, new_distances)
    output = tensor + (tensor == 0) * (distances < radius+1) * (0.5 + 0.5 * torch.cos(np.pi * distances / radius))
    return output

