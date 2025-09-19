import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from source.reconstruction.utils import real_round, calculate_fsc, create_fourier_distance_map, calculate_resolution
from source.reconstruction.symmetry import Symmetry


def determine_randomization_threshold(fsc: torch.Tensor, randomize_fsc_at: float = 0.8) -> torch.Tensor:
    """
    Find the first index where the FSC drops below the randomization threshold.

    Args:
        fsc (torch.Tensor): Fourier Shell Correlation curve.
        randomize_fsc_at (float): Threshold value for randomization.

    Returns:
        int: Index where FSC drops below the threshold, or length of fsc if not found.
    """
    vec = torch.nonzero(fsc < randomize_fsc_at, as_tuple=False)
    if len(vec) > 0:
        return vec[0].item()
    return len(fsc)

def randomize_phases(half_map: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Randomize the phases of a half-map beyond a given radius in Fourier space.

    Args:
        half_map (torch.Tensor): Input half-map (3D tensor).
        radius (int): Radius beyond which phases are randomized.

    Returns:
        torch.Tensor: Half-map with randomized phases beyond the radius.
    """
    f_half_map = torch.fft.rfftn(half_map, norm="forward")
    distance = create_fourier_distance_map(f_half_map.shape)
    mask = (distance >= radius).flatten()
    mag = f_half_map.flatten()[mask].abs()
    phases = torch.rand(size=(mag.shape[0], ), dtype=half_map.dtype) * 2 * np.pi
    f_half_map.view(-1)[mask] = mag * torch.complex(torch.cos(phases), torch.sin(phases))
    return torch.fft.irfftn(f_half_map, norm="forward")

def calculate_fsc_true(fsc_masked: torch.Tensor, fsc_random_masked: torch.Tensor, randomize_at: int) -> torch.Tensor:
    """
    Calculate the corrected FSC curve using masked and randomized FSCs.

    Args:
        fsc_masked (torch.Tensor): FSC of masked maps.
        fsc_random_masked (torch.Tensor): FSC of masked maps with randomized phases.
        randomize_at (int): Index at which randomization starts.

    Returns:
        torch.Tensor: Corrected FSC curve.
    """
    i = 2
    if randomize_at >= fsc_masked.shape[0] - i:
        return fsc_masked
    corected_fsc = (fsc_masked[randomize_at+i:] - fsc_random_masked[randomize_at+i:]) / (1 - fsc_random_masked[randomize_at+i:])
    return torch.cat([fsc_masked[:randomize_at+i], corected_fsc])

def prepare_g_plot_for_fit(ln_f: torch.Tensor, autob_lowres: float, angpix: float, size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare data for Guinier plot fitting.

    Args:
        ln_f (torch.Tensor): Logarithm of Fourier amplitudes.
        autob_lowres (float): Low resolution cutoff.
        angpix (float): Pixel size in Angstroms.
        size (int): Map size.

    Returns:
        tuple: (x, ln_f, w) where x is 1/res^2, ln_f is log amplitudes, w is mask for fitting.
    """
    mask = torch.isfinite(ln_f)
    res = size * angpix / torch.arange(ln_f.shape[0], dtype=torch.double)
    x = 1 / res.square()
    w = mask * (res <= autob_lowres) * (res >= (2 * angpix))
    return x, ln_f, w

def make_guinier_plot(f_density_map: torch.Tensor, autob_lowres: float, angpix: float) -> torch.Tensor:
    """
    Compute data for a Guinier plot from a Fourier density map.

    Args:
        f_density_map (torch.Tensor): Fourier-transformed density map.
        autob_lowres (float): Low resolution cutoff.
        angpix (float): Pixel size in Angstroms.

    Returns:
        tuple: (x, y, w) for Guinier plot fitting.
    """
    r = create_fourier_distance_map(f_density_map.shape).flatten()
    r = real_round(r)
    mask = r < f_density_map.shape[2]
    r = r[mask]
    rings = torch.nn.functional.one_hot(r.long()).double()
    ln_f = f_density_map.flatten()[mask].abs().double() @ rings
    radial_count = torch.ones_like(r) @ rings
    ln_f = torch.log(ln_f / radial_count)
    x, y, w = prepare_g_plot_for_fit(ln_f, autob_lowres, angpix, f_density_map.shape[0])
    return x, y, w

def apply_fsc_weighting(f_density_map: torch.Tensor, fsc: torch.Tensor) -> torch.Tensor:
    """
    Apply FSC-based weighting to a Fourier density map.

    Args:
        f_density_map (torch.Tensor): Fourier-transformed density map.
        fsc (torch.Tensor): FSC curve.

    Returns:
        torch.Tensor: Weighted Fourier density map.
    """
    i = torch.nonzero(fsc <= 0, as_tuple=False)
    if len(i) > 0:
        i = i[0].item()
        fsc[i:] = 0
    weights = (2 * fsc / (1 + fsc)).sqrt()
    
    r = create_fourier_distance_map(f_density_map.shape)
    r = real_round(r.flatten()).int()
    mask = r < f_density_map.shape[2]
    out_map = f_density_map.clone()
    out_map.view(-1)[mask] *= weights[r[mask]]
    return out_map

def fit_straight_line_lstsq(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Fit a straight line using torch.linalg.lstsq.

    Args:
        x (torch.Tensor): 1D tensor of x-coordinates.
        y (torch.Tensor): 1D tensor of y-coordinates.

    Returns:
        slope (float): Slope of the fitted line.
        intercept (float): Intercept of the fitted line.
    """
    # Add a column of ones to x for the intercept
    x = x[w]
    y = y[w].unsqueeze(1)
    A = torch.stack([torch.ones_like(x), x], dim=1)

    # Solve the least squares problem
    solution = torch.linalg.lstsq(A, y).solution
    intercept, slope = solution.squeeze()

    return slope.item(), intercept.item()

def apply_bfactor(f_density_map: torch.Tensor, global_bfactor: float, angpix: float) -> None:
    """
    Apply a global B-factor to a Fourier density map in place.

    Args:
        f_density_map (torch.Tensor): Fourier-transformed density map.
        global_bfactor (float): B-factor value.
        angpix (float): Pixel size in Angstroms.
    """
    r = create_fourier_distance_map(f_density_map.shape)
    r = (r / (f_density_map.shape[0] * angpix))
    mask = r <= (0.5/angpix)
    weights = torch.exp((-global_bfactor / 4) * r.square()) * mask
    f_density_map *= weights

def plot_with_weighted_dashes(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, ax, color: str, label: str, true_style: str = '-', false_style: str = '--'):
    """
    Plot a line with parts styled as solid or dashed based on binary weights.

    Args:
        x (torch.Tensor): X-coordinates of the line.
        y (torch.Tensor): Y-coordinates of the line.
        w (torch.Tensor): Binary weights (0 or 1) for each segment of the line.
        ax: Matplotlib axis to plot on.
        color (str): Line color.
        label (str): Label for the plot.
        true_style (str): Line style for weighted (1) segments.
        false_style (str): Line style for unweighted (0) segments.
    """
    # Ensure inputs are numpy arrays
    x = x.numpy()
    y = y.numpy()
    weights = w.numpy()
    for i in range(len(x) - 1):
        label_ = label if i == len(x)//2 - 1 else None
        ax.plot(x[i:i+2], 
                y[i:i+2], 
                color=color, 
                linestyle=true_style if weights[i] == 1 else false_style, 
                label=label_)

def sharpen_map(density_map: torch.Tensor, fsc: torch.Tensor, angpix: float, autob_lowres: float) -> dict[str, torch.Tensor]:
    """
    Sharpen a density map using FSC weighting and Guinier plot fitting.

    Args:
        density_map (torch.Tensor): Input density map.
        fsc (torch.Tensor): FSC curve.
        angpix (float): Pixel size in Angstroms.
        autob_lowres (float): Low resolution cutoff.

    Returns:
        dict: Contains sharpened map, B-factor, intercept, and Guinier plot data.
    """
    f_density_map = torch.fft.rfftn(density_map, norm="forward")
    guiner = {"Original": make_guinier_plot(f_density_map, autob_lowres, angpix)}
    f_density_map = apply_fsc_weighting(f_density_map, fsc)
    guiner["Weighted"] = make_guinier_plot(f_density_map, autob_lowres, angpix)
    
    slope, intercept = fit_straight_line_lstsq(*guiner["Weighted"])
    b_factor = slope * 4
    apply_bfactor(f_density_map, b_factor, angpix)
    guiner["Sharpened"] = make_guinier_plot(f_density_map, autob_lowres, angpix)
    
    return {"sharp_map": torch.fft.irfftn(f_density_map, norm="forward"), "b_factor": b_factor, "intercept": intercept, "guiner": guiner}

def raised_cosine_mask(mask_shape, radius, radius_p, center_x, center_y, center_z):
    """
    Create a raised cosine mask in PyTorch.

    Args:
        mask_shape (tuple): Shape of the 3D mask (depth, height, width).
        radius (float): Inner radius where the mask value is 1.
        radius_p (float): Outer radius where the mask value is 0.
        center_x (int): X-coordinate of the center.
        center_y (int): Y-coordinate of the center.
        center_z (int): Z-coordinate of the center.

    Returns:
        torch.Tensor: A 3D tensor representing the raised cosine mask.
    """
    # Create a 3D grid of coordinates
    coords = torch.stack(torch.meshgrid(
        torch.arange(mask_shape[0], dtype=torch.double),
        torch.arange(mask_shape[1], dtype=torch.double),
        torch.arange(mask_shape[2], dtype=torch.double),
        indexing='ij'
    ), dim=-1) - torch.Tensor([center_x, center_y, center_z])[None, None, None, :]
    coords = coords.square().sum(-1).sqrt()

    mask = (coords <= radius).double()
    crown = (coords > radius) & (coords <= radius_p)
    mask[crown] = 0.5 - 0.5 * torch.cos(np.pi * (radius_p - coords[crown]) / (radius_p - radius))
    return mask

def low_pass_filter_map(FT, ori_size, low_pass, angpix, filter_edge_width, do_highpass_instead=False):
    """
    Apply a low-pass or high-pass filter to a 3D Fourier-transformed tensor.

    Args:
        FT (torch.Tensor): 3D Fourier-transformed tensor (complex-valued).
        ori_size (int): Original size of the input data.
        low_pass (float): Low-pass filter cutoff frequency (in 1/Angstrom).
        angpix (float): Pixel size (in Angstroms).
        filter_edge_width (int): Width of the soft edge for the filter.
        do_highpass_instead (bool): If True, apply a high-pass filter instead of a low-pass filter.

    Returns:
        torch.Tensor: Filtered 3D Fourier-transformed tensor.
    """
    # Calculate the filter resolution shell
    ires_filter = round((ori_size * angpix) / low_pass)
    filter_edge_halfwidth = filter_edge_width / 2

    # Define the soft edge range
    edge_low = max(0.0, (ires_filter - filter_edge_halfwidth) / ori_size)  # in 1/pixel
    edge_high = min(FT.shape[-1], (ires_filter + filter_edge_halfwidth) / ori_size)  # in 1/pixel
    edge_width = edge_high - edge_low

    # Create a 3D grid of frequencies
    res = create_fourier_distance_map(FT.shape) / ori_size
    
    # Apply the filter
    cos_mask = (res >= edge_low) & (res <= edge_high)
    mask = None
    if do_highpass_instead:
        mask = (res >= edge_low).double()
        mask[cos_mask] = 0.5 - 0.5 * torch.cos(np.pi * (res[cos_mask] - edge_low) / edge_width)
    else:
        mask = (res <= edge_high).double()
        mask[cos_mask] = 0.5 + 0.5 * torch.cos(np.pi * (res[cos_mask] - edge_low) / edge_width)

    return FT * mask

def generate_3d_coordinates(tensor_shape: tuple[int, int, int], step: int) -> torch.Tensor:
    """
    Generate 3D coordinates inside a sphere for a tensor with the given shape.

    Args:
        tensor_shape (tuple): Shape of the tensor (depth, height, width).
        step (int): Step size for spacing between points.
        radius (float): Radius of the sphere.

    Returns:
        torch.Tensor: A tensor of shape [N, 3] containing the 3D coordinates.
    """
    center = torch.tensor([s // 2 for s in tensor_shape], dtype=torch.float32)
    radius = torch.ceil(torch.tensor(tensor_shape[0] - step) / 2)
    
    coords = torch.stack(torch.meshgrid(
        torch.arange(0, tensor_shape[0], step, dtype=torch.float32),
        torch.arange(0, tensor_shape[1], step, dtype=torch.float32),
        torch.arange(0, tensor_shape[2], step, dtype=torch.float32),
        indexing='ij'
    ), dim=-1).reshape(-1, 3)
    distances = (coords - center).square().sum(dim=1).sqrt()
    valid_coords = coords[distances <= radius]
    return valid_coords

def filter_coordinates_by_mask(coordinates: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Filters 3D coordinates based on a given 3D mask.

    Args:
        coordinates (torch.Tensor): A 2D tensor of shape (N, 3) containing 3D coordinates.
        mask (torch.Tensor): A 3D tensor representing the mask (values > 0 indicate valid regions).

    Returns:
        torch.Tensor: A 2D tensor containing only the coordinates that lie within the mask.
    """
    coordinates = coordinates.long()
    z, y, x = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
    valid_mask = mask[x, y, z] > 0
    return coordinates[valid_mask]

def locres(half_1: torch.Tensor, 
           half_2: torch.Tensor, 
           half_1_pr: torch.Tensor, 
           half_2_pr: torch.Tensor, 
           b_factor: float, 
           angpix: float, 
           randomize_at: int,
           mask: torch.Tensor = None,
           mean_map: torch.Tensor = None, 
           locres_sampling: int = 25, 
           filter_edge_width: int = 2,
           symmetry: Symmetry = None) -> dict[str, torch.Tensor]:
    """
    Calculate local resolution and filtered map using local FSC and B-factor correction.

    Args:
        half_1 (torch.Tensor): First half-map.
        half_2 (torch.Tensor): Second half-map.
        half_1_pr (torch.Tensor): First half-map with randomized phases.
        half_2_pr (torch.Tensor): Second half-map with randomized phases.
        b_factor (float): Global B-factor.
        angpix (float): Pixel size in Angstroms.
        randomize_at (int): Index for phase randomization.
        mask (torch.Tensor, optional): Mask to apply.
        mean_map (torch.Tensor, optional): Mean map for sharpening.
        locres_sampling (int, optional): Sampling step for local resolution.
        filter_edge_width (int, optional): Edge width for filtering.
        symmetry (Symmetry, optional): Symmetry object for symmetrization.

    Returns:
        dict: Contains local resolution map, filtered map, and optionally resolution histogram.
    """
    fsc_unmasked = calculate_fsc(half_1, half_2)
    step_size = real_round(torch.tensor(locres_sampling / angpix))
    maskrad_pix = real_round(torch.tensor(locres_sampling / angpix / 2))
    edgewidth_pix = step_size

    if mean_map is None:
        mean_map = (half_1+half_2)/2
    f_sharp_map = torch.fft.rfftn(mean_map, norm="forward")
    apply_bfactor(f_sharp_map, b_factor, angpix)
    
    Ifil = Ilocres = Isumw = 0
    coords = generate_3d_coordinates(half_1.shape, step_size)
    if mask is not None:
        coords = filter_coordinates_by_mask(coords, mask > 0)

    for z, y, x in tqdm(coords, total=coords.shape[0]):
        # Create a raised cosine mask
        loc_mask = raised_cosine_mask(half_1_pr.shape, maskrad_pix, maskrad_pix + edgewidth_pix, x, y, z)
        if mask is not None:
            loc_mask *= mask

        # Apply the mask and compute FSC
        half_1_m = half_1 * loc_mask
        half_2_m = half_2 * loc_mask
        fsc_masked = calculate_fsc(half_1_m, half_2_m)
        
        half_1_pr_m = half_1_pr * loc_mask
        half_2_pr_m = half_2_pr * loc_mask
        fsc_random_masked = calculate_fsc(half_1_pr_m, half_2_pr_m)

        fsc_true = calculate_fsc_true(fsc_masked, fsc_random_masked, randomize_at)
        local_resol = calculate_resolution(fsc_true, angpix)

        f_corrected_map = apply_fsc_weighting(f_sharp_map, fsc_true)
        f_corrected_map = low_pass_filter_map(f_corrected_map, half_1.shape[0], local_resol, angpix, filter_edge_width)
        corrected_map = torch.fft.irfftn(f_corrected_map, norm="forward")


        Ifil += corrected_map * loc_mask
        Ilocres += loc_mask / local_resol
        Isumw += loc_mask

    if symmetry is not None:
        Isumw = symmetrise_results(Isumw, symmetry)
        Ilocres = symmetrise_results(Ilocres, symmetry)
        Ifil = symmetrise_results(Ifil, symmetry)

    locres_mask = torch.where((Isumw > 0), (1 / (Ilocres / Isumw)), 0)
    locres_filtered = torch.where((Isumw > 0), (Ifil / Isumw), 0)

    results = {"local_resolution": locres_mask, "filtered_map": locres_filtered}
    if mask is not None:
        results["resolution_histogram"] = torch.histogram(half_1.shape[0] * angpix / locres_mask[mask>0.5], bins=fsc_unmasked.shape[0], range=(0, fsc_unmasked.shape[0]))
    return results

def symmetrise_results(data: torch.Tensor, symmetry: Symmetry) -> torch.Tensor:
    D, H, W = data.shape
    z, y, x = torch.meshgrid(
        torch.linspace(-1, 1, D, dtype=data.dtype, device=data.device),
        torch.linspace(-1, 1, H, dtype=data.dtype, device=data.device),
        torch.linspace(-1, 1, W, dtype=data.dtype, device=data.device),
        indexing="ij"
    )
    """
    Apply symmetry operations to a 3D tensor by averaging over all symmetry-related positions.

    Args:
        data (torch.Tensor): 3D tensor to be symmetrized.
        symmetry (Symmetry): Symmetry object containing rotation matrices.

    Returns:
        torch.Tensor: Symmetrized tensor.
    """
    grid_flat = torch.stack([x, y, z], dim=-1).reshape(-1, 3)  # Shape: [D, H, W, 3]

    rotated_sum = data.clone()
    for r in symmetry.rr:
        rotation = r[:3, :3].to(device=data.device, dtype=data.dtype)  # Shape: [3, 3]
        rotated_grid_flat = torch.matmul(grid_flat, rotation.T)  # Shape: [D*H*W, 3]

        # Reshape back to grid shape
        rotated_grid = rotated_grid_flat.reshape(1, D, H, W, 3)  # Shape: [D, H, W, 3]

        # Normalize the grid to [-1, 1] for grid_sample
        # rotated_grid = rotated_grid / rotated_grid.abs().max()

        rotated_tensor = F.grid_sample(
            data.unsqueeze(0).unsqueeze(0),
            rotated_grid,  # Add batch dimension to grid
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        )
        rotated_sum += rotated_tensor.squeeze(0).squeeze(0)

    return rotated_sum / (len(symmetry.rr) + 1)

def postprocess(I1: torch.Tensor, 
                I2, angpix: torch.Tensor, 
                autob_lowres: float, 
                mask: torch.Tensor = None, 
                mean_map: torch.Tensor = None, 
                randomize_fsc_at: float = 0.8, 
                symmetry: Symmetry = None) -> dict[str, torch.Tensor]:
    """
    Perform postprocessing on two half-maps, including FSC calculation, sharpening, and local resolution estimation.

    Args:
        I1 (torch.Tensor): First half-map (3D tensor).
        I2 (torch.Tensor): Second half-map (3D tensor).
        angpix (float): Pixel size in Angstroms.
        autob_lowres (float): Low resolution cutoff for sharpening.
        mask (torch.Tensor, optional): Mask to apply (3D tensor).
        mean_map (torch.Tensor, optional): Mean map for sharpening.
        randomize_fsc_at (float, optional): FSC threshold for phase randomization.
        symmetry (Symmetry, optional): Symmetry object for symmetrization.

    Returns:
        dict: Dictionary containing sharpened map, FSC curves, local resolution map, and filtered map.
    """
    fsc_unmasked = calculate_fsc(I1, I2)
    fsc_dict = {"fsc unmasked": fsc_unmasked}

    fsc_true = fsc_unmasked
    I1_r = I1
    I2_r = I2
    randomize_at = 25
    if mask is not None:
        fsc_masked = calculate_fsc(I1 * mask, I2 * mask)
        fsc_dict["fsc masked"] = fsc_masked

        # Randomize phases
        randomize_at = determine_randomization_threshold(fsc_unmasked, randomize_fsc_at)
        I1_r = randomize_phases(I1, randomize_at)
        I2_r = randomize_phases(I2, randomize_at)
        fsc_random_masked = calculate_fsc(I1_r * mask, I2_r * mask)
        
        fsc_dict["fsc random masked"] = fsc_random_masked
        # Calculate FSC_true
        fsc_true = calculate_fsc_true(fsc_masked, fsc_random_masked, randomize_at)

    fsc_dict["fsc true"] = fsc_true
    
    if mean_map is None:
        mean_map = (I1 + I2) / 2
    results = sharpen_map(mean_map, fsc_true, angpix, autob_lowres)
    results["fsc_postprocess"] = fsc_dict
    results["sharp_resolution"] = calculate_resolution(fsc_true, angpix)
    results.update(locres(I1, I2, I1_r, I2_r, results["b_factor"], angpix, randomize_at, mask, mean_map, symmetry=symmetry))
    return results
