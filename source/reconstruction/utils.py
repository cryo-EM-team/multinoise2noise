import torch


def real_round(input: torch.Tensor) -> torch.Tensor:
    return torch.floor(input + 0.5)

def create_centered_zyx_grid(dimz: int, dimy: int, dimx: int) -> torch.Tensor:
    x = torch.arange(0, dimx, dtype=torch.double)
    y = torch.arange(-(dimy // 2), dimx, dtype=torch.double)
    z = torch.arange(-(dimz // 2), dimx, dtype=torch.double)
    z_grid, y_grid, x_grid = torch.meshgrid(z, y, x, indexing="ij")
    zyx = torch.stack((z_grid, y_grid, x_grid), dim=-1)
    return zyx
    
def calculate_fsc(half_1: torch.Tensor, half_2: torch.Tensor) -> torch.Tensor:
    f_half_1 = half_1 if half_1.is_complex() else torch.fft.rfftn(half_1, norm="forward")
    f_half_2 = half_2 if half_2.is_complex() else torch.fft.rfftn(half_2, norm="forward")

    r = create_fourier_distance_map(f_half_1.shape).flatten()
    r = real_round(r).int()
    mask = r < f_half_1.shape[2]
    r = r[mask]
    h_1 = f_half_1.flatten()[mask]
    h_2 = f_half_2.flatten()[mask]
    norm_1 = h_1.real ** 2 + h_1.imag ** 2
    norm_2 = h_2.real ** 2 + h_2.imag ** 2
    norm_both = h_1.real * h_2.real + h_1.imag * h_2.imag
    num = torch.zeros([f_half_1.shape[2]], dtype=half_1.real.dtype)
    den1, den2 = num.clone(), num.clone()
    num.index_add_(0, r, norm_both)
    den1.index_add_(0, r, norm_1)
    den2.index_add_(0, r, norm_2)

    fsc = torch.nan_to_num(num / torch.sqrt(den1 * den2), 0)
    fsc[0] = 1.
    return fsc

def create_fourier_distance_map(shape: torch.Tensor) -> torch.Tensor:
    coords = torch.stack(torch.meshgrid(
        torch.fft.fftfreq(shape[0], 1/shape[0], dtype=torch.double),
        torch.fft.fftfreq(shape[1], 1/shape[1], dtype=torch.double),
        torch.arange(shape[2], dtype=torch.double),
        indexing='ij'
    ), dim=-1)
    return coords.square().sum(dim=-1).sqrt()

def low_resolution_join_halves(fsc: torch.Tensor, angpix: float, threshold: float) -> torch.Tensor:
    return torch.where(((fsc.shape[0] - 1) * 2 * angpix / torch.arange(fsc.shape[0])) < threshold, fsc, 1)

def calculate_resolution(fsc: torch.Tensor, angpix: float) -> float:
    vec = torch.nonzero(fsc < 0.143, as_tuple=False)
    if len(vec) > 0:
        i = vec[0].item() - 1
    else:
        i = len(fsc) - 1
    if i > 0:
        return (fsc.shape[0] - 1) * 2 * angpix / i
    return 50.

def get_downsampled_average(data: torch.Tensor, weight: torch.Tensor, divide: bool=True) -> torch.Tensor:
    downsampled_data = downsample(data)
    downsampled_weight = downsample(weight) if divide else 8
    downsampled_data /= downsampled_weight
    return downsampled_data

def downsample(data: torch.Tensor) -> torch.Tensor:
    data_patches = torch.cat((data[:, :, 0:1], data, data[:, :, -1:]), dim=2)
    data_patches = torch.cat((data_patches[:, :1, :],
                              data_patches[:, :(data_patches.shape[1] // 2 + 1), :], 
                              data_patches[:, (data_patches.shape[1] // 2):, :],
                              data_patches[:, -1:, :]), dim=1)
    data_patches = torch.cat((data_patches[:1, :, :],
                              data_patches[:(data_patches.shape[0] // 2 + 1), :, :], 
                              data_patches[(data_patches.shape[0] // 2):, :, :],
                              data_patches[-1:, :, :]), dim=0)
    data_patches = data_patches.unfold(0, 2, 2) \
                               .unfold(1, 2, 2) \
                               .unfold(2, 2, 2)  # Shape: (out_D, out_H, out_W, kernel_size, kernel_size, kernel_size)
    return data_patches.sum(dim=(-1, -2, -3))
