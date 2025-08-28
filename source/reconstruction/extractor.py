import torch

from source.reconstruction.ctf import CTF
from source.reconstruction.dose_weighter import DoseWeighter


class Extractor(torch.nn.Module):
    """
    Class for extraction of particles from micrographs and their fouriercropping.
    Hparams:
        ctf: CTF - Module responsible for contrast transfre function correction
        dose_weighter: DoseWeighter - Module responsible for dose weighting,
        new_size: int - new size of particle,
        bg_radius: int - radius of background,
        invert_contrast: bool - should contrast be inverted?,
        ctf_mode: str - mode of CTF correction, can be 'none', 'premultiply' or 'sign',
        dw_mode: bool - should dose weighting be applied post extraction?
    """
    def __init__(self, ctf: CTF, dose_weighter: DoseWeighter, new_size: int, bg_radius: int, invert_contrast: bool, ctf_mode: str, dw_mode: bool=False, *args, **kwargs):
        super(Extractor, self).__init__()
        self.invert_contrast = invert_contrast
        self.ctf_mode = ctf_mode
        self.dw_mode = dw_mode
        self.ctf = ctf
        self.dose_weighter = dose_weighter
        self.crop_size = new_size // 2 + 1
        self.new_size = new_size
        self.register_buffer("coords", 
                             create_centered_xy1_grid(new_size, new_size).permute((2, 0, 1)), 
                             persistent=False)
        r2 = (self.coords[:-1] ** 2).sum(dim=0)
        mask = r2 > bg_radius**2
        self.mask_f = mask.flatten()
        x_f = self.coords[0].flatten()[self.mask_f]
        y_f = self.coords[1].flatten()[self.mask_f]
        xy1 = torch.stack([x_f, y_f, torch.ones_like(x_f)], dim=0)
        self.register_buffer("mat", (torch.linalg.inv(xy1 @ xy1.T) @ xy1).permute((1,0)), persistent=False)

    def rescale_patch(self, patch: torch.Tensor, ctf_params: dict[str, object]) -> torch.Tensor:
        f_patch = torch.fft.rfft2(patch, norm='forward')
        f_patch_crop = torch.cat([f_patch[..., :self.crop_size, :self.crop_size], 
                                  f_patch[..., (2-self.crop_size):, :self.crop_size]], 
                                  dim=-2)
        c_patch = torch.fft.irfft2(f_patch_crop, norm='forward')
        
        n_patch = self.normalise_patch(c_patch)
        if self.invert_contrast:
            n_patch *= -1

        if self.ctf_mode != 'none' or self.dw_mode:
            n_patch = torch.fft.rfft2(n_patch, norm='forward')

            if self.ctf_mode != 'none':
                fftw_image = self.ctf.get_fftw_image(**ctf_params)
                if self.ctf_mode == 'premultiply':
                    n_patch *= fftw_image
                    n_patch = torch.fft.irfft2(n_patch, norm='forward')
                elif self.ctf_mode == 'sign':
                    n_patch *= fftw_image.sign()

            if self.dw_mode:
                n_patch *= self.dose_weighter.weight
                
            n_patch = torch.fft.irfft2(n_patch, norm='forward')

        return n_patch

    def normalise_patch(self, patch: torch.Tensor) -> torch.Tensor:
        v_f = patch.flatten(start_dim=-2, end_dim=-1)[..., self.mask_f].mean(dim=1, keepdim=True) ### check if mistake, shouldn't mean have dim=2
        plane_img = torch.einsum('bij,jm,mhw->bihw', v_f, self.mat, self.coords)
        v_f -= plane_img.flatten(start_dim=-2, end_dim=-1)[..., self.mask_f]
        std, mean = torch.std_mean(v_f, dim=-1, keepdim=True)
        n_patch = (patch - plane_img - mean[..., None]) / std[..., None]
        return n_patch

def create_centered_xy1_grid(dimy: int, dimx: int) -> torch.Tensor:
    x = torch.arange(-(dimx // 2), dimx // 2, dtype=torch.double)
    y = torch.arange(-(dimy // 2), dimy // 2, dtype=torch.double)
    y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
    xy1 = torch.stack((x_grid, y_grid, torch.ones_like(y_grid)), dim=-1)
    return xy1
