import torch
import torch.nn.functional as F
import lightning as pl
import os
import mrcfile
import hydra
import tifffile as tiff
import numpy as np


class MotionCorrector(pl.LightningModule):
    """
    Class for motion correction of micrograph movies.
    Hparams:
        max_iter: int - maximal amount of iterations for local alignment
        b_factor: int - measure of (local) mobility (usually 150)
        prescaling: int - prescaling factor (usually 1)
        patch_x: int - On how many patches should micrograph frame be split along X axis
        patch_y: ing - On how many patches should micrograph frame be split along Y axis
        center_by_interpolation: bool - should data be centered by interpolation (in relion version True)
        apply_dose_weighting: bool - should dose weighting be applied (usually True)
        dose_weighter: dict - hyperparameters used to instantiate DoseWeighter class
    """
    def __init__(self, 
                 *args, 
                 **kwargs):
        super(MotionCorrector, self).__init__()
        self.save_hyperparameters(logger=False)
        self.dose_weighter = hydra.utils.instantiate(self.hparams.dose_weighter)
        self.register_buffer("scaled_b", torch.tensor([self.hparams.b_factor], dtype=torch.float64) / torch.tensor([self.hparams.prescaling], dtype=torch.float64).square(), persistent=False)
        self.register_buffer("ccf_requested_scale", torch.sqrt(-torch.log(torch.tensor(1E-8)) / (2 * self.scaled_b)) if self.scaled_b > 0 else torch.ones(1, dtype=torch.float64), persistent=False)
        
        # Hot pixel elimination constants
        self.hotpixel_sigma = 6
        self.D_MAX = 2
        self.num_min_ok = 6

        # Global and local motion correction constants
        self.search_range = torch.tensor([50], dtype=torch.float64)
        self.register_buffer("mult_vec", torch.tensor([1, -2, 1], dtype=torch.float64))
        self.register_buffer("channel_indices", torch.arange(self.hparams.n_frames, dtype=torch.int))
        self.tolerance = 0.5
        self.patch_EPS = 1e-15
        self.local_EPS = 1e-10
        
        # Helpful indices
        z_center = torch.arange(self.hparams.n_frames, dtype=torch.float64)
        self.register_buffer("z3", torch.stack([z_center, z_center ** 2, z_center ** 3], dim=-1), persistent=False)
        z_center += 0.5
        self.register_buffer("z3_5", torch.stack([z_center, z_center ** 2, z_center ** 3], dim=-1).unsqueeze(-2), persistent=False)

        self.register_buffer("bad_pixels_coords", torch.tensor([0]), persistent=False)
        #self.bad_pixels_coords = None

    def set_bad_pixels_coords(self, bad_pixels_coords: torch.Tensor):
        self.bad_pixels_coords = bad_pixels_coords
    
    def motion_correction(self, input_data: torch.Tensor):
        ### Hot pixels
        input_data = self.eliminate_hot_pixels(input_data[0])
        ### FFT
        f_data = torch.fft.rfft2(input_data, norm="forward")
        ### Global alignment
        self.align_patch(f_data, True)
        print("aligned")
        ### Inverse Fourier Transform
        input_data = torch.fft.irfft2(f_data, norm="forward")
        print("fft")
        ### Local alignment
        coeff = self.local_alignment(input_data)
        print("local")
        ### Dose weighting
        if self.hparams.apply_dose_weighting:
            f_data = self.dose_weighter.weight * f_data
        print("weighted")
        ### Inverse weighted transform
        input_data = torch.fft.irfft2(f_data, norm="forward")
        print("fft 2")
        del f_data
        ### Interpolation based on local shifts
        if coeff is not None:
            input_data = self.real_space_interpolation_third_order_polynomial_without_sum(input_data, coeff)
        print("coeff")
        return input_data[None, :, :, :]
    
    def local_alignment(self, input_data: torch.Tensor) -> torch.Tensor:
        x_size = input_data.shape[-1] // self.hparams.patch_x
        y_size = input_data.shape[-2] // self.hparams.patch_y
        
        patches = input_data.unfold(1, y_size, y_size).unfold(2, x_size, x_size)
        
        if x_size % 2 == 1:
            patches = patches[..., :-1]
        if y_size % 2 == 1:
            patches = patches[..., :-1, :]

        f_patch = torch.fft.rfft2(patches, norm="forward")
        y_centers = ((torch.arange(self.hparams.patch_y, dtype=torch.float64, device=input_data.device) + 0.5) * y_size - 1).floor()
        x_centers = ((torch.arange(self.hparams.patch_x, dtype=torch.float64, device=input_data.device) + 0.5) * x_size - 1).floor()
        print("centers", y_centers, x_centers)
        y_norm = y_centers / (input_data.shape[-2]-1) - 0.5
        x_norm = x_centers / (input_data.shape[-1]-1) - 0.5

        patch_matA, patch_shifts = [], []

        if torch.cuda.is_available():
            self.z3_5 = self.z3_5.to("cuda")
        for y, y_n in zip(range(self.hparams.patch_y), y_norm):
            for x, x_n in zip(range(self.hparams.patch_x), x_norm):
                converged, shifts = self.align_patch(f_patch[:, y, x, ...].clone())
                if converged:
                    if self.hparams.center_by_interpolation:
                        interpolated_shifts = self.interpolate_shifts(shifts)
                        shifts -= interpolated_shifts[:, 0:1]
                    matA = torch.tensor([[[1], [x_n], [x_n ** 2], [y_n], [y_n ** 2], [x_n * y_n]]], dtype=torch.float64, device=self.mult_vec.device)
                    matA = (matA * self.z3_5).reshape(self.z3_5.shape[0], 18)
                    patch_matA.append(matA)
                    patch_shifts.append(shifts)
        coeff = None
        if len(patch_matA) > 0:
            matA = torch.cat(patch_matA, dim=0)
            shifts = torch.cat(patch_shifts, dim=1).permute(1,0)
            U, S, Vt = torch.linalg.svd(matA, full_matrices=False)
            S_inv = torch.diag(torch.where(S > self.local_EPS, 1.0 / S, torch.zeros_like(S)))
            
            base_solution = Vt.T @ S_inv @ U.T
            coeff = base_solution @ shifts
        
        return coeff
    
    def real_space_interpolation_third_order_polynomial_without_sum(self, input_data: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
        H, W = input_data.shape[-2:]
        grid = torch.stack(
            torch.meshgrid([
                torch.linspace(-0.5, 0.5, H, device=input_data.device, dtype=input_data.dtype),
                torch.linspace(-0.5, 0.5, W, device=input_data.device, dtype=input_data.dtype)], 
                indexing="ij"), dim=-1)

        basis = torch.stack([
            torch.ones_like(grid[..., 0]),
            grid[..., 1],
            grid[..., 1].square(),
            grid[..., 0],
            grid[..., 0].square(),
            grid[..., 0] * grid[..., 1]
        ], dim=0)  # shape: [6, H, W]

        div = torch.tensor([H-1, W-1], dtype=coeff.dtype, device=coeff.device)
        coeff_ = torch.stack([coeff[::3], coeff[1::3], coeff[2::3]]) / div
        if torch.cuda.is_available():
            self.z3 = self.z3.to("cuda")
        coeff_ = torch.einsum('ab, bcd -> acd', self.z3, coeff_).cpu()
        shift_field = torch.einsum('abc, bde -> adec', coeff_, basis)
        grid = (grid.unsqueeze(0) - shift_field).flip(dims=[-1]) * 2
        corrected = F.grid_sample(input_data.unsqueeze(1), grid, mode='bilinear', align_corners=True, padding_mode='border')
        return corrected.squeeze(1)
    
    def eliminate_hot_pixels(self, new_data: torch.Tensor) -> torch.Tensor:
        sum_input = new_data.sum(dim=0)
        std, mean = torch.std_mean(sum_input)
        threshold = mean + self.hotpixel_sigma * std
        bad_pixels = sum_input.gt(threshold)
        
        frame_mean = mean / new_data.shape[0]
        frame_std = std / new_data.shape[0]

        bad_pixels[self.bad_pixels_coords[:, 0], self.bad_pixels_coords[:, 1]] = True
        coordinates = torch.nonzero(bad_pixels, as_tuple=False)

        if coordinates.shape[0] == 0:
            return new_data
        
        padded_image = new_data
        padding = (torch.logical_or(
                   torch.logical_or(coordinates[:, 0].ge(new_data.shape[1] - self.D_MAX), coordinates[:, 1].ge(new_data.shape[2] - self.D_MAX)), 
                   torch.logical_or(coordinates[:, 0].le(self.D_MAX), coordinates[:, 1].le(self.D_MAX)))).any()
        if padding:
            padded_image = F.pad(padded_image, (self.D_MAX, self.D_MAX, self.D_MAX, self.D_MAX), mode='reflect')
            bad_pixels = F.pad(bad_pixels[None, ...].int(), (self.D_MAX, self.D_MAX, self.D_MAX, self.D_MAX), mode='reflect')[0].bool()
            coordinates += self.D_MAX

        patches_mask = ~torch.stack([bad_pixels[(coord[0] - self.D_MAX):(coord[0] + self.D_MAX + 1), 
                                                (coord[1] - self.D_MAX):(coord[1] + self.D_MAX + 1)] for coord in coordinates])
        
        frame_indexes = torch.arange(new_data.size(0))
        
        patches = [
            padded_image[:, 
                         coord[0] - self.D_MAX: coord[0] + self.D_MAX + 1, 
                         coord[1] - self.D_MAX: coord[1] + self.D_MAX + 1
            ][:, mask.view(mask.shape[0], mask.shape[1]).repeat(1, 1)] 
            for coord, mask in zip(coordinates, patches_mask)]
        
        substitute_values = torch.stack([patch[frame_indexes, torch.randint(0, patch.size(1), (patch.size(0),))] 
                                         if patch.size(1) > self.num_min_ok else 
                                         torch.normal(mean=frame_mean.repeat(patch.size(0)), std=frame_std.repeat(patch.size(0))) 
                                         for patch in patches])
        if padding:
            coordinates -= self.D_MAX
        new_data[frame_indexes.repeat(coordinates.size(0)), 
                 coordinates[:, 0].repeat_interleave(new_data.size(0)), 
                 coordinates[:, 1].repeat_interleave(new_data.size(0))] = substitute_values.flatten()
        
        return new_data
    
    def align_patch(self, Fframes: torch.Tensor, shift_input: bool = False) -> tuple[bool, torch.Tensor]:
        pn = torch.tensor([Fframes.shape[-2], Fframes.shape[-1] * 2 - 2], dtype=torch.float64)

        if pn[0] % 2 == 1 or pn[1] % 2 == 1:
            raise ValueError("Patch size must be even")

        pn_scaled = (pn * self.ccf_requested_scale).int()
        ccf_ny = self.find_good_size(pn_scaled[0])
        ccf_nx = self.find_good_size(pn_scaled[1]) #// 2 + 1
        ccf = torch.stack([ccf_ny, ccf_nx], dim=0)
        ccf = torch.minimum(ccf, pn)

        ccf_scale = pn / ccf
        search_range = torch.minimum(self.search_range // ccf_scale.amax(dim=0), (ccf // 2 - 1).amin(dim=0)).int().item()

        ccf = ccf.int()
        ccf_ny = ccf[0].item()
        ccf_nx = ccf[1].item()

        y = (torch.fft.fftfreq(ccf_ny, d=1 / ccf_ny, device=Fframes.device, dtype=torch.float64) / Fframes.shape[-2]).square()
        x = (torch.fft.rfftfreq(ccf_nx, d=1 / ccf_nx, device=Fframes.device, dtype=torch.float64) / Fframes.shape[-1]).square()
        y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
        if torch.cuda.is_available():
            self.mult_vec = self.mult_vec.to("cuda")
        weight = torch.exp(-2 * (y_grid + x_grid) * self.scaled_b).to(device=self.mult_vec.device)
        del y_grid, x_grid, y, x

        Fref = torch.cat([Fframes[:, :ccf_ny // 2 + 1, :ccf_nx // 2 + 1], 
                          Fframes[:, -ccf_ny // 2 + 1:, :ccf_nx // 2 + 1]], dim=1).to(device=self.mult_vec.device)
        shifts = torch.zeros((2, self.hparams.n_frames), dtype=torch.float64, device=self.mult_vec.device)
        converged = False
        for iter_num in range(self.hparams.max_iter):
            Iccs = torch.fft.irfft2((Fref.sum(dim=0, keepdim=True) - Fref) * weight * Fref.conj(), norm="forward")
            Iccs_cut = torch.cat([
                torch.cat([
                    Iccs[:, ccf_ny-search_range:ccf_ny, ccf_nx-search_range:ccf_nx],
                    Iccs[:, 0:search_range+1, ccf_nx-search_range:ccf_nx]
                ], 1), 
                torch.cat([
                    Iccs[:, ccf_ny-search_range:ccf_ny, 0:search_range+1],
                    Iccs[:, 0:search_range+1, 0:search_range+1]
                ], 1)
            ], 2)
            
            max_x, posx = Iccs_cut.max(dim=-1)
            _, posy = max_x.max(dim=-1)
            posx, posy = posx.to(torch.int64), posy.to(torch.int64)
            posx = posx.gather(1, posy[..., None])[..., 0]
            pos = torch.stack([posy, posx], dim=0) - search_range

            vals = torch.stack([
                    torch.stack([Iccs[self.channel_indices, pos[0] - 1, pos[1]], Iccs[self.channel_indices, pos[0], pos[1]], Iccs[self.channel_indices, pos[0] + 1, pos[1]]], dim=-1),
                    torch.stack([Iccs[self.channel_indices, pos[0], pos[1] - 1], Iccs[self.channel_indices, pos[0], pos[1]], Iccs[self.channel_indices, pos[0], pos[1] + 1]], dim=-1)
                ], dim=0)
            del Iccs, Iccs_cut

            vals_sum = vals @ self.mult_vec
            cur_shifts = pos.double() - torch.where(vals_sum.abs().gt(self.patch_EPS), (vals[..., 2] - vals[..., 0]) / vals_sum / 2, 0.)
            cur_shifts *= ccf_scale[:, None].to(device=self.mult_vec.device)
            
            cur_shifts -= cur_shifts[:, 0:1].clone()
            rmsd = cur_shifts.square().sum(dim=0).mean().sqrt()
            #cur_shifts[:, 0] = 0
            shifts += cur_shifts

            Fref[1:] = self.shift_non_square_image_in_fourier_transform(
                Fref[1:], -cur_shifts[1, 1:], -cur_shifts[0, 1:], x_size=pn[1], y_size=pn[0]
            )

            print("rmsd", rmsd)
            if rmsd < self.tolerance:
                converged = True
                break
        if shift_input:
            Fframes[1:] = self.shift_non_square_image_in_fourier_transform(
                Fframes[1:], -shifts[1, 1:].cpu(), -shifts[0, 1:].cpu(), x_size=pn[1], y_size=pn[0]
            )
        return converged, shifts

    @staticmethod
    def shift_non_square_image_in_fourier_transform(Fframes: torch.Tensor, xshift: torch.Tensor, yshift: torch.Tensor, x_size: int, y_size: int) -> torch.Tensor:
        """
        Shift a batch of Fourier-transformed images by x and y shifts.

        Args:
            Fframes (torch.Tensor): Batch of Fourier-transformed images of shape (B, H, W//2+1).
            xshift (torch.Tensor): Batch of x-shifts of shape (B,).
            yshift (torch.Tensor): Batch of y-shifts of shape (B,).

        Returns:
            torch.Tensor: Shifted Fourier-transformed images, same shape as Fframes.
        """
        _, H, W_half = Fframes.shape
        W = 2 * (W_half - 1) if W_half > 1 else 1
        #x_size_full = 2 * (x_size - 1) if x_size > 1 else 1

        freq_y = torch.fft.fftfreq(H, d=y_size/H, device=Fframes.device, dtype=torch.double).view(H, 1)  # Frequencies along height
        freq_y[H//2] *= -1
        freq_x = torch.fft.rfftfreq(W, d=x_size/W, device=Fframes.device, dtype=torch.double).view(1, W_half)  # Frequencies along width

        phase = 2j * torch.pi * (yshift.view(-1, 1, 1) * freq_y + xshift.view(-1, 1, 1) * freq_x)
        phase_shift = torch.exp(phase)

        shifted_Fframes = Fframes * phase_shift

        return shifted_Fframes

    @staticmethod
    def interpolate_shifts(shifts: torch.Tensor) -> torch.Tensor:
        interpolated_shifts = (shifts[:, 1:] + shifts[:, :-1]) / 2
        interpolated_shifts = torch.cat([-interpolated_shifts[:, 0:1], interpolated_shifts], dim=1)
        return interpolated_shifts

    @staticmethod
    def find_good_size(request: int) -> torch.Tensor:
        # Numbers that do not contain large prime numbers
        good_numbers = torch.tensor([
            192, 216, 256, 288, 324,
            384, 432, 486, 512, 576, 648,
            768, 800, 864, 972, 1024,
            1296, 1536, 1728, 1944,
            2048, 2304, 2592, 3072, 3200,
            3456, 3888, 4096, 4608, 5000, 5184,
            6144, 6250, 6400, 6912, 7776, 8192,
            9216, 10240, 12288, 12500, request + request % 2
        ])
        valid_numbers = good_numbers[good_numbers >= request]
        return valid_numbers[0]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.motion_correction(x)

    def test_step(
        self, batch: dict[str, torch.Tensor], *args, **kwargs
    ):
        self.predict_step(batch=batch, args=args, kwargs=kwargs)

    def predict_step(
        self, batch: dict[str, torch.Tensor], *args, **kwargs
    ):
        data, path = batch['images'], batch['out_paths']
        extracted = self.forward(data)
        self.save(extracted, path)

    def save(self, images: torch.Tensor, out_paths: list[str]):
        """
            Save reconstructed data
        Args:
            images: Batch of images
            out_paths: paths under which results will be saved

        Returns:
            Computed predictions
        """
        for image, out_path in zip(images, out_paths):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            if '.mrc' in out_path:
                with mrcfile.new(out_path, overwrite=True) as mrc:
                    mrc.set_data(image.cpu().float().numpy())
                    mrc.voxel_size = self.hparams.pixel_size
                    mrc.header.mx = image.size(-1)
                    mrc.header.my = image.size(-2)
                    mrc.header.mz = image.size(-3)
                    mrc.update_header_stats()
            elif '.tif' in out_path:
                tiff.imwrite(out_path, image.cpu().float().numpy().astype(np.float32))
