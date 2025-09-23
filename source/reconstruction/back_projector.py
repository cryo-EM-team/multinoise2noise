import itertools
import torch
import torch.nn.functional as F
import gc

from source.reconstruction.utils import real_round, create_centered_zyx_grid, create_centered_xyz_grid
from source.reconstruction.symmetry import Symmetry


class BackProjector(torch.nn.Module):
    """
    Module performing operations on 3D Fourier volume such as backprojection of 2D slices,
    projection of 3D volume to 2D slices, enforcing symmetries, and reconstruction of 3D volume.
    """
    def __init__(self,
                 ori_size: int,
                 symmetry: Symmetry,
                 interpolator: bool = True,
                 padding_factor_3d: float = 2.0,
                 r_min_nn: int = 10):
        """
        Args:
            ori_size (int): size of original volume (expressed as length of an edge of cube).
            symmetry (Symmetry): Module responsible for handling symmetries.
            interpolator (bool, optional): should values be interpolated during backprojection.
            padding_factor_3d (float, optional): padding factor.
            r_min_nn (int, optional): threshold of pixels above which interpolation is used. 
        """
        super(BackProjector, self).__init__()
        self.ori_size = int(ori_size)
        self.r_max = torch.Tensor([self.ori_size / 2])
        self.padding_factor = padding_factor_3d
        self.register_buffer("max_r2", (self.r_max * self.padding_factor) ** 2, persistent=False)
        self.r_min_nn = torch.Tensor([r_min_nn])
        self.register_buffer("min_r2_nn", (self.r_min_nn * self.padding_factor) ** 2, persistent=False)
        self.symmetry = symmetry
        self.symmetry.create_rotation_matrices()

        if padding_factor_3d < 1.0:
            raise ValueError("Padding factor cannot be less than 1.")

        self.interpolator = interpolator

        self.dimx = self.ori_size + int(self.padding_factor)
        self.dimy = self.dimz = self.dimx * 2 - 1
        self.data = torch.zeros((self.dimz, self.dimy, self.dimx), dtype=torch.complex128)
        self.weight = torch.zeros((self.dimz, self.dimy, self.dimx), dtype=torch.double)
        self.weight_precalculated = False
        self.register_buffer("init_coords", torch.tensor([1 - self.dimx, 1 - self.dimx, 0], dtype=torch.long), persistent=False)
        self.register_buffer("offset_combinations", torch.tensor(list(itertools.product([0, 1], repeat=3)), dtype=torch.long), persistent=False)

    def backproject2Dto3D(self, f2d: torch.Tensor, A: torch.Tensor, Mweight: torch.Tensor):
        """
        Backproject 2D Fourier slices into 3D Fourier volume.
        Args:
            f2d (torch.Tensor): 2D Fourier slices to be backprojected. Shape: (N, H, W), where N is number of slices, H is height, W is width.
            A (torch.Tensor): Affine matrices defining orientations of the slices. Shape: (N, 3, 3).
            Mweight (torch.Tensor): Weights associated with each pixel in the slices. Shape: (N, H, W).
        """
        _, _, H, W = f2d.shape
        m = torch.eye(2, dtype=torch.double, device=A.device)

        Ainv = A.mT * self.padding_factor
        Am = Ainv[..., :2] @ m
        AtA = torch.bmm(Am.mT, Am)
        AtA_xx, AtA_xy, AtA_yy = AtA[:, 0, 0].unsqueeze(1), AtA[:, 0, 1].unsqueeze(1), AtA[:, 1, 1].unsqueeze(1)
        AtA_xy2 = AtA_xy ** 2

        y = torch.cat((torch.arange(W, device=f2d.device, dtype=torch.double),
                       torch.arange(start=W - H, end=0, device=f2d.device, dtype=torch.double)), 0)
        y2 = y ** 2
        discr = AtA_xy2 * y2 - AtA_xx * (AtA_yy * y2 - self.max_r2)
        q0 = torch.sqrt(discr) / AtA_xx
        q1 = -AtA_xy * y / AtA_xx

        first_x = torch.ceil(q1 - q0).unsqueeze(-1).clamp_min(0.)
        first_x[:, W:] = first_x[:, W:].clamp_min(1.)
        last_x = torch.floor(q1 + q0).unsqueeze(-1).clamp_max(W - 1)

        y_grid, x_grid = torch.meshgrid(y, torch.arange(W, device=f2d.device, dtype=torch.double), indexing="ij")
        yx = torch.stack((y_grid, x_grid), dim=-1) @ m
        p = torch.einsum('nij,abj->nabi', Ainv[..., :2].flip((-2, -1)), yx)
        r2_3D = (p ** 2).sum(dim=-1)
        conj = torch.conj(f2d).resolve_conj()
        x_grid = x_grid.unsqueeze(0)
        mask = (x_grid >= first_x) & (x_grid <= last_x) & (Mweight > 0.).prod(dim=1).bool() & (r2_3D <= self.max_r2) & (
                    discr.unsqueeze(-1) >= 0.)
        mask = mask.reshape(-1)

        if self.interpolator or r2_3D.lt(self.min_r2_nn).all():
            neg_x = (p[..., 2] < 0)
            p *= 1 - 2 * neg_x.unsqueeze(-1)
            my_val = torch.where(neg_x.unsqueeze(1), conj, f2d)

            init_coords = self.init_coords.clone()
            init_coords[..., 2] = 0

            p0 = torch.floor(p).long()
            f = p - p0
            f = torch.stack([1. - f, f], dim=-1)
            dd = torch.einsum('...i,...j,...k->...ijk', *(f.unbind(-2)))
            p0, dd, my_val = [t[mask] for t in (
            p0.reshape(-1, 3), dd.reshape(-1, 2, 2, 2), my_val.reshape(-1))]
            if not self.weight_precalculated:
                my_weight = Mweight.reshape(-1)[mask]
            p0 -= init_coords

            in_bounds = (p0 >= 0).prod(dim=-1).bool() & \
                        (p0[..., 0] < self.dimz) & (p0[..., 1] < self.dimy) & (p0[..., 2] < self.dimx)

            p0, dd, my_val = [t[in_bounds] for t in (p0, dd, my_val)]
            if not self.weight_precalculated:
                my_weight = my_weight[in_bounds]
            indices = p0[..., 0] * (self.dimy * self.dimx) + p0[..., 1] * self.dimx + p0[..., 2]
            for offset in self.offset_combinations:
                flat_indices = (indices + (offset[0] * (self.dimy * self.dimx) + offset[1] * self.dimx + offset[2])).cpu()

                o_z, o_y, o_x = offset.unbind(-1)

                self.data.view(-1).index_add_(0, flat_indices, (dd[..., o_z, o_y, o_x] * my_val).cpu())
                if not self.weight_precalculated:
                    self.weight.view(-1).index_add_(0, flat_indices, (dd[..., o_z, o_y, o_x] * my_weight).cpu())

        elif not self.interpolator:
            p0 = real_round(p).long()
            neg_x = (p0[..., 2] < 0)
            p0 *= 1 - 2 * neg_x.unsqueeze(-1)
            my_val = torch.where(neg_x.unsqueeze(1), conj, f2d)

            p0, my_val = [t[mask] for t in (p0.view(-1, 3), my_val.view(-1))]
            if not self.weight_precalculated:
                my_weight = Mweight.reshape(-1)[mask]
            p0 -= self.init_coords
            in_bounds = (p0 >= 0).prod(dim=-1).bool() & \
                        (p0[..., 0] < self.data.shape[-3]) & (p0[..., 1] < self.data.shape[-2]) & (
                                    p0[..., 2] < self.data.shape[-1])

            p0, my_val = [t[in_bounds] for t in (p0, my_val)]
            if not self.weight_precalculated:
                my_weight = my_weight[in_bounds]

            flat_indices = (p0[..., 0] * (self.dimy * self.dimx) + p0[..., 1] * self.dimx + p0[..., 2]).cpu()
            self.data.view(-1).index_add_(0, flat_indices, my_val.cpu())
            if not self.weight_precalculated:
                self.weight.view(-1).index_add_(0, flat_indices, my_weight.cpu())

        else:
            raise ValueError("FourierInterpolator::backproject ERROR: unrecognized interpolator")

    def project3Dto2D(self, A: torch.Tensor) -> torch.Tensor:
        """
        Project 3D Fourier volume into 2D Fourier slices.
        Args:
            A (torch.Tensor): Affine matrices defining orientations of the slices. Shape: (N, 3, 3).
        Returns:
            torch.Tensor: 2D Fourier slices. Shape: (N, H, W), where N is number of slices, H is height, W is width.
        """
        W = self.data.shape[-1] // 2
        H = W * 2 - 2
        y = torch.cat((torch.arange(W, device=A.device, dtype=torch.double),
                    torch.arange(start=W - H, end=0, device=A.device, dtype=torch.double)), 0)
        y_grid, x_grid = torch.meshgrid(y, torch.arange(W, device=A.device, dtype=torch.double), indexing="ij")
        yx = torch.stack((y_grid, x_grid), dim=-1) @ torch.eye(2, dtype=torch.double, device=A.device)
    
        Ainv = A.mT * self.padding_factor
        p = torch.einsum('nij,abj->nabi', Ainv[..., :2].flip((-2, -1)), yx)
    
        neg_x = (p[..., 2] < 0)
        p *= 1 - 2 * neg_x.unsqueeze(-1)
        p -= self.init_coords
        p /= torch.tensor([(self.dimz - 1), (self.dimy - 1), (self.dimx - 1)], device=p.device)/ 2
        p = (p.flip(-1)[:, None, ...] - 1).cpu()
    
        real = F.grid_sample(self.data.real[None, None, ...].expand(A.shape[0], -1, -1, -1, -1), 
                             p, align_corners=True, padding_mode='zeros')[:,:,0]
        imag = F.grid_sample(self.data.imag[None, None, ...].expand(A.shape[0], -1, -1, -1, -1), 
                             p, align_corners=True, padding_mode='zeros')[:,:,0]
        rotated_data = torch.complex(real, imag).to(device=A.device)
        rotated_data = torch.nan_to_num(torch.where(neg_x.unsqueeze(1), torch.conj(rotated_data), rotated_data), nan=0)
        return rotated_data
    
    def symmetrise(self):
        """
        Enforce Hermitian and point group symmetries on the 3D Fourier volume.
        """
        self.enforce_hermitian_symmetry()
        self.apply_point_group_symmetry()

    def enforce_hermitian_symmetry(self):
        """Enforce Hermitian symmetry on the 3D Fourier volume."""
        d = self.data[:, :, self.init_coords[..., 2].item()]
        self.data[:, :, self.init_coords[..., 2].item()] += d.conj().resolve_conj().flip(dims=[0, 1])
        if not self.weight_precalculated:
            w = self.weight[:, :, self.init_coords[..., 2].item()]
            self.weight[:, :, self.init_coords[..., 2].item()] += w.flip(dims=[0, 1])

    def apply_point_group_symmetry(self):
        """Enforce point group symmetry on the 3D Fourier volume."""
        xyz = create_centered_xyz_grid(self.dimz, self.dimy, self.dimx).to(device=self.data.device)[None, ...]
        tmp_data = self.data.clone()[None, None, ...]
        if not self.weight_precalculated:
            tmp_weight = self.weight.clone()[None, None, ...]

        for r in self.symmetry.rr:
            r_ = r[:3, :3].clone().to(device=self.data.device) / (self.dimx - 1)
            r_[0] *= 2
            grid = torch.einsum('nabcd,ed->nabce', xyz, r_)
            neg_x = (1 - 2 * (grid[..., 0] < 0)).double()
            grid *= neg_x.unsqueeze(-1)
            grid[..., 0] -= 1

            self.data += torch.complex(
                F.grid_sample(tmp_data.real, grid, align_corners=True, padding_mode='zeros')[0],
                F.grid_sample(tmp_data.imag, grid, align_corners=True, padding_mode='zeros')[0] * neg_x)[0]
            if not self.weight_precalculated:
                self.weight += F.grid_sample(tmp_weight, grid, align_corners=True, padding_mode='zeros')[0, 0]
    
    def mask_background(self, input: torch.Tensor, cosine_width: int = 3) -> None:
        """Apply a spherical mask to remove background of the reconstructed volume.
        Args:
            input (torch.Tensor): The input 3D volume to be masked.
            cosine_width (int, optional): Width of the cosine edge for the mask.
        """
        r = (torch.stack(
            torch.meshgrid(
                torch.arange(-self.r_max.item(), self.r_max.item(), device=self.data.device, dtype=torch.double), 
                torch.arange(-self.r_max.item(), self.r_max.item(), device=self.data.device, dtype=torch.double), 
                torch.arange(-self.r_max.item(), self.r_max.item(), device=self.data.device, dtype=torch.double), 
                indexing="ij"), 
            dim=-1) ** 2).sum(-1).sqrt()

        rval = torch.pi * r / (self.ori_size * self.padding_factor)
        rval = (torch.sin(rval) / (rval)) ** 2
        rval[r == 0] = 1

        radius_p = self.r_max + cosine_width
        background = 0.5 + 0.5 * torch.cos((radius_p - r) * torch.pi / cosine_width)
        background *= ((r > self.r_max) & (r <= radius_p)).double()
        background += (r > radius_p).double()
        del r
        avg_bg = (input * background).sum() / background.sum()

        input *= (1 - background) 
        input += background * avg_bg
        
        input /= rval

    def prepare_reconstruction_weights(self, weight_modifiers: torch.Tensor | None = None) -> None:
        """
        Prepare weights for the reconstruction process by performing radial averaging
        and applying any provided weight modifiers.
        Args:
            weight_modifiers (torch.Tensor | None, optional): Additional weight modifiers to be applied.
        """
        ires = (create_centered_zyx_grid(self.dimz, self.dimy, self.dimx) ** 2).sum(-1).view(-1)
        r2_mask = (ires < (real_round(self.r_max * self.padding_factor) ** 2))
        ires = torch.floor(ires[r2_mask].sqrt() / self.padding_factor).long()
        f_val = self.weight.view(-1)[r2_mask]

        radavg_weight = torch.zeros([self.r_max.long().item()], dtype=torch.double)
        radavg_weight.index_add_(0, ires, f_val)
        radavg_weight[radavg_weight <= 0] = 1
        counter = torch.zeros_like(radavg_weight)
        counter.index_add_(0, ires, torch.ones_like(ires, dtype=torch.double))
        counter[counter <= 0] = 1
        radavg_weight /= counter
        del counter
        radavg_weight = radavg_weight.index_select(0, ires)
        if weight_modifiers is not None:
            radavg_weight *= weight_modifiers.view(-1)[r2_mask]
        radavg_weight /= 1000
        self.weight.view(-1)[r2_mask] = torch.stack((radavg_weight, f_val), dim=-1).max(dim=-1)[0]
        self.weight.view(-1)[~r2_mask] = float('inf')

    def reconstruct(self, weight_modifiers: torch.Tensor | None = None) -> torch.Tensor:
        """
        Reconstruct the 3D volume from the accumulated Fourier data.
        Args:
            weight_modifiers (torch.Tensor | None, optional): Additional weight modifiers to be applied during weight preparation.
        Returns:
            torch.Tensor: The reconstructed 3D volume.
        """
        o1 = -1
        o2 = 2
        if not self.weight_precalculated:
            self.prepare_reconstruction_weights(weight_modifiers)
            del weight_modifiers
            self.weight_precalculated = True

        self.data /= self.weight
        window_data = self.data[o2:o1, o2:o1, :-1].roll(shifts=self.dimx + o1, dims=1).roll(shifts=self.dimx + o1, dims=0)
        window_data[1::2] *= -1
        if window_data.shape[-2] % 2 == 0:
            window_data[:, 1::2] *= -1
        if window_data.shape[-1] % 2 == 1:
            window_data[:, :, 1::2] *= -1
        gc.collect()
        # THIS OPERATION IS VERY MEMORY INTENSIVE (SERIOUSLY)
        window_data = torch.fft.irfftn(window_data, norm="forward") / (self.padding_factor ** 3 * self.ori_size)
        gc.collect()
        window_data = window_data[(window_data.shape[0] // 4):(window_data.shape[0] // 4) * 3,
                                  (window_data.shape[1] // 4):(window_data.shape[1] // 4) * 3,
                                  (window_data.shape[2] // 4):(window_data.shape[2] // 4) * 3]
        self.mask_background(window_data)
        return window_data
