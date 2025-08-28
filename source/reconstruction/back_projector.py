import itertools
import torch
import torch.nn.functional as F

from source.reconstruction.utils import real_round, create_centered_zyx_grid
from source.reconstruction.symmetry import Symmetry


class BackProjector(torch.nn.Module):

    def __init__(self,
                 ori_size: int,
                 symmetry: Symmetry,
                 interpolator: bool = True,
                 padding_factor_3d: float = 2.0,
                 r_min_nn: int = 10):
        super(BackProjector, self).__init__()
        # Store original dimension
        self.ori_size = int(ori_size)
        self.register_buffer("r_max", torch.Tensor([self.ori_size / 2]), persistent=False)
        self.symmetry = symmetry #hydra.utils.instantiate(symmetry)
        self.symmetry.create_rotation_matrices()

        # Padding factor for the map
        if padding_factor_3d < 1.0:
            raise ValueError("Padding factor cannot be less than 1.")

        self.padding_factor = padding_factor_3d

        # Interpolation scheme
        self.interpolator = interpolator

        # Minimum radius for NN interpolation
        self.register_buffer("r_min_nn", torch.Tensor([r_min_nn]), persistent=False)

        self.dimx = self.ori_size + int(self.padding_factor)
        self.dimy = self.dimz = self.dimx * 2 - 1
        self.data = torch.zeros((self.dimz, self.dimy, self.dimx), dtype=torch.complex128).cpu()
        self.weight = torch.zeros((self.dimz, self.dimy, self.dimx), dtype=torch.double).cpu()
        self.weight_precalculated = False
        self.register_buffer("init_coords", torch.tensor([1 - self.dimx, 1 - self.dimx, 0], dtype=torch.long), persistent=False)
        self.register_buffer("offset_combinations", torch.tensor(list(itertools.product([0, 1], repeat=3)), dtype=torch.long), persistent=False)

        self.reconstruction_weights = None
        self.weighted_data = None
        self.r2_mask = None

    def backproject2Dto3D(self, f2d: torch.Tensor, A: torch.Tensor, Mweight: torch.Tensor):
        _, _, H, W = f2d.shape
        m = torch.eye(2, dtype=torch.double, device=A.device)

        max_r2 = (self.r_max * self.padding_factor) ** 2
        min_r2_nn = (self.r_min_nn * self.padding_factor) ** 2

        Ainv = A.mT * self.padding_factor
        Am = Ainv[..., :2] @ m
        AtA = torch.bmm(Am.mT, Am)
        AtA_xx, AtA_xy, AtA_yy = AtA[:, 0, 0].unsqueeze(1), AtA[:, 0, 1].unsqueeze(1), AtA[:, 1, 1].unsqueeze(1)
        AtA_xy2 = AtA_xy ** 2

        y = torch.cat((torch.arange(W, device=f2d.device, dtype=torch.double),
                       torch.arange(start=W - H, end=0, device=f2d.device, dtype=torch.double)), 0)
        y2 = y ** 2
        discr = AtA_xy2 * y2 - AtA_xx * (AtA_yy * y2 - max_r2)
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
        mask = (x_grid >= first_x) & (x_grid <= last_x) & (Mweight > 0.).prod(dim=1).bool() & (r2_3D <= max_r2) & (
                    discr.unsqueeze(-1) >= 0.)
        mask = mask.reshape(-1)

        if self.interpolator or r2_3D.lt(min_r2_nn).all():
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

    def project3Dto2D(self, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # H = self.weighted_data.shape[-2] // 2
        # W = H // 2 + 1
        W = self.weighted_data.shape[-1] // 2
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
    
        real = F.grid_sample(self.weighted_data.real[None, None, ...].expand(A.shape[0], -1, -1, -1, -1), 
                             p, align_corners=True, padding_mode='zeros')[:,:,0]
        imag = F.grid_sample(self.weighted_data.imag[None, None, ...].expand(A.shape[0], -1, -1, -1, -1), 
                             p, align_corners=True, padding_mode='zeros')[:,:,0]
        rotated_data = torch.complex(real, imag).to(device=A.device)
        rotated_data = torch.nan_to_num(torch.where(neg_x.unsqueeze(1), torch.conj(rotated_data), rotated_data), nan=0)
        # weights = F.grid_sample(self.reconstruction_weights[None, None, ...].expand(A.shape[0], -1, -1, -1, -1), 
        #                         p, align_corners=True, padding_mode='zeros')[:,:,0]
        return rotated_data#, weights
    
    def symmetrise(self):
        self.enforce_hermitian_symmetry()
        self.apply_point_group_symmetry()

    def enforce_hermitian_symmetry(self):
        d = self.data[:, :, self.init_coords[..., 2].item()]
        self.data[:, :, self.init_coords[..., 2].item()] += d.conj().resolve_conj().flip(dims=[0, 1])
        if not self.weight_precalculated:
            w = self.weight[:, :, self.init_coords[..., 2].item()]
            self.weight[:, :, self.init_coords[..., 2].item()] += w.flip(dims=[0, 1])

    def apply_point_group_symmetry(self):
        rmax2 = (real_round(self.r_max * self.padding_factor) ** 2).to(device=self.data.device)
        zyx = create_centered_zyx_grid(self.dimz, self.dimy, self.dimx).to(device=self.data.device)
        r2 = (zyx ** 2).sum(-1)
        self.r2_mask = (r2 <= rmax2)
        init_coords = self.init_coords.cpu().clone()[None, None, ...]
        init_coords[..., 2] = 0
        tmp_data = self.data.clone()[None, None, ...]
        if not self.weight_precalculated:
            tmp_weight = self.weight.clone()[None, None, ...]

        for r in self.symmetry.rr:
            r = r[:3, :3].to(device=self.data.device)
            grid = torch.einsum('abcd,ed->abce', zyx, r.flip([0, 1]))
            neg_x = (1 - 2 * (grid[..., 2] < 0)).double().unsqueeze(-1)
            grid *= neg_x
            grid -= init_coords
            grid /= torch.tensor([(self.dimz - 1), (self.dimy - 1), (self.dimx - 1)])/ 2
            grid = grid.flip(-1)[None, ...] - 1

            real = F.grid_sample(tmp_data.real, grid, align_corners=True, padding_mode='zeros')
            imag = F.grid_sample(tmp_data.imag, grid, align_corners=True, padding_mode='zeros') * neg_x[None, None , ..., 0]
            rotated_data = torch.complex(real, imag)
            self.data += rotated_data[0, 0]
            if not self.weight_precalculated:
                rotated_weight = F.grid_sample(tmp_weight, grid, align_corners=True, padding_mode='zeros')
                self.weight += rotated_weight[0, 0]

    def decenter(self, input: torch.Tensor) -> torch.Tensor:
        output = input.clone()
        output[~self.r2_mask] = 0.
        output = output.roll(shifts=self.dimx, dims=0).roll(shifts=self.dimx, dims=1)
        return output

    def mask_background(self, input: torch.Tensor, cosine_width: int = 3) -> torch.Tensor:
        radius = input.shape[0] // 2
        x = torch.arange(-radius, radius, device=self.data.device, dtype=torch.double)
        y = torch.arange(-radius, radius, device=self.data.device, dtype=torch.double)
        z = torch.arange(-radius, radius, device=self.data.device, dtype=torch.double)
        z_grid, y_grid, x_grid = torch.meshgrid(z, y, x, indexing="ij")
        zyx = torch.stack((z_grid, y_grid, x_grid), dim=-1)
        r = (zyx ** 2).sum(-1).sqrt()
        radius_p = radius + cosine_width
        background_mask = r > radius_p
        semi_background_mask = (r > radius) & ~background_mask

        raisedcos = 0.5 + 0.5 * torch.cos((radius_p - r) * torch.pi / cosine_width)
        sum_bg = (input * background_mask + input * semi_background_mask * raisedcos).sum()
        count_bg = background_mask.sum() + raisedcos[semi_background_mask].sum()
        sum_bg /= count_bg
        background = background_mask * sum_bg + semi_background_mask * raisedcos * sum_bg
        output = input * ~(semi_background_mask | background_mask) + semi_background_mask * (
                    1 - raisedcos) * input + background

        rval = r / (self.ori_size * self.padding_factor)
        sinc2 = (torch.sin(torch.pi * rval) / (torch.pi * rval)) ** 2
        sinc2[r == 0] = 1

        return output / sinc2

    def reconstruct(self) -> torch.Tensor:
        f_weight = self.decenter(self.weight)
        self.weighted_data = self.decenter(self.data)
        o1 = -1
        o2 = 2
        if self.reconstruction_weights is None:
            round_max_r2 = real_round((self.r_max * self.padding_factor) ** 2).cpu()
            zyx = create_centered_zyx_grid(self.dimz, self.dimy, self.dimx).roll(shifts=self.dimx, dims=0).roll(shifts=self.dimx, dims=1)
            r2 = (zyx ** 2).sum(-1).view(-1)
    
            mask = r2 < round_max_r2
            ires = torch.floor(r2[mask].sqrt() / self.padding_factor).long().cpu()
    
            f_val = f_weight.view(-1)[mask]
    
            radavg_weight = torch.zeros([self.r_max.long().item()], dtype=torch.double)
            radavg_weight.index_add_(0, ires, f_val)
    
            counter = torch.zeros([self.r_max.long().cpu().item()], dtype=torch.double)
            counter.index_add_(0, ires, torch.ones_like(ires, dtype=torch.double))
            counter[counter <= 0] = 1
            radavg_weight /= counter
            radavg_weight /= 1000
    
            ires = ires.clamp_max(self.r_max.cpu().item() - 1).long()
            radavg_weight = radavg_weight.index_select(0, ires)
            weight_selected = torch.zeros_like(f_weight).view(-1)
            weight_selected[r2 < round_max_r2] = radavg_weight
            self.reconstruction_weights = torch.stack((weight_selected.view(f_weight.shape), f_weight), dim=-1).max(dim=-1)[0]
            self.reconstruction_weights[self.reconstruction_weights == 0.] = 1.
        # self.weighted_data /= self.reconstruction_weights
            self.reconstruction_weights = self.reconstruction_weights.roll(shifts=-self.dimx, dims=0).roll(shifts=-self.dimx, dims=1)
        self.weighted_data = self.weighted_data.roll(shifts=-self.dimx, dims=0).roll(shifts=-self.dimx, dims=1)
        self.weighted_data /= self.reconstruction_weights
        window_data = self.weighted_data[o2:o1, o2:o1, :-1].clone().roll(shifts=self.dimx + o1, dims=1).roll(shifts=self.dimx + o1, dims=0)
        window_data[1::2] *= -1
        if window_data.shape[-2] % 2 == 0:
            window_data[:, 1::2] *= -1
        if window_data.shape[-1] % 2 == 1:
            window_data[:, :, 1::2] *= -1
        transformed = torch.fft.irfftn(window_data, norm="forward") / (self.padding_factor ** 3 * self.ori_size)
        transformed = transformed[(transformed.shape[0] // 4):(transformed.shape[0] // 4) * 3,
                      (transformed.shape[1] // 4):(transformed.shape[1] // 4) * 3,
                      (transformed.shape[2] // 4):(transformed.shape[2] // 4) * 3]
        masked = self.mask_background(transformed)
        
        return masked
