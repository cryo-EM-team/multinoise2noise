import torch


class DoseWeighter:
    def __init__(self, n_frames: int, dose_per_frame: float, pre_exposure: float, voltage: int, pixel_size: float):
        self.n_frames = n_frames
        self.dose_per_frame = dose_per_frame
        self.pre_exposure = pre_exposure
        self.voltage = voltage
        self.pixel_size = pixel_size
        self.dose_weighting_params = {'A': 0.245, 'B': -1.665, 'C': 2.81}
        self.doses = self.create_doses()
        self.weight = None

    def create_doses(self):
        doses = torch.arange(1, self.n_frames + 1, dtype=torch.double)
        doses *= self.dose_per_frame
        doses += self.pre_exposure
        doses /= 0.8 ** (3 - self.voltage//100)
        return doses

    def dose_weighting(self, f_frames: torch.Tensor) -> torch.Tensor:
        if self.weight is None:
            _, nfy, nfx = f_frames.shape
            nfy2 = nfy ** 2
            nfx2 = ((nfx - 1) ** 2) * 4
            self.doses = self.doses.to(device=f_frames.device)
            y2 = torch.concat([
                torch.arange(0, nfy // 2 + 1, device=f_frames.device, dtype=torch.double), 
                torch.arange(-(nfy // 2) + 1, 0, device=f_frames.device, dtype=torch.double)]).square() / nfy2
            x2 = torch.arange(0, nfx, device=f_frames.device, dtype=torch.double).square() / nfx2
            doses_grid, y_grid, x_grid = torch.meshgrid(self.doses, y2, x2, indexing="ij")
            dinv = torch.sqrt(y_grid + x_grid) / self.pixel_size
            Ne = (self.dose_weighting_params['A'] * torch.pow(dinv, self.dose_weighting_params['B']) + self.dose_weighting_params['C']) * 2
            self.weight = torch.exp(-doses_grid / Ne)
            sum_norm = self.weight.square().sum(dim=0, keepdim=True).sqrt()
            self.weight /= sum_norm
        return f_frames * self.weight
