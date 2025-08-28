import torch


class DoseWeighter(torch.nn.Module):
    def __init__(self, n_frames: int, dose_per_frame: float, pre_exposure: float, voltage: int, pixel_size: float, size: tuple[int, int]):
        super(DoseWeighter, self).__init__()
        self.n_frames = n_frames
        self.dose_per_frame = dose_per_frame
        self.pre_exposure = pre_exposure
        self.voltage = voltage
        self.pixel_size = pixel_size
        self.size = size
        self.dose_weighting_params = {'A': 0.245, 'B': -1.665, 'C': 2.81}
        self.doses = self.create_doses()
        self.register_buffer("weight", self.create_weights(), persistent=False)

    def create_doses(self):
        doses = torch.arange(1, self.n_frames + 1, dtype=torch.double)
        doses *= self.dose_per_frame
        doses += self.pre_exposure
        doses /= 0.8 ** (3 - self.voltage//100)
        return doses
    
    def create_weights(self) -> torch.Tensor:
        nfy, nfx = self.size[0], self.size[1] // 2 + 1
        nfy2 = nfy ** 2
        nfx2 = ((nfx - 1) ** 2) * 4
        self.doses = self.doses
        y2 = torch.concat([
            torch.arange(0, nfy // 2 + 1, dtype=torch.double), 
            torch.arange(-(nfy // 2) + 1, 0, dtype=torch.double)]).square() / nfy2
        x2 = torch.arange(0, nfx, dtype=torch.double).square() / nfx2
        doses_grid, y_grid, x_grid = torch.meshgrid(self.doses, y2, x2, indexing="ij")
        dinv = torch.sqrt(y_grid + x_grid) / self.pixel_size
        Ne = (self.dose_weighting_params['A'] * torch.pow(dinv, self.dose_weighting_params['B']) + self.dose_weighting_params['C']) * 2
        weight = torch.exp(-doses_grid / Ne)
        sum_norm = weight.square().sum(dim=0, keepdim=True).sqrt()
        weight /= sum_norm
        return weight
