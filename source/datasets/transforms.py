import torch


class Rotation90:
    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return torch.rot90(x, k=torch.randint(low=0, high=4, size=(1, 1))[0][0], dims=(-1, -2))
    
class Cutoff:
    def __init__(self, cutoff: float):
        self.cutoff = cutoff

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x[(x < -self.cutoff) | (x > self.cutoff)] = 0
        return x