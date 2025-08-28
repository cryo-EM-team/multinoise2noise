from torchmetrics import Metric
from source.reconstruction.utils import real_round
import torch


class FourierRingCorrelation(Metric):
    def __init__(self, size: int, **kwargs):
        # Initialize both parent classes.
        super().__init__(**kwargs)

        coords = torch.stack(torch.meshgrid(
            torch.fft.fftfreq(size, 1/size, dtype=torch.double),
            torch.arange(size//2+1, dtype=torch.double),
            indexing='ij'
        ), dim=-1)

        r = (coords ** 2).sum(dim=-1).sqrt()
        r = r.flatten()
        self.mask = r <= (size // 2)
        r = real_round(r[self.mask])
        r = r.long()
        self.register_buffer("rings", torch.nn.functional.one_hot(r).float(), persistent=False)
        self.ranges = torch.arange(self.rings.shape[-1])[None, :].float()
        self.ranges /= self.ranges.sum()

        # State variables to accumulate loss sum and count.
        self.add_state("frc", default=torch.zeros((self.rings.shape[-1])), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss on the current batch, update metric state,
        and return the loss tensor for backpropagation.
        For example, here we implement a Mean Squared Error loss.
        """
        f_preds = torch.fft.rfft2(preds, norm="forward").flatten(start_dim=-2)[:, :, self.mask]
        f_target = torch.fft.rfft2(target, norm="forward").flatten(start_dim=-2)[:, :, self.mask]

        norm_preds = (f_preds.real ** 2 + f_preds.imag ** 2)[None, :]
        norm_target = (f_target.real ** 2 + f_target.imag ** 2)[None, :]
        norm_both = (f_preds.real * f_target.real + f_preds.imag * f_target.imag)[None, :]
        norm_preds = norm_preds @ self.rings
        norm_target = norm_target @ self.rings
        norm_both = norm_both @ self.rings

        frc = torch.nan_to_num(norm_both / torch.sqrt(norm_preds * norm_target), 0)[0] #/ batch_size
        frc = frc.mean(dim=(-3,-2)).flip([0])
        self.frc += frc
        #loss = self.ranges @ frc
        return frc

    def compute(self) -> torch.Tensor:
        """Compute the aggregated loss over all batches."""
        return self.frc
        # return self.ranges @ self.frc