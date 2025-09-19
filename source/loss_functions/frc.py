from torchmetrics import Metric
from source.reconstruction.utils import real_round
import torch


class FourierRingCorrelation(Metric):
    """
    Metric for computing the Fourier Ring Correlation (FRC) between two images.

    This class accumulates FRC values over batches and provides methods to update and compute the metric.

    Attributes:
        mask (torch.Tensor): Boolean mask for valid frequency rings.
        rings (torch.Tensor): One-hot encoded ring indices for frequency bins.
        ranges (torch.Tensor): Normalized range tensor for weighting.
        frc (torch.Tensor): Accumulated FRC values.
    """
    def __init__(self, size: int, **kwargs):
        """
        Initialize the FourierRingCorrelation metric.

        Args:
            size (int): Size of the input images (assumed square).
            **kwargs: Additional keyword arguments for torchmetrics.Metric.
        """
        super().__init__(**kwargs)

        coords = torch.stack(torch.meshgrid(
            torch.fft.fftfreq(size, 1/size, dtype=torch.float32),
            torch.arange(size//2+1, dtype=torch.float32),
            indexing='ij'
        ), dim=-1)

        r = (coords ** 2).sum(dim=-1).sqrt()
        r = r.flatten()
        self.mask = r <= (size // 2)
        self.register_buffer("rings", real_round(r[self.mask]).long(), persistent=False)
        self.add_state("frc", default=torch.zeros(size // 2 + 1), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(1), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Update the FRC metric with a new batch of predictions and targets.

        Args:
            preds (torch.Tensor): Predicted images (batch of 2D tensors).
            target (torch.Tensor): Target images (batch of 2D tensors).

        Returns:
            torch.Tensor: FRC values for the current batch.
        """
        f_preds = torch.fft.rfft2(preds, norm="forward").flatten(start_dim=-2).flatten(start_dim=0, end_dim=2)[:, self.mask]
        f_target = torch.fft.rfft2(target, norm="forward").flatten(start_dim=-2).flatten(start_dim=0, end_dim=2)[:, self.mask]

        norm_preds = (f_preds.real ** 2 + f_preds.imag ** 2)
        norm_target = (f_target.real ** 2 + f_target.imag ** 2)
        norm_both = (f_preds.real * f_target.real + f_preds.imag * f_target.imag)

        n = int(torch.prod(torch.tensor(preds.shape[:-2])))
        num = torch.zeros((n, self.frc.shape[0]), device=preds.device, dtype=preds.dtype)
        den1 = torch.zeros_like(num)
        den2 = torch.zeros_like(num)
        for i in range(n):
            num[i].index_add_(0, self.rings, norm_both[i])
            den1[i].index_add_(0, self.rings, norm_preds[i])
            den2[i].index_add_(0, self.rings, norm_target[i])

        frc = torch.nan_to_num(num / torch.sqrt(den1 * den2), 0).mean(dim=0) * preds.shape[0]

        self.frc += frc
        self.total += preds.shape[0]
        return frc / preds.shape[0]

    def compute(self) -> torch.Tensor:
        """
        Compute the aggregated FRC over all batches.

        Returns:
            torch.Tensor: Aggregated FRC values.
        """
        return self.frc / self.total
