import torch

from source.reconstruction.dose_weighter import DoseWeighter


class DW_MSELoss(torch.nn.Module):
    """
    Mean Squared Error loss function with dose weighting for electron microscopy images.

    This class applies dose weighting to the predicted and target images in the Fourier domain
    before computing the loss. Optionally, regularization can be applied to encourage similarity
    between frames.

    Attributes:
        dose_weighter (DoseWeighter): Module for dose weighting.
        correct_frequencies (bool): Whether to apply frequency correction.
        regularization (float): Regularization strength for inter-frame similarity.
        frames_weight_sum (torch.Tensor): Sum of absolute dose weights across frames.
    """
    def __init__(self, dose_weighter: DoseWeighter, correct_frequencies: bool = True, regularization: float = 0., *args, **kwargs):
        """
        Initialize the DW_MSELoss module.

        Args:
            dose_weighter (DoseWeighter): Dose weighting module.
            correct_frequencies (bool, optional): Whether to apply frequency correction. Defaults to True.
            regularization (float, optional): Regularization strength for inter-frame similarity. Defaults to 0.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.correct_frequencies = correct_frequencies
        self.regularization = regularization
        self.dose_weighter = dose_weighter

        self.dose_weighter.weight = self.dose_weighter.weight[None, :, None, ...]
        self.register_buffer("frames_weight_sum", 
                             self.dose_weighter.weight.abs().sum(dim=-4, keepdim=True), 
                             persistent=False)
    
    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the dose-weighted MSE loss between predictions and targets.

        Args:
            preds (torch.Tensor): Predicted images (real or complex tensor).
            target (torch.Tensor): Target images (real or complex tensor).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        if preds.dtype == torch.complex64:
            f_preds = preds
        else:
            f_preds = torch.fft.rfft2(preds, norm="ortho")

        if f_preds.shape[-4] == self.dose_weighter.weight.shape[-4]:
            f_preds *= self.dose_weighter.weight
        else:
            f_preds *= self.frames_weight_sum
        if target.dtype == torch.complex64:
            f_target = target
        else:
            f_target = torch.fft.rfft2(target, norm="ortho")
        loss = torch.nn.functional.mse_loss(f_preds.imag, f_target.imag, reduction='none') + torch.nn.functional.mse_loss(f_preds.real, f_target.real, reduction='none')

        if self.correct_frequencies:
            loss = loss / self.frames_weight_sum
        loss = loss.mean()
        if self.regularization > 0.:
            for i in range(f_preds.shape[-4]):
                for j in range(i+1, f_preds.shape[-4]):
                    loss += (torch.nn.functional.mse_loss(f_preds[..., i, :, :, :].real, f_preds[..., j, :, :, :].real, reduction='mean') + torch.nn.functional.mse_loss(f_preds[..., i, :, :, :].imag, f_preds[..., j, :, :, :].imag, reduction='mean')) * self.regularization / (f_preds.shape[-4] * (f_preds.shape[-4] - 1) * 2)
        return loss
