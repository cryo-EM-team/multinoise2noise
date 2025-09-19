import torch


class CTF_MSELoss(torch.nn.Module):
    """
    Mean Squared Error loss function with CTF and dose weighting for electron microscopy images.

    Inherits from CTF_MSELoss and applies dose weighting to the predicted and target images
    in the Fourier domain before computing the loss.

    Attributes:
        dose_weighter (DoseWeighter): Module for dose weighting.
        frames_weight_sum (torch.Tensor): Sum of absolute dose weights across frames.
        loss_div (torch.Tensor): Frequency correction divisor.
    """
    def __init__(self, n: int = 1, correct_frequencies: bool = True, **kwargs):
        """
        Initialize the CTF_DW_MSELoss module.

        Args:
            dose_weighter (DoseWeighter): Dose weighting module.
            n (int, optional): Power to raise CTF to in the loss. Defaults to 1.
            correct_frequencies (bool, optional): Whether to apply frequency correction. Defaults to True.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.n = n
        self.correct_frequencies = correct_frequencies
        self.register_buffer("loss_div", torch.ones((1,1,1,1,1)), persistent=False)
    
    def forward(self, preds: torch.Tensor, target: torch.Tensor, ctf: torch.Tensor) -> torch.Tensor:
        """
        Compute the dose-weighted CTF MSE loss between predictions and targets.

        Args:
            preds (torch.Tensor): Predicted images (real or complex tensor).
            target (torch.Tensor): Target images (real or complex tensor).
            ctf (torch.Tensor): CTF weighting tensor.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        if preds.dtype == torch.complex64:
            f_preds = preds
        else:
            f_preds = torch.fft.rfft2(preds, norm="ortho")
        if target.dtype == torch.complex64:
            f_target = target
        else:
            f_target = torch.fft.rfft2(target, norm="ortho")
        loss = ((f_preds.real - f_target.real) ** 2 + 
                (f_preds.imag - f_target.imag) ** 2) * (ctf ** self.n)
        if self.correct_frequencies:
            loss = loss / self.loss_div
        loss = loss.mean()
        return loss
