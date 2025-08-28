import torch


class CTF_MSELoss(torch.nn.Module):
    def __init__(self, n: int = 1, correct_frequencies: bool = True, **kwargs):
        super().__init__()
        self.n = n
        self.correct_frequencies = correct_frequencies
        self.register_buffer("loss_div", torch.ones((1,1,1,1,1)), persistent=False)
    
    def forward(self, preds: torch.Tensor, target: torch.Tensor, ctf: torch.Tensor) -> torch.Tensor:
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
        
        # f_preds = torch.fft.rfft2(preds, norm="ortho")
        # f_target = torch.fft.rfft2(target, norm="ortho")
        # loss = torch.abs(f_preds - f_target) ** 2
        # #loss = (self.base_loss(f_preds.real, f_target.real) + self.base_loss(f_preds.imag, f_target.imag)) / 2
        # if self.n % 2 == 1:
        #     ctf = ctf.abs()
        # loss = loss / 2 * (ctf ** self.n)
        # return loss.mean()
