import torch
import hydra

from source._noise2noise.multinoise2noise import MultiNoise2NoiseLightningModel


class MultiNoise2NoiseCTFLightningModel(MultiNoise2NoiseLightningModel):
    def __init__(self, *args, **kwargs) -> None:
        super(MultiNoise2NoiseCTFLightningModel, self).__init__()

    def exchange_data(self, **kwargs):
        self.criterion.loss_div = self.criterion.loss_div * kwargs['datamodule'].ctf_mean(self.reconstructor.extractor.ctf)
    
    def _shared_step(self, x1: torch.Tensor, x2: torch.Tensor, ctf: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        x1 = self.flatten_tensor(x1)
        denoised = self.model(x1)
        denoised = self.restore_tensor(denoised)
        loss = self.criterion(x2, denoised, ctf)
        return loss, denoised

    def training_step(
        self, 
        batch: list[dict[str, torch.Tensor]], 
        *args: object, 
        **kwargs: object
    ) -> torch.Tensor:
        loss = 0
        for i in range(len(batch)):
            ctf_weights = self.reconstructor.extractor.ctf.get_fftw_image(**batch[i]['ctf_params']).detach().float().abs()[:, None, ...]
            if self.hparams.ctf_premultiply:
                batch[i]['input'] = self.correct_ctf([i]['input'], ctf_weights)
            loss += self._shared_step(x1=batch[i]['input'], x2=batch[i]['target'], ctf=ctf_weights)[0]
        loss /= len(batch)

        self.log_dict(
            {f'train/loss': loss.detach().item()},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return loss

    def _evaluation_step(
        self, 
        batch: dict[str, torch.Tensor], 
        phase: str, 
        dataloader_idx: int,
        *args: object, 
        **kwargs: object
    ) -> dict[str, list[torch.Tensor]]:
        ctf_weights = self.reconstructor.extractor.ctf.get_fftw_image(**batch['ctf_params']).detach().float().abs()[:, None, ...]
        if self.hparams.ctf_premultiply:
            batch['input'] = self.correct_ctf(batch['input'], ctf_weights)
            batch['full_dose'] = self.correct_ctf(batch['full_dose'], ctf_weights)
        
        loss, batch['denoised'] = self._shared_step(x1=batch['input'], x2=batch['target'], ctf=ctf_weights)
        _, batch['denoised_full_dose'] = self._shared_step(x1=batch['full_dose'], x2=batch['target'][:, 0:1, ...], ctf=ctf_weights)
        batch['denoised_mean'] = batch['denoised'].mean(dim=1, keepdim=True)

        self.backproject_denoised(batch, bckp_idx=dataloader_idx, phase=phase)
        if self.hparams.eval_half != dataloader_idx+1:
            self.compute_metrics(batch, loss, phase=phase)
            self.store_images(batch)

    @staticmethod
    def correct_ctf(self, data: torch.Tensor, ctf: torch.Tensor) -> torch.Tensor:
        """
        Correct the sign of the data based on the CTF weights.
        """
        f_data = torch.fft.rfft2(data, norm='forward') * ctf
        data = torch.fft.irfft2(f_data, norm='forward')
        return data
