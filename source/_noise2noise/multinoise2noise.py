import torch
import hydra
import lightning as pl
from matplotlib.figure import Figure
from PIL import Image
from omegaconf import DictConfig
from torch.optim import Optimizer
from matplotlib.backends.backend_agg import FigureCanvasAgg
import mrcfile
import os
import numpy as np
import matplotlib.pyplot as plt

from source.reconstruction.visualization import visualize_slices, visualize_projections, visualize_fsc, visualize_samples, visualize_guiner, visualize_histogram
from source.reconstruction.postprocessing import postprocess
from source.reconstruction.utils import calculate_resolution

from source.utils import RankedLogger


logger = RankedLogger(__name__)


class MultiNoise2NoiseLightningModel(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super(MultiNoise2NoiseLightningModel, self).__init__()
        self.save_hyperparameters(logger=False)

        # instantiate model
        self.model = hydra.utils.instantiate(self.hparams.model)

        # instantiate loss function
        self.criterion = hydra.utils.instantiate(self.hparams.criterion)
        
        self.reconstructor = hydra.utils.instantiate(self.hparams.reconstructor)
        
        self.metrics_set = torch.nn.ModuleDict({
            'singular': hydra.utils.instantiate(self.hparams.metrics),
            'mean': hydra.utils.instantiate(self.hparams.metrics),
            'full_dose': hydra.utils.instantiate(self.hparams.metrics)
        })

        # validation step outputs and test step outputs for logging
        self.last_shape = None
        self._set_eval_outputs()
        self.reconstruction_condition = False
        self.logging_condition = None
        self.original_reconstruction = torch.tensor(mrcfile.open(self.hparams.original_reconstruction_path, permissive=True).data, dtype=torch.double)

    def exchange_data(self, **kwargs):
        pass
    
    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.hparams.std + self.hparams.mean

    def _set_eval_outputs(self):
        self.evaluation_step_outputs = {
            "input": [],
            "target": [],
            "full_dose": [],
            "denoised": [],
            "denoised_mean": [],
            "denoised_full_dose": [],
            "clean" : []
        }
        self.cummulative_loss = 0
        self.batch_counter = 0

    def _process_eval_outputs(self) -> None:
        for k in [k for k, v in self.evaluation_step_outputs.items() if not v]:
            self.evaluation_step_outputs.pop(k, None)
        for k, v in self.evaluation_step_outputs.items():
            self.evaluation_step_outputs[k] = torch.stack(v, axis=0)
            for e in v:
                del e
            del v

    def compute_metrics(self, batch: dict[str, torch.Tensor], loss: torch.Tensor, phase: str) -> None:
        with torch.no_grad():
            self.cummulative_loss += loss.item()
            self.batch_counter += 1
            if 'clean' in batch.keys():
                clean = normalize_for_metrics(self._denormalize(batch["clean"]))
                for key, metric in self.metrics_set.items():
                    if key == 'singular':
                        metric(normalize_for_metrics(batch['denoised']), 
                            clean.repeat(1, batch['denoised'].shape[1], 1, 1, 1))
                    else:
                        metric(normalize_for_metrics(batch[f'denoised_{key}']), clean)

    def _log_metrics_directly(self, phase: str, evaluation_metrics_outputs: dict[str, torch.Tensor]):
        evaluation_metrics_outputs.update({f"loss": self.cummulative_loss / self.batch_counter})
        for key, metric_set in self.metrics_set.items():
            evaluation_metrics_outputs.update(self._add_prefix_to_metrics(f'{key}_', metric_set.compute()))
            metric_set.reset()
        self.cummulative_loss = 0
        self.batch_counter = 0
        evaluation_metrics_outputs = self._add_prefix_to_metrics(f'{phase}/', evaluation_metrics_outputs)
        for key, value in evaluation_metrics_outputs.items():
            if "FourierRingCorrelation"in key and self.logging_condition:
                self.log_plot(
                    visualize_fsc(
                        {key: value}, 
                        calculate_resolution(value, self.reconstructor.extractor.ctf.angpix), 
                        self.reconstructor.extractor.ctf.angpix),
                    key)
            else:
                self.log(key, value, prog_bar=True, sync_dist=True)

    def log_plot(self, fig: Figure, title: str) -> None:
        """
        Logs a matplotlib figure to the logger.

        Args:
            fig (Figure): The matplotlib figure to log.
            title (str): The title for the logged image.
        """
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        img = Image.frombytes("RGBA", canvas.get_width_height(), canvas.buffer_rgba())
        self.logger.log_image(key=title, images=[img])
        plt.close()
        del canvas
        del img
        del fig

    def store_images(self, batch: dict[str, torch.Tensor]) -> None:
        if len(self.evaluation_step_outputs["input"]) < self.hparams.logging.sample_epochs:
            for key in self.evaluation_step_outputs.keys():
                if key in batch.keys():
                    self.evaluation_step_outputs[key].append(batch[key][0, 0, 0].detach().cpu())

    def configure_optimizers(
        self,
    ) -> list[Optimizer] | tuple[list[Optimizer] | list[object]]:
        """Configure optimizer and lr scheduler.
        Returns:
            list[Optimizer] | tuple[list[Optimizer], list[object]]:
                Optimizer or optimizer and lr scheduler.
        """
        params = self.model.parameters()
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer, params=params, _convert_="partial"
        )
        if "lr_scheduler" not in self.hparams:
            return [optimizer]
        scheduler = hydra.utils.instantiate(
            self.hparams.lr_scheduler, optimizer=optimizer, _convert_="partial"
        )
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": self.hparams.monitor,
            }
        return [optimizer], [scheduler]

    def on_fit_start(self) -> None:
        self.logging_condition = self.logger is not None and not isinstance(self.logger, pl.pytorch.loggers.csv_logs.CSVLogger)

    def on_test_start(self) -> None:
        self.logging_condition = self.logger is not None and not isinstance(self.logger, pl.pytorch.loggers.csv_logs.CSVLogger)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        denoised = self.restore_tensor(self.model(self.flatten_tensor(x1)))
        loss = self.criterion(x2, denoised)
        return loss, denoised

    def flatten_tensor(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Reshapes a tensor of shape (B, S, C, H, W) into a tensor of shape ((B * S), 1, C, H, W).

        Args:
            input_tensor (torch.Tensor): The input tensor with shape (B, S, C, H, W).

        Returns:
            torch.Tensor: The flattened tensor with shape ((B * S), 1, C, H, W).
        """
        self.last_shape = input_tensor.shape
        B, S, C, H, W = input_tensor.shape
        return input_tensor.view(B * S, C, H, W)
    
    def restore_tensor(self, flattened_tensor: torch.Tensor) -> torch.Tensor:
        """
        Reshapes a tensor of shape ((B * S), 1, C, H, W) back to (B, S, C, H, W).

        Args:
            flattened_tensor (torch.Tensor): The flattened tensor with shape ((B * S), 1, C, H, W).

        Returns:
            torch.Tensor: The restored tensor with shape (B, S, C, H, W).
        """
        if flattened_tensor.shape[1] != 1:
            raise ValueError("Expected the second dimension to be 1, got {}".format(flattened_tensor.shape[1]))
        
        return flattened_tensor.view(*self.last_shape)

    @staticmethod
    def _add_prefix_to_metrics(prefix: str, logs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Add prefix to all keys in dictionary
        Args:
            prefix: Train, val or test
            logs: Dictionary with measured metrics

        Returns:
            Dictionary with added prefixes
        """
        logs = {(prefix + key): value for key, value in logs.items()}
        return logs

    def training_step(
        self, 
        batch: dict[str, torch.Tensor], 
        *args: object, 
        **kwargs: object
    ) -> torch.Tensor:
        loss = 0
        for i in range(len(batch)):
            loss += self._shared_step(x1=batch[i]['input'], x2=batch[i]['target'])[0]
        loss /= len(batch)

        self.log_dict(
            {f'train/loss': loss.detach().item()},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return loss

    def on_validation_epoch_start(self) -> None:
        self._on_evaluation_epoch_start("val")

    def on_test_epoch_start(self) -> None:
        self._on_evaluation_epoch_start("test")

    def _on_evaluation_epoch_start(self, phase: str) -> None:
        self.reconstruction_condition = (phase == 'test' or 
                                         (self.current_epoch >= self.hparams.logging.reconstruction_wait and 
                                          self.current_epoch % self.hparams.logging.reconstruction_step == 0))

    def _evaluation_step(
        self, 
        batch: dict[str, torch.Tensor], 
        phase: str, 
        dataloader_idx: int,
        *args: object, 
        **kwargs: object
    ) -> None:
        if self.hparams.eval_half != dataloader_idx+1 and self.logging_condition:
            loss, batch['denoised'] = self._shared_step(x1=batch['input'], x2=batch['target'])
            _, batch['denoised_full_dose'] = self._shared_step(x1=batch['full_dose'], x2=batch['target'][:, 0:1, ...])
            batch['denoised_mean'] = batch['denoised'].mean(dim=1, keepdim=True)

            self.compute_metrics(batch, loss, phase=phase)
            self.store_images(batch)

        if self.reconstruction_condition:
            if not (self.hparams.eval_half != dataloader_idx+1 and self.logging_condition):
                if self.hparams.reconstruction_mode == 'denoised_full_dose':
                    _, batch['denoised_full_dose'] = self._shared_step(x1=batch['full_dose'], x2=batch['target'][:, 0:1, ...])
                elif self.hparams.reconstruction_mode == 'denoised_mean':
                    loss, batch['denoised'] = self._shared_step(x1=batch['input'], x2=batch['target'])
                    batch['denoised_mean'] = batch['denoised'].mean(dim=1, keepdim=True)

            self.reconstructor.forward(image=batch[self.hparams.reconstruction_mode][:, 0].double(),
                                       shifts=batch['shifts'], 
                                       angles_matrix=batch['angles_matrix'],
                                       ctf_params=batch['ctf_params'], 
                                       fom=batch['fom'].double(), 
                                       bckp_idx=dataloader_idx)

    def validation_step(self, 
                        batch: dict[str, torch.Tensor], 
                        batch_idx: int,
                        dataloader_idx: int,
                        *args: object, 
                        **kwargs: object):
        self._evaluation_step(batch=batch, phase='val', dataloader_idx=dataloader_idx, args=args, kwargs=kwargs)

    def test_step(self, 
                  batch: dict[str, torch.Tensor], 
                  batch_idx: int, dataloader_idx: int, 
                  *args: object, 
                  **kwargs: object):
        self._evaluation_step(batch=batch, phase='test', dataloader_idx=dataloader_idx, args=args, kwargs=kwargs)

    def predict_step(self, 
                     batch: dict[str, torch.Tensor], 
                     batch_idx: int, 
                     dataloader_idx: int, 
                    *args: object, 
                    **kwargs: object):
        denoised = self.model(batch['full_dose'][:, 0])[:, 0]

        if self.reconstructor.hparams.ctf_mode == "abs":
            ctf = self.reconstructor.extractor.ctf.get_fftw_image(**batch['ctf_params']).detach().float().sign()[:, 0]
            f_denoised = torch.fft.rfft2(denoised, norm='forward') * ctf
            denoised = torch.fft.irfft2(f_denoised, norm='forward')

        for file, img in zip(batch['file'], denoised):
            idx, path = file.split('@')
            idx = int(idx) - 1
            path = os.path.join(self.reconstructor.hparams.output_dir, path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._set_tensor_to_mrc_slice(path, img, idx)

    def _set_tensor_to_mrc_slice(self, mrc_path: str, tensor2d: torch.Tensor, index: int) -> None:
        """
        Sets the values of a 2D torch tensor to a slice of a 3D MRC file at the given index.
        If the index is out of bounds, the MRC file is resized to accommodate the new slice.

        Args:
            mrc_path (str): Path to the MRC file.
            tensor2d (torch.Tensor): 2D tensor to write.
            index (int): Slice index to write to.
        """
        tensor_np = tensor2d.cpu().numpy().astype(np.float32)
        # If file does not exist, create a new one with a single slice
        if not os.path.exists(mrc_path):
            with mrcfile.new(mrc_path, overwrite=True) as mrc:
                mrc.set_data(np.zeros((index + 1, *tensor_np.shape), dtype=np.float32))
                mrc.data[index] = tensor_np
                mrc.update_header_stats()
            return

        with mrcfile.open(mrc_path, mode='r+') as mrc:
            data = mrc.data
            # Resize if index is out of bounds
            if index >= data.shape[0]:
                new_shape = (index + 1, data.shape[1], data.shape[2])
                new_data = np.zeros(new_shape, dtype=np.float32)
                new_data[:data.shape[0]] = data
                mrc.set_data(new_data)
                data = mrc.data
            data[index] = tensor_np

            mrc.voxel_size = self.reconstructor.extractor.ctf.angpix
            mrc.header.mx = data.shape[2]
            mrc.header.my = data.shape[1]
            mrc.header.mz = data.shape[0]
            mrc.update_header_stats()

    def _update_output_path(self) -> None:
        if self.hparams.reconstruction_mode not in self.reconstructor.hparams.output_dir:
            self.reconstructor.hparams.output_dir = os.path.join(self.hparams.reconstructor.output_dir, self.hparams.reconstruction_mode)
        self.reconstructor.hparams.output_dir = self.reconstructor.hparams.output_dir.replace(
            "denoised_full_dose", self.hparams.reconstruction_mode).replace(
                "denoised_mean", self.hparams.reconstruction_mode)

    def _on_evaluation_epoch_end(self, phase: str) -> None:
        with torch.no_grad():
            if self.logging_condition:
                self._process_eval_outputs()
                self.log_plot(visualize_samples(self.evaluation_step_outputs, self.hparams.logging.sample_epochs), f"{phase}/sample")
                for key in list(self.evaluation_step_outputs.keys()):
                    del self.evaluation_step_outputs[key]
                self.evaluation_step_outputs = {}
            
            evaluation_metrics_outputs = {}
            if self.reconstruction_condition:
                self._update_output_path()
                self.reconstructor.full_on_predict_epoch_end()
                reconstruction_results = self.reconstructor.finish_reconstruction()
                evaluation_metrics_outputs[f"{self.hparams.reconstruction_mode}/relative_resolution"] = reconstruction_results['resolution']
                if self.logging_condition:
                    self.log_plot(visualize_slices(reconstruction_results['reconstruction']), f"{phase}/{self.hparams.reconstruction_mode}/slices")
                    self.log_plot(visualize_projections(reconstruction_results['reconstruction']), f"{phase}/{self.hparams.reconstruction_mode}/projections")
                    self.log_plot(visualize_fsc(
                        {"fsc": reconstruction_results['fsc']}, 
                        reconstruction_results['resolution'], 
                        self.reconstructor.extractor.ctf.angpix), f"{phase}/{self.hparams.reconstruction_mode}/RelativeFourierShellCorrelation")
                if 'postprocess_results' in reconstruction_results.keys():
                    postprocess_results = reconstruction_results['postprocess_results']
                else:
                    if self.hparams.eval_half == 1:
                        eval_half = reconstruction_results['half_1']
                    elif self.hparams.eval_half == 2:
                        eval_half = reconstruction_results['half_2']
                    else:
                        eval_half = reconstruction_results['reconstruction']
                    mean_map = (reconstruction_results['half_1'] + reconstruction_results['half_2']) / 2
                    del reconstruction_results
                    postprocess_results = postprocess(eval_half, 
                                                      self.original_reconstruction, 
                                                      self.reconstructor.extractor.ctf.angpix,
                                                      self.reconstructor.hparams.autob_lowres, 
                                                      mask=self.reconstructor.mask,
                                                      mean_map=mean_map,
                                                      symmetry=self.reconstructor.back_projector[0].symmetry)
                    del mean_map, eval_half


                evaluation_metrics_outputs[f"{self.hparams.reconstruction_mode}/resolution"] = postprocess_results['sharp_resolution']
                evaluation_metrics_outputs[f"{self.hparams.reconstruction_mode}/b_factor"] = postprocess_results['b_factor']
                evaluation_metrics_outputs[f"{self.hparams.reconstruction_mode}/intercept"] = postprocess_results['intercept']
                self.reconstructor.reset_data()

                if self.logging_condition:
                    self.log_plot(visualize_slices(postprocess_results['sharp_map']), f"{phase}/{self.hparams.reconstruction_mode}/slices_postprocess")
                    self.log_plot(visualize_projections(postprocess_results['sharp_map']), f"{phase}/{self.hparams.reconstruction_mode}/projections_postprocess")
                    self.log_plot(visualize_slices(postprocess_results['filtered_map']), f"{phase}/{self.hparams.reconstruction_mode}/slices_filtered")
                    self.log_plot(visualize_projections(postprocess_results['filtered_map']), f"{phase}/{self.hparams.reconstruction_mode}/projections_filtered")

                    self.log_plot(visualize_fsc(
                        postprocess_results['fsc_postprocess'], 
                        postprocess_results['sharp_resolution'], 
                        self.reconstructor.extractor.ctf.angpix), f"{phase}/{self.hparams.reconstruction_mode}/FourierShellCorrelation")
                    
                    self.log_plot(visualize_guiner(postprocess_results['guiner'], postprocess_results['b_factor']), f"{phase}/{self.hparams.reconstruction_mode}/guiner_plot")
                    self.log_plot(visualize_histogram(postprocess_results['resolution_histogram'], 
                                                    self.reconstructor.extractor.ctf.angpix, 
                                                    self.original_reconstruction.shape[0]), f"{phase}/{self.hparams.reconstruction_mode}/resolution_histogram")
                    
                    self.reconstructor.save_reconstruction(postprocess_results['sharp_map'], f"reconstruction_sharp.mrc")
                    self.reconstructor.save_reconstruction(postprocess_results['local_resolution'], f"local_resolution.mrc")
                    self.reconstructor.save_reconstruction(postprocess_results['filtered_map'], f"filtered_reconstruction.mrc")
                del postprocess_results

            self._log_metrics_directly(phase, evaluation_metrics_outputs)
            self._set_eval_outputs()

    def on_validation_epoch_end(self) -> None:
        self._on_evaluation_epoch_end(phase='val')

    def on_test_epoch_end(self) -> None:
        self._on_evaluation_epoch_end(phase='test')

    def denoise(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Method for denoising.
        """
        return self.model(x.to(self.device))

def normalize_for_metrics(tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a tensor's values to the range [0, 1].

    Parameters:
    tensor (torch.Tensor): Input tensor to normalize.

    Returns:
    torch.Tensor: Normalized tensor with values in [0, 1].
    """
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor
