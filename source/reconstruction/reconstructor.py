import lightning as L
import torch
from torch import Tensor
import mrcfile
import os
import pandas as pd
import copy
import starfile

from source.reconstruction.postprocessing import postprocess
from source.reconstruction.utils import calculate_resolution, real_round, calculate_fsc, low_resolution_join_halves, create_fourier_distance_map, get_downsampled_average
from source.reconstruction.back_projector import BackProjector
from source.reconstruction.extractor import Extractor
from source.reconstruction.fourier_transformer import FourierTransformer


class Reconstructor(L.LightningModule):
    """Module performing process of reconstruction. It is composed of all reconstruction steps.
    """

    def __init__(self, back_projector: BackProjector, fourier_transformer: FourierTransformer, extractor: Extractor, **kwargs):
        """
        Args:
            back_projector: Object responsible for back-projecting 2D images onto 3D space
            fourier_transformer: Object responsible for transforming 2D images into fourier space
            extractor: Object responsible for computation of ctf weights and normalisation of image
            **kwargs
        """
        super().__init__()
        self.save_hyperparameters(ignore=['extractor', 'back_projector', 'fourier_transformer'])
        self.back_projector = torch.nn.ModuleList([back_projector, copy.deepcopy(back_projector)])
        self.fourier_transformer = fourier_transformer
        self.extractor = extractor
        self.bckp_idx = 0
        self.mask = torch.tensor(mrcfile.open(self.hparams.mask_path, permissive=True).data, dtype=torch.double)

        if not os.path.exists(self.hparams.output_dir):
            os.makedirs(self.hparams.output_dir)

    def forward(self, image: Tensor, shifts: dict[str, torch.Tensor], angles_matrix: torch.Tensor, ctf_params: dict[str, object], fom: torch.Tensor, bckp_idx: int = None) -> None:
        """
        Transforming and back-projecting batch of images
        Args:
            image: Tensor representing picture
            ctf_params: parameters of the ctf correction
            angles_matrix: angles_matrix
            shifts: shifts
            fom: figure of merit weight
        """
        if bckp_idx is None:
            bckp_idx = self.bckp_idx
        normalised_image = self.extractor.normalise_patch(image)
        shifted_f = self.fourier_transformer.fourier_preprocessing(normalised_image, shifts['x'].double(), shifts['y'].double())
        fftw_image = self.extractor.ctf.get_fftw_image(**ctf_params)
        if self.hparams.ctf_mode == "normal":
            shifted_f *= fftw_image
        elif self.hparams.ctf_mode == "abs":
            shifted_f *= fftw_image.abs()
        fftw_image *= fftw_image
        shifted_f *= fom
        fftw_image *= fom
        self.back_projector[bckp_idx].backproject2Dto3D(shifted_f, angles_matrix.clone(), fftw_image)
        return None

    def backward(self, shifts: dict[str, torch.Tensor], angles_matrix: torch.Tensor, bckp_idx: int = None) ->  torch.Tensor:
        if bckp_idx is None:
            bckp_idx = self.bckp_idx
        shifted_f = self.back_projector[bckp_idx].project3Dto2D(angles_matrix.clone())
        image = self.fourier_transformer.reverse_fourier_preprocessing(shifted_f, shifts['x'].double(), shifts['y'].double())
        normalised_image = self.extractor.normalise_patch(image)
        return normalised_image
    
    def predict_step(self, test_batch: object, batch_idx: int) -> dict[str, torch.Tensor]:
        """
            Perform a prediction step
        Args:
            test_batch: Batch of test data
            batch_idx: index of batch

        Returns:
            Computed predictions
        """
        self.forward(**test_batch)
        return None

    def on_predict_epoch_end(self, bckp_idx: int = None) -> None:
        bckp_idx = bckp_idx if bckp_idx is not None else self.bckp_idx
        self.back_projector[bckp_idx].symmetrise()

    def full_on_predict_epoch_end(self) -> None:
        self.on_predict_epoch_end(0)
        self.on_predict_epoch_end(1)
    
    def finish_reconstruction(self) -> dict[str, torch.Tensor]:
        avg_1 = torch.fft.ifftshift(
            get_downsampled_average(self.back_projector[0].data, self.back_projector[0].weight, self.hparams.divide_avg), 
            dim=(0,1))
        avg_2 = torch.fft.ifftshift(
            get_downsampled_average(self.back_projector[1].data, self.back_projector[1].weight, self.hparams.divide_avg), 
            dim=(0,1))
        results = {}
        results['fsc'] = low_resolution_join_halves(calculate_fsc(avg_1, avg_2), self.extractor.ctf.angpix, self.hparams.low_resol_join_halves)

        weight_modifiers, metadata = self.create_ssnr_arrays(results['fsc'])
        results['resolution'] = calculate_resolution(results['fsc'], self.extractor.ctf.angpix)
        results['half_1'] = self.back_projector[0].reconstruct()
        results['half_2'] = self.back_projector[1].reconstruct()
        self.save_reconstruction(results['half_1'], "half_1_reconstruction.mrc")
        self.save_reconstruction(results['half_2'], "half_2_reconstruction.mrc")
        results['final_backprojector'] = copy.deepcopy(self.back_projector[0])
        results['final_backprojector'].weight += self.back_projector[1].weight + weight_modifiers
        results['final_backprojector'].data += self.back_projector[1].data
        results['reconstruction'] = results['final_backprojector'].reconstruct()
        
        self.save_reconstruction(results['reconstruction'], "reconstruction.mrc")

        metadata = {"general": {"resolution": results['resolution']}, "fsc": metadata}
        if self.hparams.postprocess:
            postprocess_results = postprocess(results['half_1'], results['half_2'], self.extractor.ctf.angpix, self.hparams.autob_lowres, mask=self.mask, symmetry=self.back_projector[0].symmetry)
            metadata["general"]["b_factor"] = postprocess_results['b_factor']
            metadata["general"]["sharpened_resolution"] = postprocess_results['sharp_resolution']

            self.save_reconstruction(postprocess_results['sharp_map'], "reconstruction_sharp.mrc")
            self.save_reconstruction(postprocess_results['local_resolution'], "local_resolution.mrc")
            self.save_reconstruction(postprocess_results['filtered_map'], "filtered_reconstruction.mrc")
            results['postprocess_results'] = postprocess_results

        self.save_metadata(metadata)

        for i in range(2):
            self.back_projector[i].weight_precalculated = True
        return results

    def save_reconstruction(self, reconstruction: torch.Tensor, file_name: str):
        full_path = os.path.join(self.hparams.output_dir, file_name)
        directory = os.path.dirname(full_path)
        os.makedirs(directory, exist_ok=True)

        with mrcfile.new(full_path, overwrite=True) as mrc:
            mrc.set_data(reconstruction.cpu().float().numpy())
            mrc.voxel_size = self.extractor.ctf.angpix
            mrc.header.mx = reconstruction.size(2)
            mrc.header.my = reconstruction.size(1)
            mrc.header.mz = reconstruction.size(0)
            mrc.update_header_stats()

    def save_metadata(self, metadata: dict[str, pd.DataFrame | dict[str, float]]) -> None:
        starfile.write(metadata, os.path.join(self.hparams.output_dir, "metadata.star"), overwrite=True)

    def create_ssnr_arrays(self, fsc: torch.Tensor, is_whole_instead_of_half: bool = True) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        rmax = real_round(torch.Tensor([self.back_projector[0].ori_size / 2]) * self.back_projector[0].padding_factor)
        oversampling_correction = self.back_projector[0].padding_factor**3

        weight = self.back_projector[0].weight + self.back_projector[1].weight

        r = create_fourier_distance_map(weight.shape).flatten()
        fmask = (r < rmax)
        ires = real_round(r / self.back_projector[0].padding_factor).int()
        invw = oversampling_correction * weight.flatten()
        invw_f = invw[fmask]
        ires_f = ires[fmask]

        counter = torch.zeros_like(fsc)
        counter.index_add_(0, ires_f, torch.ones_like(invw_f))

        sigma2 = torch.zeros_like(fsc)
        sigma2.index_add_(0, ires_f, invw_f)
        sigma2 = torch.nan_to_num(counter / sigma2)

        myfsc = torch.clamp(fsc, min=0.001)
        if is_whole_instead_of_half:
            myfsc = torch.sqrt(2 * myfsc / (myfsc + 1))
        myfsc = torch.clamp(myfsc, max=0.999)
        ssnr = myfsc / (1. - myfsc) * self.hparams.tau2_fudge

        tau2 = ssnr * sigma2

        invtau2 = torch.where(tau2 > 0, 1. / (oversampling_correction * self.hparams.tau2_fudge * tau2), 0.)

        weight_modifiers = torch.zeros_like(fmask, dtype=torch.double)
        weight_modifiers[fmask] = torch.index_select(invtau2, 0, ires_f)
        weight_modifiers = weight_modifiers.unflatten(0, weight.shape)
        return weight_modifiers, pd.DataFrame({"ssnr": ssnr, "sigma2": sigma2, "tau2": tau2, "fsc": fsc})

    def reset_data(self) -> None:
        """Reset data in back projectors"""
        for back_projector in self.back_projector:
            back_projector.data.zero_()
