from typing import Optional
import os

import hydra
import starfile
import numpy as np
import lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from lightning.pytorch.utilities.combined_loader import CombinedLoader

from source.utils import RankedLogger


logger = RankedLogger(__name__)


class MultiNoise2NoiseDatamodule(pl.LightningDataModule):
    """
    Class of datamodule for MultiNoise2Noise.
    Hparams:
        particles_star_path: Optional[str] - path to star file with extracted particles,
        recalculate_splits: bool  - create new train/test split,
        train_perc: float - percent of train size
        batch_size: int - batch size xD,
        num_workers: int,
    """

    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        super(MultiNoise2NoiseDatamodule, self).__init__()
        self.save_hyperparameters()
        logger.info(f'Created datamodule with hparams: {self.hparams}')
        self.star = starfile.read(self.hparams.particles_star_path)
        if self.hparams.create_projections:
            self.star['particles']['rlnProjectionImage'] = self.star['particles']['rlnNoisyImage'].apply(lambda x: os.path.join(*(['Projections'] + x.split('/')[1:])))
        
        self.particles = self.star['particles'].copy()
        self.particles['noisy_path'] = self.particles['rlnNoisyImage'].apply(lambda x: os.path.join(self.hparams.dataset_path, x))
        if 'rlnCleanImage' in self.particles.keys():
            self.particles['clean_path'] = self.particles['rlnCleanImage'].apply(lambda x: os.path.join(self.hparams.dataset_path, x))
        if self.hparams.create_projections:
            self.particles["projection_path"] = self.particles["rlnProjectionImage"].apply(lambda x: os.path.join(self.hparams.dataset_path, x))
        self.particles['rlnOriginXAngst'] /= self.hparams.angpix
        self.particles['rlnOriginYAngst'] /= self.hparams.angpix
        if 'split' not in self.star['particles'].columns or self.hparams.recalculate_splits:
            self._create_splits()

        # Instantiate dose_weighter once to share the same instance across all datasets
        self.dose_weighter = hydra.utils.instantiate(self.hparams.dose_weighter)
        self.current_epoch = 0
        self.dose_weights = None
    
    def _create_splits(self):
        """Creates the train/val split movie-wise

        Args:
            files_df (pd.DataFrame): Dataframe with the files (movies).
        """
        val_perc = (1 - self.hparams.train_perc)
        logger.info(f'Creating training split: {self.hparams.train_perc * 100}% of the whole dataset')

        val_size = int(len(self.star['particles']) * val_perc)
        train_size = len(self.star['particles']) - val_size

        splits = ['train'] * train_size + ['val_test'] * val_size
        np.random.shuffle(splits)
        self.star['particles']['split'] = splits
        self.particles['split'] = splits

        count = len(self.star['particles'].index)
        logger.info(f'Created splits. Original dataframe had {count} pairs. Train dataframe: {train_size} pairs. Val/test: {val_size} pairs (For validation and testing also train data is being used)')
        starfile.write(self.star, self.hparams.particles_star_path)
        logger.info(f'Updated file {self.hparams.particles_star_path}.')

    def prepare_data(self, stage: Optional[str] = None) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        self.particles['rlnImageName'] = (self.particles.groupby('rlnMicrographName').cumcount() + 1).astype(str).str.zfill(6) + \
            '@' + self.particles['rlnMicrographName'].apply(lambda x: x.split('/')[-1].split('.')[0] + '.mrcs')

        self.particles = self.particles.iloc[::-1].reset_index(drop=True)

        if self.hparams.weight_doses:
            dose_weights = self.dose_weighter.create_weights()
            # dose_weights = (dose_weights.sum(0, keepdim=True) - dose_weights) / (dose_weights.shape[0] - 1)
            full_dose = dose_weights.sum(dim=0, keepdim=True)
            if self.hparams.datasets.train.splits == 1:
                x_split = full_dose
                x_rest = None
            else:
                x_split = dose_weights.view(-1, self.hparams.datasets.train.splits, *dose_weights.shape[1:]).sum(dim=0)
                x_rest = full_dose - x_split

        
            if self.hparams.datasets.train.average_strategy == 'mean':
                x_split /= (dose_weights.shape[0] // self.hparams.datasets.train.splits)
                if x_rest is not None:
                    x_rest /= (dose_weights.shape[0] - dose_weights.shape[0] // self.hparams.datasets.train.splits)
                full_dose /= dose_weights.shape[0]

            if self.hparams.datasets.train.reverse:
                x_tmp = x_split.clone()
                x_split = x_rest
                x_rest = x_tmp
            self.dose_weights = torch.stack([full_dose / x_split, full_dose / x_rest], dim=1).float() if x_rest is not None else torch.stack([full_dose / x_split], dim=1).float()
        
        if self.hparams.train_half == 1:
            self.dataset_train = hydra.utils.call(
                self.hparams.datasets.train,
                data=self.particles[(self.particles.split == 'train') & (self.particles['rlnRandomSubset']==1)],
                dose_weights=self.dose_weights,
                _recursive_=True,
            )
            self.train_loader = CombinedLoader([DataLoader(
                dataset=self.dataset_train,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                shuffle=True, 
                persistent_workers=True,
                pin_memory=True,
            )])
        elif self.hparams.train_half == 2:
            self.dataset_train = hydra.utils.call(
                self.hparams.datasets.train,
                data=self.particles[(self.particles.split == 'train') & (self.particles['rlnRandomSubset']==2)],
                dose_weights=self.dose_weights,
                _recursive_=True,
            )
            self.train_loader = CombinedLoader([DataLoader(
                dataset=self.dataset_train,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                shuffle=True, 
                persistent_workers=True,
                pin_memory=True,
            )])
        else:
            self.dataset_train = [
                hydra.utils.call(
                    self.hparams.datasets.train,
                    data=self.particles[(self.particles.split == 'train') & (self.particles['rlnRandomSubset']==idx)],
                    dose_weights=self.dose_weights,
                    _recursive_=True,
                ) for idx in range(1,3)]
            self.train_loader = CombinedLoader([DataLoader(
                dataset=dataset,
                batch_size=self.hparams.batch_size // 2,
                num_workers=self.hparams.num_workers // 2,
                shuffle=True,
                persistent_workers=True,
                pin_memory=True,
            ) for dataset in self.dataset_train])
        
        # self.dataset_train = hydra.utils.call(
        #     self.hparams.datasets.train,
        #     data=self.particles[self.particles.split == 'train'],
        #     _recursive_=True,
        # )

        self.dataset_val = [
            hydra.utils.call(
                self.hparams.datasets.val,
                data=self.particles[self.particles['rlnRandomSubset']==idx],
                dose_weights=self.dose_weights,
                _recursive_=True,
            ) for idx in range(1,3)]

        self.val_loaders = [DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False, 
            persistent_workers=True,
            pin_memory=True,
        ) for dataset in self.dataset_val]

    def ctf_mean(self, ctf) -> torch.Tensor:
        ctfs = self.particles[self.particles.split == 'train'][['rlnDefocusU', 'rlnDefocusV', 'rlnDefocusAngle', 'rlnCtfBfactor', 'rlnPhaseShift', 'rlnNoisyImage']].groupby(
            ['rlnDefocusU', 'rlnDefocusV', 'rlnDefocusAngle', 'rlnCtfBfactor', 'rlnPhaseShift']).count().reset_index()
        ctfs = ctfs.rename(columns={'rlnNoisyImage': 'count'})
        ctf_weights = ctf.get_fftw_image(phase_shift=torch.tensor(ctfs['rlnPhaseShift'], dtype=torch.double), 
                                        azimuthal_angle=torch.tensor(ctfs['rlnDefocusAngle'], dtype=torch.double),
                                        b_fac=torch.tensor(ctfs['rlnCtfBfactor'], dtype=torch.double), 
                                        deltaf_u=torch.tensor(ctfs['rlnDefocusU'], dtype=torch.double), 
                                        deltaf_v=torch.tensor(ctfs['rlnDefocusV'], dtype=torch.double)) * torch.tensor(ctfs['count'], dtype=torch.double)[:, None, None, None]
        ctf_weights_abs = (ctf_weights.abs().sum(0, keepdim=True) / len(self.particles[self.particles.split == 'train'].index)).float()
        return ctf_weights_abs[None, ...]

    
    def train_dataloader(self) -> CombinedLoader:
        return self.train_loader

    def val_dataloader(self) -> CombinedLoader:
        return CombinedLoader(self.val_loaders, mode='sequential')

    def test_dataloader(self) -> CombinedLoader:
        return CombinedLoader(self.val_loaders, mode='sequential')

    def predict_dataloader(self) -> CombinedLoader:
        dataset = hydra.utils.call(
            self.hparams.datasets.val,
            data=self.particles,
            dose_weights=self.dose_weights,
            _recursive_=True,
        )

        predict_loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False, 
            persistent_workers=True,
            pin_memory=True,
        )

        out_star = self.star.copy()
        out_star['particles'] = self.particles
        starfile.write(out_star, os.path.join(os.path.dirname(self.hparams.particles_star_path), 'denoised.star'))
        return predict_loader #CombinedLoader([predict_loader], mode='sequential')


class EmptyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 0
    def __getitem__(self, idx):
        raise IndexError("EmptyDataset has no items")
