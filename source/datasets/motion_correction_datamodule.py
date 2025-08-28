from typing import Optional
import os

import hydra
import starfile
import numpy as np
import lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from source.utils import RankedLogger


logger = RankedLogger(__name__)


class MotionCorrectionDatamodule(pl.LightningDataModule):
    """
    Class of datamodule for MultiNoise2Noise.
    Hparams:
        micrographs_star_path: Optional[str] - path to star file with imported micrographs,
        dataset_path: str - Base directory of dataset,
        out_dir: str - path to output directory,
        old_size: int - size of extraction box (original box size),
        dose_weighter: dict - configuration of dose weighter,
        batch_size: int - batch size xD,
        num_workers: int,
    """

    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        super(MotionCorrectionDatamodule, self).__init__()
        self.save_hyperparameters()
        logger.info(f'Created datamodule with hparams: {self.hparams}')
        self.star = starfile.read(self.hparams.micrographs_star_path)
        self.star['movies']['rlnMicrographName'] = self.star['movies']['rlnMicrographMovieName'].map(self.create_out_path)
        os.makedirs(self.hparams.out_dir, exist_ok=True)
        starfile.write(self.star, os.path.join(self.hparams.out_dir, f"{self.hparams.dataset_name}.star"))

        self.micrographs = self.star['movies'].copy()
        self.micrographs['rlnMicrographName'] = self.micrographs['rlnMicrographName'].apply(lambda x: os.path.join(self.hparams.out_dir, x))
        self.micrographs['rlnMicrographMovieName'] = self.micrographs['rlnMicrographMovieName'].apply(lambda x: os.path.join(self.hparams.dataset_path, x))
        if not self.hparams.do_all:
            self.remove_done()
        self.prepare_data()

    def remove_done(self) -> None:
        not_processed = self.micrographs['rlnMicrographName'].apply(lambda path: not os.path.exists(path))
        self.micrographs = self.micrographs[not_processed]
        print(f"correcting {len(self.micrographs.index)} movies")

    def prepare_data(self, stage: Optional[str] = None) -> None:
        self.dataset = hydra.utils.instantiate(config=self.hparams.dataset, df=self.micrographs)

    def create_out_path(self, path: str) -> str:
        modified_string = os.path.join(self.hparams.prefix, path)
        dot_count = modified_string.count('.')
        if dot_count <= 1:
            modified_string = os.path.splitext(path)[0] + '.mrc'
            return modified_string
        split_parts = modified_string.rsplit('.', 1)
        modified_string = split_parts[0].replace('.', '_') + '.mrc'# + split_parts[1]
        return modified_string

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def test_dataloader(self) -> list[DataLoader]:
        return self.predict_dataloader()
    
    def predict_dataloader(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False, 
            pin_memory=True
        )
