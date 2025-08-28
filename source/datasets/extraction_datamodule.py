from functools import partial
from typing import Optional
import os
import hydra
import starfile
import lightning as pl
from torch.utils.data import DataLoader

from source.utils import RankedLogger


logger = RankedLogger(__name__)


class ExtractionDatamodule(pl.LightningDataModule):
    """
    Class of datamodule for MultiNoise2Noise.
    Hparams:
        particles_star_path: Optional[str] - path to star file with extracted particles,
        dataset_path: str - Base directory of dataset,
        out_dir: str - path to output directory,
        old_size: int - size of extraction box (original box size),
        dose_weighter: dict - configuration of dose weighter,
        batch_size: int - batch size,
        num_workers: int - number of workers,
        file_mode: str - should data be saved as mrcs files or h5py files,
    """

    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        super(ExtractionDatamodule, self).__init__()
        self.save_hyperparameters()
        logger.info(f'Created datamodule with hparams: {self.hparams}')
        self.star = starfile.read(self.hparams.particles_star_path)
        target_name = 'rlnImageName'
        if 'rlnNoisyImage' in self.star['particles'].columns:
            target_name = 'rlnNoisyImage'
        # elif 'rlnMicrographName' in self.star['particles'].columns:
        #     target_name = 'rlnMicrographName'

        if self.hparams.file_mode == 'mrcs':
            self.star['particles'][target_name] = self.star['particles'][target_name].map(self.create_out_path_mrcs)
        elif self.hparams.file_mode == 'h5':
            self.star['particles']['particle_index'] = self.star['particles'].index
            self.h5_path = os.path.join(self.hparams.out_dir, f"{self.hparams.dataset_name}.h5")
            self.star['particles'][target_name] = f"{self.hparams.dataset_name}.h5"
        else:
            self.star['particles']['particle_index'] = self.star['particles'][target_name].map(self.create_out_idx)
            self.star['particles'][target_name] = self.star['particles'][target_name].map(partial(self.create_out_path, extension=self.hparams.file_mode))
        os.makedirs(self.hparams.out_dir, exist_ok=True)
        starfile.write(self.star, os.path.join(self.hparams.out_dir, f"{self.hparams.dataset_name}.star"))

        #self.angpix = self.star['optics']['rlnImagePixelSize'][0]
        self.particles = self.star['particles'].copy()
        self.particles['rlnImageName'] = self.particles[target_name].apply(lambda x: os.path.join(self.hparams.out_dir, x))
        self.particles['rlnMicrographName'] = self.particles['rlnMicrographName'].apply(lambda x: os.path.join(self.hparams.dataset_path, x))
        self.particles = self.particles.iloc[::-1].reset_index(drop=True)
        if self.hparams.file_mode != 'h5':
            self.remove_done()

    def remove_done(self) -> None:
        not_extracted = self.particles['rlnImageName'].apply(lambda path: not os.path.exists(path))
        self.particles = self.particles[not_extracted]
        print(f"extracting {len(self.particles.index)} particles")

    def prepare_data(self, stage: Optional[str] = None) -> None:
        pass

    def create_out_path_mrcs(self, path: str) -> str:
        if '@' in path:
            num, out_path = path.split('@')
            out_path = out_path[out_path.find('/')+1:].replace(".", f"/{int(num)}.")
        else:
            out_path = path[path.find('/')+1:]
        out_path = os.path.join(self.hparams.out_dir, out_path)
        return out_path

    def create_out_idx(self, path: str) -> str:
        num = int(path.split('@')[0]) - 1
        return num

    def create_out_path(self, path: str, extension: str = 'h5') -> str:
        if '@' in path:
            _, out_path = path.split('@')
        out_path = os.path.splitext(out_path[out_path.find('/')+1:])[0] + f'.{extension}'
        out_path = os.path.join(self.hparams.out_dir, out_path)
        return out_path

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = hydra.utils.instantiate(df=self.particles, config=self.hparams.dataset)
        
    def test_dataloader(self) -> list[DataLoader]:
        return self.predict_dataloader()
    
    def predict_dataloader(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False
        )
