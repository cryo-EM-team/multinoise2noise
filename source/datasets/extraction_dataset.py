import mrcfile
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from torchvision.transforms import v2


class ExtractionDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 box: int,
                 transform: v2.Compose = v2.Compose([v2.ToTensor(), v2.ToDtype(torch.float64)])):
        self.df = df
        self.box = box
        self.half_box = self.box // 2
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df.index)

    def __getitem__(self, idx) -> tuple[torch.Tensor, str]:
        row = self.df.iloc[int(idx)].to_dict()
        img_path = row['rlnMicrographName']
        x = int(row['rlnCoordinateX'])
        y = int(row['rlnCoordinateY'])
        image = self.load_image_patch(img_path, x, y)
        image = np.moveaxis(image, 0, -1)
        image = self.transform(image)
        out_path = row['rlnImageName']

        ctf_params = {
            'deltaf_u': row['rlnDefocusU'],
            'deltaf_v': row['rlnDefocusV'],
            'azimuthal_angle': row['rlnDefocusAngle'],
            'b_fac': row['rlnCtfBfactor'],
            'phase_shift': row['rlnPhaseShift'],
        }

        output = {"images": image, "out_paths": out_path, "ctf_params": ctf_params}
        if 'particle_index' in row.keys():
            output["particle_index"] = row['particle_index']
        return output

    def load_image_patch(self, path: str, x: int, y: int) -> np.ndarray:
        with mrcfile.mmap(path, permissive=True, mode='r') as mrc:
            nx = mrc.header.nx.item()
            ny = mrc.header.ny.item()

            x_min_ = x - self.half_box
            x_max_ = x + self.half_box
            y_min_ = y - self.half_box
            y_max_ = y + self.half_box
            x_min = max(x_min_, 0)
            x_max = min(x_max_, nx-1)
            y_min = max(y_min_, 0)
            y_max = min(y_max_, ny-1)

            image = mrc.data[...,
                             y_min:y_max,
                             x_min:x_max].copy().astype(np.float64)
            
            if len(image.shape) < 3:
                image = np.expand_dims(image, axis=0)
            image = np.pad(image, ((0,0),(y_min - y_min_, y_max_ - y_max),(x_min - x_min_, x_max_ - x_max)), mode='reflect')
        return image
