import mrcfile
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torchvision.transforms import v2
import tifffile as tiff

class MotionCorrectionDataset(Dataset):
    def __init__(self, 
                 df: pd.DataFrame, 
                 gain_path: str,
                 transform: v2.Compose = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float64)])):
        self.df = df
        self.gain_path = gain_path
        self.transform = transform
        with mrcfile.mmap(self.gain_path, permissive=True, mode='r+')as mrc:
            self.gain = self.transform(mrc.data.copy())
        self.bad_pixels_coords = torch.nonzero(self.gain[0] == 0, as_tuple=False)

    def __len__(self) -> int:
        return len(self.df.index)

    def __getitem__(self, idx) -> tuple[torch.Tensor, str]:
        row = self.df.iloc[int(idx)].to_dict()
        img_path = row['rlnMicrographMovieName']
        if ".mrc" in img_path:
            with mrcfile.open(img_path, permissive=True, mode='r') as mrc:
                image = np.moveaxis(mrc.data.copy(), 0, -1)
        elif ".tif" in img_path:
            image = np.moveaxis(np.flip(tiff.imread(img_path, mode='r'), 1).copy(), 0, -1)

        image = self.transform(image)
        image *= self.gain
        out_path = row['rlnMicrographName']
        return {"images": image, "out_paths": out_path}
