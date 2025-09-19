import mrcfile
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import v2
import h5py
import tifffile as tiff

# from source.utils import RankedLogger

# logger = RankedLogger(__name__)


class MultiNoise2NoiseDataset(Dataset):

    def __init__(
        self,
        data: pd.DataFrame,
        average_strategy: str,
        splits: int,
        reverse: bool,
        transform: v2.Compose = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32)]),
        do_fom_weighting: bool = False,
        training: bool = False,
        h5_path: str = None,
        h5_path_clean: str = None,
    ) -> None:
        self.data = data
        self.average_strategy = average_strategy
        self.splits = splits
        self.reverse = reverse
        self.transform = transform
        self.do_fom_weighting = do_fom_weighting
        self.training = training
        self.h5_path = h5_path
        self.h5_path_clean = h5_path_clean
        self.h5_data = None
        self.h5_data_clean = None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx, split: int = -1):
        row = self.data.iloc[int(idx)].to_dict()
        particle_index = row.get('particle_index', None)
        results = {}
        results['input'] = self.load_image(row['noisy_path'], particle_index)
        if 'clean_path' in self.data.columns and not self.training:
            results['clean'] = self.load_image(row['clean_path'], particle_index, clean=True)
            results['input'] = np.concatenate([results['input'], results['clean']], axis=0)
        results['input'] = np.moveaxis(results['input'], 0, -1)
        results['input'] = self.transform(results['input'])

        if 'clean_path' in self.data.columns and not self.training:
            results['clean'] = results['input'][-1][None, None, ...]
            results['input'] = results['input'][:-1]

        results['input'], results['target'], full_dose = self._average_micrograph(results['input'][:, None, ...], split=split)
        if results['target'] is None and 'clean_path' in self.data.columns:
            results['target'] = results['clean'].repeat(results['input'].shape[0], 1, 1, 1)

        if 'projection_path' in self.data.columns:
            results['out_path'] = row['projection_path']
            if 'particle_index' in self.data.columns:
                results['particle_index'] = row['particle_index']

        results['ctf_params'] = {
            'deltaf_u': row['rlnDefocusU'],
            'deltaf_v': row['rlnDefocusV'],
            'azimuthal_angle': row['rlnDefocusAngle'],
            'b_fac': row['rlnCtfBfactor'],
            'phase_shift': row['rlnPhaseShift'],
        }

        if not self.training:
            results['angles_matrix'] = euler_angles_to_matrix(rot=row['rlnAngleRot'], 
                                                              tilt=row['rlnAngleTilt'], 
                                                              psi=row['rlnAnglePsi'])
            results['shifts'] = {
                'x': row['rlnOriginXAngst'],
                'y': row['rlnOriginYAngst']
            }
            results['fom'] = torch.tensor([[[row['rlnMaxValueProbDistribution']]]], dtype=torch.double) if self.do_fom_weighting else torch.ones((1,1,1), dtype=torch.double)
            results['full_dose'] = full_dose
            #logger.info(f"input: {results['input'].shape}, target: {results['target'].shape}, full_dose: {results['full_dose'].shape}")

        results['file'] = row['rlnImageName']
        return results
    
    def load_image(self, path: str, particle_index: int, clean: bool = False) -> torch.Tensor:
        img = None
        if '.mrc' in path:
            img = mrcfile.open(path, permissive=True, mode='r').data.copy()
        elif '.h5' in path and 'particle_index' in self.data.columns:
            if clean:
                if self.h5_path_clean is not None and self.h5_data_clean is None:
                    self.h5_data_clean = h5py.File(self.h5_path_clean, 'r', libver='latest', swmr=True)
                img = self.h5_data_clean['data'][particle_index]
            else:
                if self.h5_path is not None and self.h5_data is None:
                    self.h5_data = h5py.File(self.h5_path, 'r', libver='latest', swmr=True)
                img = self.h5_data['data'][particle_index]
        elif '.npy' in path and 'particle_index' in self.data.columns:
            img = np.memmap(path, dtype='float32', mode='r')[particle_index]
        elif '.tiff' in path and 'particle_index' in self.data.columns:
            img = tiff.memmap(path, dtype='float32', mode='r')[particle_index]
        else:
            raise ValueError(f"Unsupported file format for noisy_path: {path}")
        return img

    def _average_micrograph(self, x: torch.Tensor, split: int = -1) -> torch.Tensor:
        with torch.no_grad():
            full_dose = x.sum(dim=0, keepdim=True)
            if self.splits == 1:
                x_split = full_dose
                x_rest = None
            elif split == -1:
                x_split = x.view(-1, self.splits, *x.shape[1:]).sum(dim=0)
                x_rest = full_dose - x_split
            else:
                x_split = x[split::self.splits, ...].sum(dim=0, keepdim=True)
                x_rest = full_dose - x_split
        
            if self.average_strategy == 'mean':
                x_split /= (x.shape[0] // self.splits)
                if x_rest is not None:
                    x_rest /= (x.shape[0] - x.shape[0] // self.splits)
                full_dose /= x.shape[0]

        if self.reverse:
            return x_rest, x_split, full_dose
        return x_split, x_rest, full_dose
    
    def __del__(self):
        if self.h5_data is not None:
            self.h5_data.close()
            del self.h5_data
        try:
            super().__del__()
        except AttributeError:
            pass

def euler_angles_to_matrix(rot: float, tilt: float, psi: float, homogeneous: bool = False) -> torch.Tensor:
    """
    Conver set of euler angles into rotation matrix.
    Args:
        rot: float - rot angle,
        tilt: float - tilt angle,
        psi: float - psi angle,
        homogeneous: bool - flag for setting output matrix to be homogenous,
    Returns:
        Rotation matrix.
    """
    alpha = torch.deg2rad(torch.tensor(rot, dtype=torch.double))
    beta = torch.deg2rad(torch.tensor(tilt, dtype=torch.double))
    gamma = torch.deg2rad(torch.tensor(psi, dtype=torch.double))

    ca = torch.cos(alpha)
    cb = torch.cos(beta)
    cg = torch.cos(gamma)
    sa = torch.sin(alpha)
    sb = torch.sin(beta)
    sg = torch.sin(gamma)
    cc = cb * ca
    cs = cb * sa
    sc = sb * ca
    ss = sb * sa

    if homogeneous:
        A = torch.zeros((4, 4), dtype=torch.double)
        A[3, 3] = 1
    else:
        A = torch.zeros((3, 3), dtype=torch.double)

    A[0, 0] = cg * cc - sg * sa
    A[0, 1] = cg * cs + sg * ca
    A[0, 2] = -cg * sb

    A[1, 0] = -sg * cc - cg * sa
    A[1, 1] = -sg * cs + cg * ca
    A[1, 2] = sg * sb

    A[2, 0] = sc
    A[2, 1] = ss
    A[2, 2] = cb

    return A
    