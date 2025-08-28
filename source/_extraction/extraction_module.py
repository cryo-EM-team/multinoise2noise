from typing import Any
import torch
import lightning as pl
import mrcfile
import os
import h5py
import numpy as np
import tifffile

from source.reconstruction.extractor import Extractor


class ExtractionModule(pl.LightningModule):
    """
    Class for extraction of particles from micrographs and their fouriercropping.
    Hparams:
        extractor: Extractor - Object responsible for contrast transfre function correction
    """
    def __init__(self, extractor: Extractor, *args, **kwargs):
        super(ExtractionModule, self).__init__()
        self.save_hyperparameters(logger=False, ignore=['extractor'])
        self.extractor = extractor

    def exchange_data(self, **kwargs):
        if 'h5' == self.hparams.file_mode:
            self.h5_path = kwargs['datamodule'].h5_path
            self.chunk_shape = (kwargs['datamodule'].hparams.num_frames, self.extractor.new_size, self.extractor.new_size)
            self.num_chunks = len(kwargs['datamodule'].star['particles'].index)
            if os.path.exists(self.h5_path):
                os.remove(self.h5_path)
            self._create_h5_file()

    def _create_h5_file(self):
        """Create the HDF5 file with preallocated dataset."""
        depth, height, width = self.chunk_shape
        with h5py.File(self.h5_path, 'w', libver='latest') as h5f:
            dataset_shape = (self.num_chunks, depth, height, width)
            chunk_layout = (1, depth, height, width)  # Chunk per data sample

            dset = h5f.create_dataset(
                'data',
                shape=dataset_shape,
                dtype='float32',
                chunks=chunk_layout,
                compression=None
            )
            dset.attrs["voxel_size"] = self.extractor.ctf.angpix
            dset.attrs["mx"] = width
            dset.attrs["my"] = height
            dset.attrs["mz"] = depth


    def forward(self, x: torch.Tensor, ctf_params: dict[str, Any]) -> torch.Tensor:
        return self.extractor.rescale_patch(x, ctf_params)
    
    def test_step(
        self, batch: dict[str, torch.Tensor], *args: Any, **kwargs: Any
    ) -> dict[str, list[torch.Tensor]]:
        self.predict_step(batch=batch, args=args, kwargs=kwargs)

    def predict_step(
        self, batch: dict[str, torch.Tensor], *args: Any, **kwargs: Any
    ) -> dict[str, list[torch.Tensor]]:
        extracted = self.forward(batch['images'], batch['ctf_params'])
        if self.hparams.file_mode == 'mrcs':
            self.save_mrcs(extracted, batch['out_paths'])
        elif self.hparams.file_mode == 'h5':
            self.save_h5(extracted, batch['particle_index'])
        elif self.hparams.file_mode == 'npy':
            self.save_memmap(extracted, batch['out_paths'], batch['particle_index'])
        elif self.hparams.file_mode == 'tiff':
            self.save_tiff(extracted, batch['out_paths'], batch['particle_index'])

    def save_mrcs(self, images: torch.Tensor, out_paths: list[str]):
        """
            Save reconstructed data
        Args:
            images: Batch of images
            out_paths: paths under which results will be saved

        Returns:
            Computed predictions
        """
        for image, out_path in zip(images, out_paths):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with mrcfile.new(out_path, overwrite=True) as mrc:
                mrc.set_data(image.cpu().float().numpy())
                mrc.voxel_size = self.extractor.ctf.angpix
                mrc.header.mx = image.size(-1)
                mrc.header.my = image.size(-2)
                mrc.header.mz = image.size(-3)
                mrc.update_header_stats()

    def save_h5(self, images: torch.Tensor, channel_indices: list[str]):
        with h5py.File(self.h5_path, 'a') as hf:
            for idx, image in zip(channel_indices, images):
                hf["data"][idx, ...] = image.detach().cpu().float().numpy()

    def save_tiff(self, images: torch.Tensor, out_paths: list[str], channel_indices: list[str]):
        """
        Save data as memory-mapped TIFF files using tifffile.memmap.

        Args:
            images: Batch of images.
            out_paths: Paths under which results will be saved.
            channel_indices: Indices of the channels to save.
        """
        groups = {}
        for idx, image, tiff_path in zip(channel_indices, images, out_paths):
            groups.setdefault(tiff_path, []).append((idx, image))

        for tiff_path, items in groups.items():
            os.makedirs(os.path.dirname(tiff_path), exist_ok=True)
            # Determine the maximum incoming channel index for this file.
            max_incoming_idx = max(idx for idx, _ in items)

            if os.path.exists(tiff_path):
                # File exists: open it as a memory-mapped array.
                memmap = tifffile.memmap(tiff_path, mode='r+')
                current_channels = memmap.shape[0]
                required_channels = max(current_channels, max_incoming_idx + 1)
                # Resize the memory-mapped array if necessary.
                if required_channels > current_channels:
                    new_shape = (required_channels,) + tuple(memmap.shape[1:])
                    new_memmap = tifffile.memmap(tiff_path, shape=new_shape, dtype=memmap.dtype, mode='w+')
                    new_memmap[:current_channels] = memmap[:]
                    memmap = new_memmap
                # Assign each incoming image to its designated channel.
                for idx, image in items:
                    img_array = image.cpu().float().numpy()
                    memmap[idx, ...] = img_array
            else:
                # File does not exist: create a new memory-mapped TIFF file.
                sample = items[0][1]
                sample_shape = tuple(sample.shape[-3:])
                required_channels = (max_incoming_idx + 1).item()
                # Allocate a new memory-mapped array and fill incoming channels.
                memmap = tifffile.memmap(tiff_path, shape=(required_channels,) + sample_shape,
                                        dtype=np.float32)
                for idx, image in items:
                    img_array = image.cpu().float().numpy()
                    memmap[idx, ...] = img_array
            # Ensure changes are flushed to disk.
            memmap.flush()

    def save_memmap(self, images: torch.Tensor, out_paths: list[str], channel_indices: list[str]):
        """
        Save data as memory-mapped NumPy arrays.
        
        Args:
            images: Batch of images.
            out_paths: Paths under which results will be saved.
            channel_indices: Indices of the channels to save.
        """
        groups = {}
        for idx, image, memmap_path in zip(channel_indices, images, out_paths):
            groups.setdefault(memmap_path, []).append((idx, image))
        
        for memmap_path, items in groups.items():
            os.makedirs(os.path.dirname(memmap_path), exist_ok=True)
            # Determine the maximum incoming channel index for this file.
            max_incoming_idx = max(idx for idx, _ in items)
            
            if os.path.exists(memmap_path):
                # File exists: open it in read-write mode.
                memmap = np.load(memmap_path, mmap_mode='r+')
                current_channels = memmap.shape[0]
                # Determine required number of channels.
                required_channels = max(current_channels, max_incoming_idx + 1)
                if required_channels > current_channels:
                    # Resize the memory-mapped array.
                    new_shape = (required_channels,) + memmap.shape[1:]
                    memmap = np.memmap(memmap_path, dtype=memmap.dtype, mode='r+', shape=new_shape)
                # Assign each incoming image to its designated channel.
                for idx, image in items:
                    img_array = image.cpu().float().numpy()
                    memmap[idx, ...] = img_array
            else:
                # File does not exist: create a new memory-mapped file.
                sample = items[0][1]
                sample_shape = sample.shape[-3:]
                required_channels = max_incoming_idx + 1
                # Allocate a new memory-mapped array and fill incoming channels.
                data = np.zeros((required_channels,) + sample_shape,
                                dtype=sample.cpu().float().numpy().dtype)
                for idx, image in items:
                    data[idx, ...] = image.cpu().float().numpy()
                memmap = np.memmap(memmap_path, dtype=data.dtype, mode='w+', shape=data.shape)
                memmap[:] = data[:]
            # Ensure changes are flushed to disk.
            del memmap
