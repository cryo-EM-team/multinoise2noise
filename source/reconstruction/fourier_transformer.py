import torch


class FourierTransformer:
    def sign_flip(self, f_image: torch.Tensor):
        f_image.view(f_image.shape[0], -1)[:, 1::2] *= -1

    def shift_image_in_fourier_transform(self, in_array: torch.Tensor,
                                        xshift: torch.Tensor,
                                        yshift: torch.Tensor) -> torch.Tensor:
        """
        Shift a batch of Fourier-transformed images in the Fourier domain by given x and y shifts.
        
        Args:
            in_array (torch.Tensor): Complex Fourier-transformed images of shape (B, C, YSIZE, XSIZE).
            xshift (torch.Tensor): x-shifts for each image (in pixels), shape (B,).
            yshift (torch.Tensor): y-shifts for each image (in pixels), shape (B,).
        
        Returns:
            torch.Tensor: Shifted Fourier-transformed images, same shape as in_array.
        """
        batch_size, _, YSIZE, XSIZE_half = in_array.shape
        XSIZE = 2 * (XSIZE_half - 1) if XSIZE_half > 1 else 1

        # Create frequency grids along the y and x dimensions.
        freq_y = torch.fft.fftfreq(YSIZE, device=in_array.device).view(1, 1, YSIZE, 1)
        freq_x = torch.fft.rfftfreq(XSIZE, device=in_array.device).view(1, 1, 1, XSIZE_half)
        
        # Reshape shifts so they broadcast correctly over the frequency grids.
        xshift = xshift.view(batch_size, 1, 1, 1)
        yshift = yshift.view(batch_size, 1, 1, 1)
        
        # Compute the phase shift factor.
        # The phase shift for each frequency is given by: exp(2j*pi*(yshift*freq_y + xshift*freq_x))
        phase = -2j * torch.pi * (yshift * freq_y + xshift * freq_x)
        phase_shift = torch.exp(phase)
        
        # Apply the phase shift to the Fourier-transformed image.
        shifted_F = in_array * phase_shift
        shifted_F[..., 0, 0] = 0.
        return shifted_F

    def fourier_preprocessing(self, batch: torch.Tensor, xshift: torch.Tensor, yshift: torch.Tensor) -> torch.Tensor:
        f_image = torch.fft.rfftn(batch, dim=(-2, -1), norm="forward")
        self.sign_flip(f_image)
        f_shifted = self.shift_image_in_fourier_transform(f_image, xshift.double(), yshift.double())
        return f_shifted

    def reverse_fourier_preprocessing(self, batch: torch.Tensor, xshift: torch.Tensor, yshift: torch.Tensor) -> torch.Tensor:
        f_image = self.shift_image_in_fourier_transform(batch, -xshift.double(), -yshift.double())
        self.sign_flip(f_image)
        image = torch.fft.irfftn(f_image, dim=(-2, -1), norm="forward")
        return image
