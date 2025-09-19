import torch


class CTF(torch.nn.Module):
    """
    Contrast Transfer Function (CTF) module for electron microscopy image simulation.

    This class computes the CTF for a given set of microscope and imaging parameters,
    including support for damping and phase shift effects.
    """
    def __init__(self, cs: float, kv: float, q0: float, angpix: float, size: int, do_damping: bool = True,
                 gamma_offset: float = 0.0):
        """
        Initialize the CTF module with microscope and image parameters.

        Args:
            cs (float): Spherical aberration coefficient (mm).
            kv (float): Accelerating voltage (kV).
            q0 (float): Amplitude contrast.
            angpix (float): Pixel size in Angstroms.
            size (int): Image size (for square image length of a side in pixels).
            do_damping (bool, optional): Whether to apply envelope damping. Defaults to True.
            gamma_offset (float, optional): Offset for the gamma phase. Defaults to 0.0.
        """
        super(CTF, self).__init__()
        self.cs = cs
        self.kv = kv
        self.q0 = q0
        self.angpix = angpix
        self.do_damping = do_damping
        self.gamma_offset = gamma_offset
        self.size = size

        self._initialise()

    def _initialise(self):
        """
        Precompute constants and frequency grids needed for CTF calculation.
        Registers buffers for wavelength, transformation matrices, and frequency grids.
        """
        # Convert units
        local_cs = self.cs * 1e7
        local_kv = self.kv * 1e3

        # Compute electron wavelength and constants
        self.register_buffer("lambda_electron", 12.2643247 / torch.sqrt(
            torch.tensor([local_kv * (1 + local_kv * 0.978466e-6)], dtype=torch.float64)), persistent=False)
        self.register_buffer("K1", torch.pi * self.lambda_electron, persistent=False)
        self.register_buffer("K2", (torch.pi / 2 * local_cs * self.lambda_electron ** 3), persistent=False)
        self.register_buffer("K3", torch.arctan(self.q0 / torch.sqrt(torch.tensor([1 - self.q0 ** 2], dtype=torch.float64))), persistent=False)

        oriydim = self.size
        orixdim = self.size // 2 + 1
        ys = oriydim * self.angpix
        x = torch.arange(orixdim, dtype=torch.float64) / ys
        y = torch.cat((torch.arange(orixdim, dtype=torch.float64),
                       torch.arange(start=-oriydim + orixdim, end=0, step=1, dtype=torch.float64)),
                      0) / ys
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        xx = xx.unsqueeze(0).unsqueeze(0)
        yy = yy.unsqueeze(0).unsqueeze(0)
        self.register_buffer("y", yy, persistent=False)
        self.register_buffer("x", xx, persistent=False)

    def get_ctf(self, K4: torch.Tensor, K5: torch.Tensor, Axx: torch.Tensor, Axy: torch.Tensor, Ayy: torch.Tensor) -> torch.Tensor:
        """
        Compute the CTF value for given transformation matrices and phase parameters.

        Args:
            K4 (torch.Tensor): Damping envelope parameter.
            K5 (torch.Tensor): Phase shift parameter.
            Axx (torch.Tensor): Transformation matrix element.
            Axy (torch.Tensor): Transformation matrix element.
            Ayy (torch.Tensor): Transformation matrix element.

        Returns:
            torch.Tensor: The computed CTF values.
        """
        u2 = self.x ** 2 + self.y ** 2
        u4 = u2 ** 2
        gamma = self.K1 * (
                    Axx * self.x ** 2 + 2.0 * Axy * self.x * self.y + Ayy * self.y ** 2) + self.K2 * u4 - K5 - self.K3 + self.gamma_offset
        retval = -torch.sin(gamma)

        if self.do_damping:
            # if self.dose >= 0.0:
            # E = torch.exp(-0.5 * self.dose / (0.245 * torch.pow(u2, -0.8325) + 2.81))
            E = torch.exp(K4 * u2)
            retval *= E
        return retval

    def get_fftw_image(self, phase_shift: torch.Tensor, azimuthal_angle: torch.Tensor,
                       b_fac: torch.Tensor, deltaf_u: torch.Tensor, deltaf_v: torch.Tensor, ones: bool = False) -> torch.Tensor:
        """
        Generate the CTF image in Fourier space for a batch of parameters.

        Args:
            phase_shift (torch.Tensor): Phase shift values (degrees).
            azimuthal_angle (torch.Tensor): Azimuthal angles (degrees).
            b_fac (torch.Tensor): B-factor values.
            deltaf_u (torch.Tensor): Defocus U values.
            deltaf_v (torch.Tensor): Defocus V values.
            ones (bool, optional): If True, returns an array of ones. Defaults to False.

        Returns:
            torch.Tensor: The CTF image in Fourier space.
        """
        if ones:
            return torch.ones_like(self.x).repeat(deltaf_u.shape[0], 1, 1, 1)
        # Compute important Constants
        K4 = (-b_fac.double() / 4.0).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        K5 = torch.deg2rad(phase_shift.double()).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        rad_azimuth = torch.deg2rad(azimuthal_angle.double())
        # defocus_average = -(deltaf_u + deltaf_v) / 2
        # defocus_deviation = -(deltaf_u - deltaf_v) / 2
        cos_az = torch.cos(rad_azimuth)
        sin_az = torch.sin(rad_azimuth)

        # Create transformation matrices
        deltaf_u = deltaf_u.double()
        deltaf_v = deltaf_v.double()
        
        Q = torch.stack((torch.stack((cos_az, sin_az), dim=1),
                         torch.stack((-sin_az, cos_az), dim=1)), dim=1)
        D = torch.stack((torch.stack((-deltaf_u, torch.zeros_like(deltaf_u)), dim=1),
                         torch.stack((torch.zeros_like(deltaf_v), -deltaf_v.double()), dim=1)), dim=1)

        A = Q.mT @ (D @ Q)
        Axx, Axy, Ayy = A[..., 0, 0].unsqueeze(1).unsqueeze(1).unsqueeze(1), A[..., 0, 1].unsqueeze(1).unsqueeze(
            1).unsqueeze(1), A[..., 1, 1].unsqueeze(1).unsqueeze(1).unsqueeze(1)

        # Compute CTF values using broadcasting
        fctf = self.get_ctf(K4, K5, Axx, Axy, Ayy)

        return fctf
