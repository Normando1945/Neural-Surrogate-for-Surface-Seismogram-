"""
██╗    ██╗ █████╗ ██╗   ██╗███████╗
██║    ██║██╔══██╗██║   ██║██╔════╝
██║ █╗ ██║███████║██║   ██║█████╗
██║███╗██║██╔══██║╚██╗ ██╔╝██╔══╝
╚███╔███╔╝██║  ██║ ╚████╔╝ ███████╗
 ╚══╝╚══╝ ╚═╝  ╚═╝  ╚═══╝  ╚══════╝

██████╗ ██████╗  ██████╗ ██████╗  █████╗  ██████╗  █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
██╔══██╗██╔══██╗██╔═══██╗██╔══██╗██╔══██╗██╔════╝ ██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
██████╔╝██████╔╝██║   ██║██████╔╝███████║██║  ███╗███████║   ██║   ██║██║   ██║██╔██╗ ██║
██╔═══╝ ██╔══██╗██║   ██║██╔═══╝ ██╔══██║██║   ██║██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
██║     ██║  ██║╚██████╔╝██║     ██║  ██║╚██████╔╝██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝

SIMULATION OF FINITE DIFFERENCE METHOD OF 2D ACOUSTIC WAVE EQUATION, IN HETEROGNEUS MEDIUM
                    AI SURFACE SEISMOGRAM EMULATION WORKFLOW
                  Velocity Models -> Wavefield -> Seismograms

                 Author: Msc. Ing. Carlos Andrés Celi Sánchez


                     .PY FOR VELOCITY MODELS CLASS.
"""

import numpy as np


class VelocityModel2DGenerator:
    """
    Generator of simple 2D velocity models for the first ML feasibility stage.

    Philosophy
    ----------
    This class is designed to generate a controlled family of velocity models
    in which only the medium varies, while the numerical grid remains fixed.

    Recommended first family:
    - background medium
    - one smooth localized anomaly

    The generated metadata are directly compatible with:
    - animation2D_FDM(sample_metadata=...)
    - HDF5SurfaceSeismogramWriter

    Parameters
    ----------
    nx : int
        Number of grid points in x direction.

    nz : int
        Number of grid points in z direction.

    background_velocity : float
        Reference background velocity of the medium.

    seed : int or None, optional
        Random seed for reproducibility.
    """

    def __init__(self, nx, nz, background_velocity=3000.0, seed=None):
        self.nx = int(nx)
        self.nz = int(nz)
        self.background_velocity = float(background_velocity)
        self.rng = np.random.default_rng(seed)

        # Grid in index space
        self.ix, self.iz = np.meshgrid(
            np.arange(self.nx, dtype=float),
            np.arange(self.nz, dtype=float)
        )

    # ======================================================================
    # Smooth Gaussian anomaly model
    # ======================================================================
    def gaussian_anomaly(
        self,
        anomaly_center_x,
        anomaly_center_z,
        anomaly_radius_x,
        anomaly_radius_z,
        anomaly_velocity_contrast,
        clip_min_velocity=300.0,
    ):
        """
        Create a velocity model with one smooth Gaussian anomaly.

        The anomaly is defined in index space.

        Parameters
        ----------
        anomaly_center_x : float
            x-position of the anomaly center in grid indices.

        anomaly_center_z : float
            z-position of the anomaly center in grid indices.

        anomaly_radius_x : float
            Characteristic radius of the anomaly in x direction (index units).

        anomaly_radius_z : float
            Characteristic radius of the anomaly in z direction (index units).

        anomaly_velocity_contrast : float
            Velocity contrast added to the background velocity.
            Positive values create faster anomalies.
            Negative values create slower anomalies.

        clip_min_velocity : float, optional
            Minimum allowed velocity after applying the anomaly.

        Returns
        -------
        c : ndarray, shape (nz, nx)
            Velocity model.

        metadata : dict
            Dictionary compatible with the HDF5 writer.
        """
        x0 = float(anomaly_center_x)
        z0 = float(anomaly_center_z)
        rx = float(anomaly_radius_x)
        rz = float(anomaly_radius_z)
        dv = float(anomaly_velocity_contrast)

        if rx <= 0:
            raise ValueError("anomaly_radius_x must be positive.")
        if rz <= 0:
            raise ValueError("anomaly_radius_z must be positive.")

        # gaussian = np.exp(
        #     -(
        #         ((self.ix - x0) ** 2) / (rx ** 2)
        #         + ((self.iz - z0) ** 2) / (rz ** 2)
        #     )
        # )
        # c = self.background_velocity + dv * gaussian
        # c = np.maximum(c, clip_min_velocity)

        ellipse = (
            ((self.ix - x0) / rx) ** 2
            + ((self.iz - z0) / rz) ** 2
        )

        c = np.full((self.nz, self.nx), self.background_velocity, dtype=float)

        # Three internal discrete velocity zones
        core_mask  = ellipse <= 0.20
        mid_mask   = (ellipse > 0.20) & (ellipse <= 0.55)
        outer_mask = (ellipse > 0.55) & (ellipse <= 1.00)

        c[core_mask]  += 1.00 * dv
        c[mid_mask]   += 0.60 * dv
        c[outer_mask] += 0.25 * dv

        c = np.maximum(c, clip_min_velocity)


        metadata = {
            "background_velocity": self.background_velocity,
            "anomaly_center_x": x0,
            "anomaly_center_z": z0,
            "anomaly_radius_x": rx,
            "anomaly_radius_z": rz,
            "anomaly_velocity_contrast": dv,
        }

        return c, metadata

    # ======================================================================
    # 3. Random Gaussian anomaly model
    # ======================================================================
    def random_gaussian_anomaly(
        self,
        x_frac_range=(0.20, 0.80),
        z_frac_range=(0.20, 0.80),
        radius_x_range=(8.0, 25.0),
        radius_z_range=(8.0, 25.0),
        contrast_range=(-1200.0, 1200.0),
        clip_min_velocity=300.0,
    ):
        """
        Create one random smooth Gaussian anomaly model.

        The anomaly center is sampled as a fraction of the grid dimensions,
        while the radii are sampled directly in index units.

        Parameters
        ----------
        x_frac_range : tuple(float, float), optional
            Fractional range for anomaly_center_x relative to nx.

        z_frac_range : tuple(float, float), optional
            Fractional range for anomaly_center_z relative to nz.

        radius_x_range : tuple(float, float), optional
            Sampling range for anomaly_radius_x in grid-index units.

        radius_z_range : tuple(float, float), optional
            Sampling range for anomaly_radius_z in grid-index units.

        contrast_range : tuple(float, float), optional
            Sampling range for anomaly_velocity_contrast.

        clip_min_velocity : float, optional
            Minimum allowed velocity after applying the anomaly.

        Returns
        -------
        c : ndarray, shape (nz, nx)
            Velocity model.

        metadata : dict
            Dictionary compatible with the HDF5 writer.
        """
        x0 = self.rng.uniform(x_frac_range[0] * self.nx, x_frac_range[1] * self.nx)
        z0 = self.rng.uniform(z_frac_range[0] * self.nz, z_frac_range[1] * self.nz)

        rx = self.rng.uniform(radius_x_range[0], radius_x_range[1])
        rz = self.rng.uniform(radius_z_range[0], radius_z_range[1])

        dv = self.rng.uniform(contrast_range[0], contrast_range[1])

        return self.gaussian_anomaly(
            anomaly_center_x=x0,
            anomaly_center_z=z0,
            anomaly_radius_x=rx,
            anomaly_radius_z=rz,
            anomaly_velocity_contrast=dv,
            clip_min_velocity=clip_min_velocity,
        )

    # ======================================================================
    # 4. Batch parameter sampler
    # ======================================================================
    def sample_random_parameters(
        self,
        n_samples,
        x_frac_range=(0.20, 0.80),
        z_frac_range=(0.20, 0.80),
        radius_x_range=(8.0, 25.0),
        radius_z_range=(8.0, 25.0),
        contrast_range=(-1200.0, 1200.0),
    ):
        """
        Sample random anomaly parameters without building the velocity fields yet.

        This is useful if you want to inspect the parameter space before
        generating the full pilot dataset.

        Returns
        -------
        params_list : list of dict
            List containing one parameter dictionary per sample.
        """
        params_list = []

        for _ in range(int(n_samples)):
            params = {
                "background_velocity": self.background_velocity,
                "anomaly_center_x": self.rng.uniform(x_frac_range[0] * self.nx, x_frac_range[1] * self.nx),
                "anomaly_center_z": self.rng.uniform(z_frac_range[0] * self.nz, z_frac_range[1] * self.nz),
                "anomaly_radius_x": self.rng.uniform(radius_x_range[0], radius_x_range[1]),
                "anomaly_radius_z": self.rng.uniform(radius_z_range[0], radius_z_range[1]),
                "anomaly_velocity_contrast": self.rng.uniform(contrast_range[0], contrast_range[1]),
            }
            params_list.append(params)

        return params_list

    # ======================================================================
    # 5. Build model from explicit metadata dictionary
    # ======================================================================
    def build_from_metadata(self, metadata, clip_min_velocity=300.0):
        """
        Build a velocity model from an explicit metadata dictionary.

        Parameters
        ----------
        metadata : dict
            Must contain:
            - background_velocity
            - anomaly_center_x
            - anomaly_center_z
            - anomaly_radius_x
            - anomaly_radius_z
            - anomaly_velocity_contrast

        clip_min_velocity : float, optional
            Minimum allowed velocity after applying the anomaly.

        Returns
        -------
        c : ndarray, shape (nz, nx)
            Velocity model.

        metadata_out : dict
            Cleaned metadata dictionary.
        """
        bg = float(metadata["background_velocity"])

        # Temporarily replace background velocity if needed
        original_bg = self.background_velocity
        self.background_velocity = bg

        c, metadata_out = self.gaussian_anomaly(
            anomaly_center_x=metadata["anomaly_center_x"],
            anomaly_center_z=metadata["anomaly_center_z"],
            anomaly_radius_x=metadata["anomaly_radius_x"],
            anomaly_radius_z=metadata["anomaly_radius_z"],
            anomaly_velocity_contrast=metadata["anomaly_velocity_contrast"],
            clip_min_velocity=clip_min_velocity,
        )

        self.background_velocity = original_bg

        return c, metadata_out