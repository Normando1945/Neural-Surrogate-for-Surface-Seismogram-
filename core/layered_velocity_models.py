"""
Layered 2D velocity model generator.

This module is intentionally separated from velocity_models.py so the current
VelocityModel2DGenerator workflow remains unchanged.
"""

import ast

import numpy as np


class LayeredVelocityModel2DGenerator:
    """
    Generator of 2D layered velocity models with irregular interfaces.

    The class mirrors the practical structure of VelocityModel2DGenerator:
    it receives the grid size, owns a reproducible random generator, returns
    velocity fields with shape (nz, nx), and emits metadata that can be passed
    to the existing simulation and HDF5 writers.

    Parameters
    ----------
    nx : int
        Number of grid points in x direction.

    nz : int
        Number of grid points in z direction.

    background_velocity : float, optional
        Reference shallow-layer velocity used by default random sampling.

    seed : int or None, optional
        Random seed for reproducible layered models.
    """

    def __init__(self, nx, nz, background_velocity=1500.0, seed=None):
        self.nx = int(nx)
        self.nz = int(nz)
        self.background_velocity = float(background_velocity)
        self.rng = np.random.default_rng(seed)

        # Grid in index space, consistent with VelocityModel2DGenerator.
        self.ix, self.iz = np.meshgrid(
            np.arange(self.nx, dtype=float),
            np.arange(self.nz, dtype=float),
        )

    # ======================================================================
    # Explicit layered model
    # ======================================================================
    def layered_model(
        self,
        layer_velocities,
        interface_depths=None,
        interface_profiles=None,
        clip_min_velocity=300.0,
        clip_max_velocity=None,
        min_layer_thickness=1.0,
        store_interface_profiles=True,
    ):
        """
        Build a layered velocity model from explicit layer definitions.

        Parameters
        ----------
        layer_velocities : array-like, shape (n_layers,)
            Velocity assigned to each layer from top to bottom.

        interface_depths : array-like, optional
            Mean/constant interface depths in grid indices. Must have
            n_layers - 1 values. Used only when interface_profiles is None.

        interface_profiles : array-like, optional
            Full interface geometry with shape (n_layers - 1, nx). Each row is
            an interface depth profile along x, in grid indices.

        clip_min_velocity : float, optional
            Minimum allowed velocity.

        clip_max_velocity : float or None, optional
            Optional maximum allowed velocity.

        min_layer_thickness : float, optional
            Minimum vertical separation between interfaces in grid points.

        store_interface_profiles : bool, optional
            If True, store the full interface profiles in the returned metadata.

        Returns
        -------
        c : ndarray, shape (nz, nx)
            Layered velocity model.

        metadata : dict
            Metadata compatible with the existing writers. The legacy Gaussian
            metadata keys are included as NaN so old code paths can keep running.
        """
        layer_velocities = self._as_1d_float_array(layer_velocities, "layer_velocities")
        if layer_velocities.size < 1:
            raise ValueError("layer_velocities must contain at least one layer.")

        interface_profiles = self._prepare_interface_profiles(
            n_layers=layer_velocities.size,
            interface_depths=interface_depths,
            interface_profiles=interface_profiles,
            min_layer_thickness=min_layer_thickness,
        )

        layer_indices = self._layer_index_grid(interface_profiles)
        c = layer_velocities[layer_indices]
        c = np.maximum(c, float(clip_min_velocity))
        if clip_max_velocity is not None:
            c = np.minimum(c, float(clip_max_velocity))

        metadata = self._base_metadata(background_velocity=float(layer_velocities[0]))
        metadata.update(
            {
                "model_family": "layered",
                "layer_count": int(layer_velocities.size),
                "layer_velocities": layer_velocities.copy(),
                "interface_mean_depths": self._interface_stat(interface_profiles, np.mean),
                "interface_min_depths": self._interface_stat(interface_profiles, np.min),
                "interface_max_depths": self._interface_stat(interface_profiles, np.max),
                "min_layer_thickness": float(min_layer_thickness),
            }
        )

        if store_interface_profiles:
            metadata["interface_profiles"] = interface_profiles.copy()

        return c, metadata

    # ======================================================================
    # Random layered model
    # ======================================================================
    def random_layered_model(
        self,
        n_layers_range=(4, 8),
        surface_velocity_range=None,
        velocity_step_range=(250.0, 650.0),
        max_velocity=4500.0,
        interface_roughness_frac_range=(0.015, 0.060),
        interface_correlation_frac_range=(0.08, 0.25),
        interface_jitter_frac=0.25,
        min_layer_thickness_frac=0.08,
        fault_probability=0.35,
        fault_offset_frac_range=(0.015, 0.070),
        fault_width_frac_range=(0.006, 0.025),
        clip_min_velocity=300.0,
        clip_max_velocity=None,
        store_interface_profiles=True,
    ):
        """
        Create one random layered velocity model.

        The model keeps a geologically simple but useful structure:
        velocities generally increase with depth, while interfaces include
        long-wavelength roughness and optional fault-like lateral offsets.

        Returns
        -------
        c : ndarray, shape (nz, nx)
            Velocity model.

        metadata : dict
            Layer parameters and compatibility metadata.
        """
        metadata = self._sample_random_layer_metadata(
            n_layers_range=n_layers_range,
            surface_velocity_range=surface_velocity_range,
            velocity_step_range=velocity_step_range,
            max_velocity=max_velocity,
            interface_roughness_frac_range=interface_roughness_frac_range,
            interface_correlation_frac_range=interface_correlation_frac_range,
            interface_jitter_frac=interface_jitter_frac,
            min_layer_thickness_frac=min_layer_thickness_frac,
            fault_probability=fault_probability,
            fault_offset_frac_range=fault_offset_frac_range,
            fault_width_frac_range=fault_width_frac_range,
        )

        c, metadata_out = self.layered_model(
            layer_velocities=metadata["layer_velocities"],
            interface_profiles=metadata["interface_profiles"],
            clip_min_velocity=clip_min_velocity,
            clip_max_velocity=clip_max_velocity,
            min_layer_thickness=metadata["min_layer_thickness"],
            store_interface_profiles=store_interface_profiles,
        )

        for key in (
            "interface_amplitudes",
            "interface_correlation_lengths",
            "interface_fault_x_indices",
            "interface_fault_offsets",
            "interface_fault_widths",
        ):
            metadata_out[key] = metadata[key]

        return c, metadata_out

    # ======================================================================
    # Batch parameter sampler
    # ======================================================================
    def sample_random_parameters(
        self,
        n_samples,
        n_layers_range=(4, 8),
        surface_velocity_range=None,
        velocity_step_range=(250.0, 650.0),
        max_velocity=4500.0,
        interface_roughness_frac_range=(0.015, 0.060),
        interface_correlation_frac_range=(0.08, 0.25),
        interface_jitter_frac=0.25,
        min_layer_thickness_frac=0.08,
        fault_probability=0.35,
        fault_offset_frac_range=(0.015, 0.070),
        fault_width_frac_range=(0.006, 0.025),
    ):
        """
        Sample random layered-model parameters without running simulations.

        Returns
        -------
        params_list : list of dict
            One metadata dictionary per sampled model.
        """
        params_list = []

        for _ in range(int(n_samples)):
            params_list.append(
                self._sample_random_layer_metadata(
                    n_layers_range=n_layers_range,
                    surface_velocity_range=surface_velocity_range,
                    velocity_step_range=velocity_step_range,
                    max_velocity=max_velocity,
                    interface_roughness_frac_range=interface_roughness_frac_range,
                    interface_correlation_frac_range=interface_correlation_frac_range,
                    interface_jitter_frac=interface_jitter_frac,
                    min_layer_thickness_frac=min_layer_thickness_frac,
                    fault_probability=fault_probability,
                    fault_offset_frac_range=fault_offset_frac_range,
                    fault_width_frac_range=fault_width_frac_range,
                )
            )

        return params_list

    # ======================================================================
    # Build model from explicit metadata dictionary
    # ======================================================================
    def build_from_metadata(
        self,
        metadata,
        clip_min_velocity=300.0,
        clip_max_velocity=None,
        store_interface_profiles=True,
    ):
        """
        Rebuild a layered velocity model from a metadata dictionary.

        Exact reconstruction requires "layer_velocities" and either
        "interface_profiles" or "interface_mean_depths" in metadata.
        """
        layer_velocities = self._metadata_array(metadata, "layer_velocities", ndim=1)

        if "interface_profiles" in metadata:
            interface_profiles = self._metadata_array(metadata, "interface_profiles", ndim=2)
            interface_depths = None
        elif "interface_mean_depths" in metadata:
            interface_profiles = None
            interface_depths = self._metadata_array(metadata, "interface_mean_depths", ndim=1)
        else:
            raise KeyError(
                "metadata must contain 'interface_profiles' or 'interface_mean_depths'."
            )

        min_layer_thickness = float(metadata.get("min_layer_thickness", 1.0))

        return self.layered_model(
            layer_velocities=layer_velocities,
            interface_depths=interface_depths,
            interface_profiles=interface_profiles,
            clip_min_velocity=clip_min_velocity,
            clip_max_velocity=clip_max_velocity,
            min_layer_thickness=min_layer_thickness,
            store_interface_profiles=store_interface_profiles,
        )

    # ======================================================================
    # Internal random geometry helpers
    # ======================================================================
    def _sample_random_layer_metadata(
        self,
        n_layers_range,
        surface_velocity_range,
        velocity_step_range,
        max_velocity,
        interface_roughness_frac_range,
        interface_correlation_frac_range,
        interface_jitter_frac,
        min_layer_thickness_frac,
        fault_probability,
        fault_offset_frac_range,
        fault_width_frac_range,
    ):
        n_layers = self._sample_layer_count(n_layers_range)
        min_layer_thickness = max(1.0, float(min_layer_thickness_frac) * self.nz)

        if n_layers * min_layer_thickness >= self.nz:
            raise ValueError(
                "min_layer_thickness_frac is too large for the sampled number of layers."
            )

        layer_velocities = self._sample_layer_velocities(
            n_layers=n_layers,
            surface_velocity_range=surface_velocity_range,
            velocity_step_range=velocity_step_range,
            max_velocity=max_velocity,
        )

        interface_mean_depths = self._sample_interface_mean_depths(
            n_layers=n_layers,
            min_layer_thickness=min_layer_thickness,
            interface_jitter_frac=interface_jitter_frac,
        )

        interface_profiles, interface_metadata = self._sample_interface_profiles(
            interface_mean_depths=interface_mean_depths,
            min_layer_thickness=min_layer_thickness,
            interface_roughness_frac_range=interface_roughness_frac_range,
            interface_correlation_frac_range=interface_correlation_frac_range,
            fault_probability=fault_probability,
            fault_offset_frac_range=fault_offset_frac_range,
            fault_width_frac_range=fault_width_frac_range,
        )

        metadata = self._base_metadata(background_velocity=float(layer_velocities[0]))
        metadata.update(
            {
                "model_family": "layered",
                "layer_count": int(n_layers),
                "layer_velocities": layer_velocities,
                "interface_mean_depths": interface_mean_depths,
                "interface_profiles": interface_profiles,
                "min_layer_thickness": float(min_layer_thickness),
            }
        )
        metadata.update(interface_metadata)

        return metadata

    def _sample_layer_count(self, n_layers_range):
        low, high = self._range_pair(n_layers_range, "n_layers_range")
        low = int(np.floor(low))
        high = int(np.floor(high))

        if low < 2:
            raise ValueError("n_layers_range must start at 2 or greater.")
        if high < low:
            raise ValueError("n_layers_range upper bound must be >= lower bound.")

        return int(self.rng.integers(low, high + 1))

    def _sample_layer_velocities(
        self,
        n_layers,
        surface_velocity_range,
        velocity_step_range,
        max_velocity,
    ):
        if surface_velocity_range is None:
            surface_velocity_range = (
                self.background_velocity,
                self.background_velocity,
            )

        vmin, vmax = self._range_pair(surface_velocity_range, "surface_velocity_range")
        step_min, step_max = self._range_pair(velocity_step_range, "velocity_step_range")

        if step_min < 0.0:
            raise ValueError("velocity_step_range must be non-negative.")

        surface_velocity = self.rng.uniform(vmin, vmax)
        velocity_steps = self.rng.uniform(step_min, step_max, size=n_layers - 1)

        layer_velocities = np.empty(n_layers, dtype=float)
        layer_velocities[0] = surface_velocity
        layer_velocities[1:] = surface_velocity + np.cumsum(velocity_steps)

        return np.minimum(layer_velocities, float(max_velocity))

    def _sample_interface_mean_depths(
        self,
        n_layers,
        min_layer_thickness,
        interface_jitter_frac,
    ):
        base_depths = np.linspace(0.0, float(self.nz - 1), n_layers + 1)[1:-1]
        nominal_spacing = float(self.nz - 1) / float(n_layers)

        max_jitter = max(0.0, float(interface_jitter_frac)) * nominal_spacing
        max_jitter = min(max_jitter, max(0.0, 0.45 * (nominal_spacing - min_layer_thickness)))

        if max_jitter > 0.0:
            base_depths = base_depths + self.rng.uniform(
                -max_jitter,
                max_jitter,
                size=base_depths.size,
            )

        return self._enforce_mean_depth_separation(base_depths, min_layer_thickness)

    def _sample_interface_profiles(
        self,
        interface_mean_depths,
        min_layer_thickness,
        interface_roughness_frac_range,
        interface_correlation_frac_range,
        fault_probability,
        fault_offset_frac_range,
        fault_width_frac_range,
    ):
        interface_mean_depths = np.asarray(interface_mean_depths, dtype=float)
        n_interfaces = interface_mean_depths.size

        rough_min, rough_max = self._range_pair(
            interface_roughness_frac_range,
            "interface_roughness_frac_range",
        )
        corr_min, corr_max = self._range_pair(
            interface_correlation_frac_range,
            "interface_correlation_frac_range",
        )
        fault_offset_min, fault_offset_max = self._range_pair(
            fault_offset_frac_range,
            "fault_offset_frac_range",
        )
        fault_width_min, fault_width_max = self._range_pair(
            fault_width_frac_range,
            "fault_width_frac_range",
        )

        profiles = []
        amplitudes = np.zeros(n_interfaces, dtype=float)
        correlation_lengths = np.zeros(n_interfaces, dtype=float)
        fault_x_indices = np.full(n_interfaces, -1.0, dtype=float)
        fault_offsets = np.zeros(n_interfaces, dtype=float)
        fault_widths = np.zeros(n_interfaces, dtype=float)

        x = np.arange(self.nx, dtype=float)

        for i, mean_depth in enumerate(interface_mean_depths):
            amplitude = self.rng.uniform(rough_min, rough_max) * self.nz
            correlation_length = self.rng.uniform(corr_min, corr_max) * self.nx

            profile = mean_depth + self._smooth_random_curve(
                amplitude=amplitude,
                correlation_length=correlation_length,
            )

            if self.rng.random() < float(fault_probability):
                fault_x = self.rng.uniform(0.18 * self.nx, 0.82 * self.nx)
                fault_offset = self.rng.choice([-1.0, 1.0]) * self.rng.uniform(
                    fault_offset_min,
                    fault_offset_max,
                ) * self.nz
                fault_width = max(1.0, self.rng.uniform(fault_width_min, fault_width_max) * self.nx)

                step = 0.5 * fault_offset * (1.0 + np.tanh((x - fault_x) / fault_width))
                profile = profile + step - np.mean(step)

                fault_x_indices[i] = fault_x
                fault_offsets[i] = fault_offset
                fault_widths[i] = fault_width

            profiles.append(profile)
            amplitudes[i] = amplitude
            correlation_lengths[i] = correlation_length

        profiles = np.asarray(profiles, dtype=float)
        profiles = self._enforce_interface_profile_separation(
            profiles,
            min_layer_thickness=min_layer_thickness,
        )

        metadata = {
            "interface_amplitudes": amplitudes,
            "interface_correlation_lengths": correlation_lengths,
            "interface_fault_x_indices": fault_x_indices,
            "interface_fault_offsets": fault_offsets,
            "interface_fault_widths": fault_widths,
        }

        return profiles, metadata

    def _smooth_random_curve(self, amplitude, correlation_length):
        if amplitude <= 0.0:
            return np.zeros(self.nx, dtype=float)

        x = np.arange(self.nx, dtype=float)
        control_step = max(2, int(round(correlation_length)))
        control_x = np.arange(0, self.nx, control_step, dtype=float)
        if control_x.size == 0 or control_x[-1] != self.nx - 1:
            control_x = np.append(control_x, float(self.nx - 1))

        control_y = self.rng.normal(0.0, 1.0, size=control_x.size)
        curve = np.interp(x, control_x, control_y)
        curve = self._smooth_1d(curve, window_points=max(3, int(round(correlation_length))))

        wavelength = self.rng.uniform(max(8.0, 0.35 * self.nx), max(9.0, 1.20 * self.nx))
        phase = self.rng.uniform(0.0, 2.0 * np.pi)
        curve = curve + 0.35 * np.sin(2.0 * np.pi * x / wavelength + phase)

        curve = curve - np.mean(curve)
        max_abs = np.max(np.abs(curve))
        if max_abs == 0.0:
            return np.zeros(self.nx, dtype=float)

        return amplitude * curve / max_abs

    def _smooth_1d(self, values, window_points):
        window_points = int(max(3, window_points))
        if window_points % 2 == 0:
            window_points += 1

        if window_points >= values.size:
            window_points = values.size - 1 if values.size % 2 == 0 else values.size

        if window_points < 3:
            return values.copy()

        kernel = np.hanning(window_points)
        if np.sum(kernel) == 0.0:
            kernel = np.ones(window_points, dtype=float)
        kernel = kernel / np.sum(kernel)

        pad = window_points // 2
        padded = np.pad(values, pad_width=pad, mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    # ======================================================================
    # Internal model-building helpers
    # ======================================================================
    def _prepare_interface_profiles(
        self,
        n_layers,
        interface_depths,
        interface_profiles,
        min_layer_thickness,
    ):
        n_interfaces = int(n_layers) - 1

        if n_interfaces == 0:
            return np.empty((0, self.nx), dtype=float)

        if interface_profiles is not None:
            profiles = np.asarray(interface_profiles, dtype=float)
            if profiles.ndim == 1:
                profiles = profiles.reshape(1, -1)
            if profiles.shape != (n_interfaces, self.nx):
                raise ValueError(
                    "interface_profiles must have shape "
                    f"({n_interfaces}, {self.nx}), got {profiles.shape}."
                )
        else:
            if interface_depths is None:
                raise ValueError(
                    "interface_depths or interface_profiles must be provided "
                    "when more than one layer is used."
                )

            depths = self._as_1d_float_array(interface_depths, "interface_depths")
            if depths.size != n_interfaces:
                raise ValueError(
                    f"interface_depths must contain {n_interfaces} values, got {depths.size}."
                )
            profiles = np.repeat(depths[:, np.newaxis], self.nx, axis=1)

        return self._enforce_interface_profile_separation(
            profiles,
            min_layer_thickness=float(min_layer_thickness),
        )

    def _layer_index_grid(self, interface_profiles):
        if interface_profiles.size == 0:
            return np.zeros((self.nz, self.nx), dtype=np.int32)

        layer_indices = np.sum(
            self.iz[np.newaxis, :, :] >= interface_profiles[:, np.newaxis, :],
            axis=0,
        )

        return layer_indices.astype(np.int32)

    def _enforce_mean_depth_separation(self, mean_depths, min_layer_thickness):
        mean_depths = np.sort(np.asarray(mean_depths, dtype=float))
        n_interfaces = mean_depths.size

        lower_limit = float(min_layer_thickness)
        for i in range(n_interfaces):
            mean_depths[i] = max(mean_depths[i], lower_limit)
            lower_limit = mean_depths[i] + float(min_layer_thickness)

        upper_limit = float(self.nz - 1) - float(min_layer_thickness)
        for i in range(n_interfaces - 1, -1, -1):
            mean_depths[i] = min(mean_depths[i], upper_limit)
            upper_limit = mean_depths[i] - float(min_layer_thickness)

        if np.any(np.diff(np.r_[0.0, mean_depths, float(self.nz - 1)]) < min_layer_thickness):
            raise ValueError("Could not enforce the requested minimum layer thickness.")

        return mean_depths

    def _enforce_interface_profile_separation(self, profiles, min_layer_thickness):
        profiles = np.asarray(profiles, dtype=float).copy()
        if profiles.size == 0:
            return profiles.reshape(0, self.nx)

        n_interfaces = profiles.shape[0]
        min_layer_thickness = float(min_layer_thickness)

        for i in range(n_interfaces):
            lower_limit = min_layer_thickness if i == 0 else profiles[i - 1] + min_layer_thickness
            upper_limit = float(self.nz - 1) - min_layer_thickness * (n_interfaces - i)
            profiles[i] = np.clip(np.maximum(profiles[i], lower_limit), lower_limit, upper_limit)

        for i in range(n_interfaces - 1, -1, -1):
            upper_limit = (
                float(self.nz - 1) - min_layer_thickness
                if i == n_interfaces - 1
                else profiles[i + 1] - min_layer_thickness
            )
            lower_limit = min_layer_thickness * (i + 1)
            profiles[i] = np.clip(np.minimum(profiles[i], upper_limit), lower_limit, upper_limit)

        return profiles

    # ======================================================================
    # Metadata and validation helpers
    # ======================================================================
    def _base_metadata(self, background_velocity):
        return {
            "background_velocity": float(background_velocity),
            "anomaly_center_x": np.nan,
            "anomaly_center_z": np.nan,
            "anomaly_radius_x": np.nan,
            "anomaly_radius_z": np.nan,
            "anomaly_velocity_contrast": np.nan,
        }

    def _interface_stat(self, interface_profiles, reducer):
        if interface_profiles.size == 0:
            return np.asarray([], dtype=float)
        return reducer(interface_profiles, axis=1).astype(float)

    def _range_pair(self, value, name):
        if len(value) != 2:
            raise ValueError(f"{name} must contain exactly two values.")

        low = float(value[0])
        high = float(value[1])

        if high < low:
            raise ValueError(f"{name} upper bound must be >= lower bound.")

        return low, high

    def _as_1d_float_array(self, value, name):
        array = np.asarray(value, dtype=float)
        if array.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array.")
        return array

    def _metadata_array(self, metadata, key, ndim):
        if key not in metadata:
            raise KeyError(f"metadata must contain '{key}'.")

        value = metadata[key]

        if isinstance(value, bytes):
            value = value.decode("utf-8")

        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
                array = np.asarray(value, dtype=float)
            except (SyntaxError, ValueError):
                array = np.fromstring(value.strip("[]"), sep=" ", dtype=float)
        else:
            array = np.asarray(value, dtype=float)

        if array.ndim != ndim:
            raise ValueError(f"metadata['{key}'] must have {ndim} dimensions.")

        return array
