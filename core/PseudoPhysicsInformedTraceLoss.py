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


              .PY FOR CLASS FOR NEW PSEUDO PHYSICS LOSS FUCTION.
"""

"""
PseudoPhysicsInformedTraceLoss.py

Physics-guided trace-level loss for receiver-conditioned surface seismogram emulation.

This module defines a loss function for the TIER 4 stage of the AI surface
seismogram emulation workflow. It is intentionally designed as a trace-level
physics-informed loss, not as a full PINN loss.

Important distinction
---------------------
The current receiver-conditioned neural emulator predicts one trace:

    y_pred = G_theta(velocity_model, receiver_coordinates)

It does not predict the full spatial wavefield u(x, z, t). Therefore, this loss
does not compute the full acoustic PDE residual. Instead, it adds physically
motivated constraints directly on the predicted receiver trace.

Main loss terms
---------------
1. Time-domain MSE loss.
2. Normalized waveform loss.
3. Temporal-derivative loss.
4. Frequency-domain log-amplitude loss.
5. Energy-consistency loss.
6. Arrival / causality loss using:

       t_arrival_min = t_source + d_sr / c_max

   where:

       t_source = source_time_index * dt

   and:

       d_sr = sqrt(((receiver_x - source_x) * dx)^2
                 + ((receiver_z - source_z) * dz)^2)
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorLike = Union[torch.Tensor, float, int]


class PseudoPhysicsInformedTraceLoss(nn.Module):
    """
    Physics-guided trace-level loss for receiver-conditioned seismogram emulation.

    This class is meant to replace a pure ``nn.MSELoss`` during TIER 4 training,
    while keeping the neural network architecture unchanged.

    Expected prediction problem
    ---------------------------
    The model output must be a batch of one-receiver traces:

        y_pred.shape = (batch_size, n_time)
        y_true.shape = (batch_size, n_time)

    The arrival / causality loss also requires geometry and velocity metadata:

        dt, dx, dz
        source_x, source_z
        receiver_x, receiver_z
        velocity_model

    Notes
    -----
    - ``velocity_model`` is expected to contain physical velocities, not normalized
      values. If velocity models are normalized before entering the network, pass
      the unnormalized velocity model to this loss for the arrival term.
    - This is not a full PINN loss because no spatial PDE residual is computed.
    - The class returns both the total loss and a dictionary with individual terms.

    Parameters
    ----------
    alpha_time : float
        Weight of the raw time-domain MSE loss.

    alpha_normalized : float
        Weight of the normalized waveform loss.

    alpha_derivative : float
        Weight of the temporal-derivative loss.

    alpha_frequency : float
        Weight of the frequency-domain log-amplitude loss.

    alpha_energy : float
        Weight of the log-energy consistency loss.

    alpha_arrival : float
        Weight of the arrival / causality loss.

    source_time_index : int or float
        Source activation index ``ist``. The source time is computed as
        ``t_source = source_time_index * dt`` unless ``source_time_s`` is provided.

    source_time_s : float or None
        Optional explicit source activation time in seconds. If provided, this
        value overrides ``source_time_index * dt``.

    arrival_tolerance_s : float
        Soft tolerance subtracted from the minimum admissible arrival time. The
        arrival penalty is applied only for:

            t < t_arrival_min - arrival_tolerance_s

    eps : float
        Small constant used in logarithms and divisions.

    rms_eps : float
        Small constant used when normalizing traces by their RMS value.

    use_dt_in_derivative : bool
        If True, the derivative loss compares finite-difference derivatives
        divided by dt. If False, it compares raw first differences. For this
        project, False is often more stable because dt is small.

    fmin : float or None
        Optional minimum frequency for the spectral loss. If None, no lower
        frequency cutoff is applied.

    fmax : float or None
        Optional maximum frequency for the spectral loss. If None, no upper
        frequency cutoff is applied.

    detach_loss_terms : bool
        If True, the dictionary values are detached from the computation graph.
        The returned total loss always keeps the graph for backpropagation.
    """

    def __init__(
        self,
        alpha_time: float = 1.0,
        alpha_normalized: float = 0.10,
        alpha_derivative: float = 0.05,
        alpha_frequency: float = 0.02,
        alpha_energy: float = 0.05,
        alpha_arrival: float = 0.01,
        source_time_index: Union[int, float] = 100,
        source_time_s: Optional[float] = None,
        arrival_tolerance_s: float = 0.0,
        eps: float = 1.0e-8,
        rms_eps: float = 1.0e-6,
        use_dt_in_derivative: bool = False,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        detach_loss_terms: bool = True,
    ) -> None:
        super().__init__()

        self.alpha_time = float(alpha_time)
        self.alpha_normalized = float(alpha_normalized)
        self.alpha_derivative = float(alpha_derivative)
        self.alpha_frequency = float(alpha_frequency)
        self.alpha_energy = float(alpha_energy)
        self.alpha_arrival = float(alpha_arrival)

        self.source_time_index = float(source_time_index)
        self.source_time_s = None if source_time_s is None else float(source_time_s)
        self.arrival_tolerance_s = float(arrival_tolerance_s)

        self.eps = float(eps)
        self.rms_eps = float(rms_eps)
        self.use_dt_in_derivative = bool(use_dt_in_derivative)
        self.fmin = fmin
        self.fmax = fmax
        self.detach_loss_terms = bool(detach_loss_terms)

        self._validate_weights()

    # ======================================================================
    # Public forward methods
    # ======================================================================
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        dt: Optional[TensorLike] = None,
        dx: Optional[TensorLike] = None,
        dz: Optional[TensorLike] = None,
        velocity_model: Optional[torch.Tensor] = None,
        receiver_x: Optional[TensorLike] = None,
        receiver_z: Optional[TensorLike] = None,
        source_x: Optional[TensorLike] = None,
        source_z: Optional[TensorLike] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the weighted physics-guided trace loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted traces with shape ``(batch_size, n_time)``.

        y_true : torch.Tensor
            Reference finite-difference traces with shape ``(batch_size, n_time)``.

        dt, dx, dz : tensor-like or scalar
            Time and spatial steps. They may be scalars or batch tensors.

        velocity_model : torch.Tensor
            Velocity model batch with shape ``(batch_size, 1, nz, nx)`` or
            ``(batch_size, nz, nx)``. Required only when ``alpha_arrival > 0``.

        receiver_x, receiver_z : tensor-like or scalar
            Raw receiver grid coordinates. Required only when ``alpha_arrival > 0``.

        source_x, source_z : tensor-like or scalar
            Raw source grid coordinates. Required only when ``alpha_arrival > 0``.

        Returns
        -------
        total_loss : torch.Tensor
            Scalar loss used for backpropagation.

        loss_terms : dict[str, torch.Tensor]
            Dictionary with individual and weighted loss terms.
        """
        y_pred, y_true = self._validate_traces(y_pred=y_pred, y_true=y_true)

        device = y_pred.device
        dtype = y_pred.dtype
        batch_size, n_time = y_pred.shape

        zero = torch.zeros((), device=device, dtype=dtype)

        loss_time = self._time_domain_loss(y_pred, y_true) if self.alpha_time != 0.0 else zero
        loss_normalized = (
            self._normalized_waveform_loss(y_pred, y_true)
            if self.alpha_normalized != 0.0
            else zero
        )
        loss_derivative = (
            self._temporal_derivative_loss(y_pred, y_true, dt)
            if self.alpha_derivative != 0.0
            else zero
        )
        loss_frequency = (
            self._frequency_domain_loss(y_pred, y_true, dt)
            if self.alpha_frequency != 0.0
            else zero
        )
        loss_energy = (
            self._energy_consistency_loss(y_pred, y_true, dt)
            if self.alpha_energy != 0.0
            else zero
        )
        loss_arrival = (
            self._arrival_causality_loss(
                y_pred=y_pred,
                dt=dt,
                dx=dx,
                dz=dz,
                velocity_model=velocity_model,
                receiver_x=receiver_x,
                receiver_z=receiver_z,
                source_x=source_x,
                source_z=source_z,
            )
            if self.alpha_arrival != 0.0
            else zero
        )

        weighted_time = self.alpha_time * loss_time
        weighted_normalized = self.alpha_normalized * loss_normalized
        weighted_derivative = self.alpha_derivative * loss_derivative
        weighted_frequency = self.alpha_frequency * loss_frequency
        weighted_energy = self.alpha_energy * loss_energy
        weighted_arrival = self.alpha_arrival * loss_arrival

        total_loss = (
            weighted_time
            + weighted_normalized
            + weighted_derivative
            + weighted_frequency
            + weighted_energy
            + weighted_arrival
        )

        if not torch.isfinite(total_loss):
            raise ValueError(
                "PseudoPhysicsInformedTraceLoss produced a non-finite total loss. "
                "Check loss weights, trace amplitudes, dt, dx, dz, and velocity_model."
            )

        loss_terms = {
            "loss_total": total_loss,
            "loss_time": loss_time,
            "loss_normalized": loss_normalized,
            "loss_derivative": loss_derivative,
            "loss_frequency": loss_frequency,
            "loss_energy": loss_energy,
            "loss_arrival": loss_arrival,
            "weighted_time": weighted_time,
            "weighted_normalized": weighted_normalized,
            "weighted_derivative": weighted_derivative,
            "weighted_frequency": weighted_frequency,
            "weighted_energy": weighted_energy,
            "weighted_arrival": weighted_arrival,
        }

        if self.detach_loss_terms:
            loss_terms = {key: value.detach() for key, value in loss_terms.items()}

        return total_loss, loss_terms

    def forward_from_metadata(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        velocity_model: torch.Tensor,
        metadata_batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Convenience wrapper for the current HDF5ReceiverQueryTorchDataset output.

        The receiver-query dataset returns metadata keys such as:

            metadata_batch["dt"]
            metadata_batch["dx"]
            metadata_batch["dz"]
            metadata_batch["source_x"]
            metadata_batch["source_z"]
            metadata_batch["receiver_x"]
            metadata_batch["receiver_z"]

        This method extracts those keys and calls ``forward``.
        """
        required_keys = [
            "dt",
            "dx",
            "dz",
            "source_x",
            "source_z",
            "receiver_x",
            "receiver_z",
        ]
        missing_keys = [key for key in required_keys if key not in metadata_batch]
        if missing_keys:
            raise KeyError(
                "metadata_batch is missing keys required by the physics-informed loss: "
                f"{missing_keys}"
            )

        return self.forward(
            y_pred=y_pred,
            y_true=y_true,
            dt=metadata_batch["dt"],
            dx=metadata_batch["dx"],
            dz=metadata_batch["dz"],
            velocity_model=velocity_model,
            receiver_x=metadata_batch["receiver_x"],
            receiver_z=metadata_batch["receiver_z"],
            source_x=metadata_batch["source_x"],
            source_z=metadata_batch["source_z"],
        )

    # ======================================================================
    # Individual loss terms
    # ======================================================================
    def _time_domain_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute raw time-domain MSE."""
        return F.mse_loss(y_pred, y_true)

    def _normalized_waveform_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compare trace shape after RMS normalization.

        This term is specifically useful when the network tends to collapse toward
        near-zero predictions. It forces the model to learn waveform shape, not only
        absolute amplitude.
        """
        pred_rms = torch.sqrt(torch.mean(y_pred**2, dim=-1, keepdim=True) + self.rms_eps)
        true_rms = torch.sqrt(torch.mean(y_true**2, dim=-1, keepdim=True) + self.rms_eps)

        y_pred_norm = y_pred / pred_rms
        y_true_norm = y_true / true_rms

        return F.mse_loss(y_pred_norm, y_true_norm)

    def _temporal_derivative_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        dt: Optional[TensorLike],
    ) -> torch.Tensor:
        """Compare first temporal differences of predicted and reference traces."""
        dy_pred = y_pred[:, 1:] - y_pred[:, :-1]
        dy_true = y_true[:, 1:] - y_true[:, :-1]

        if self.use_dt_in_derivative:
            if dt is None:
                raise ValueError("dt is required when use_dt_in_derivative=True.")
            dt_batch = self._to_batch_vector(dt, y_pred.shape[0], y_pred.device, y_pred.dtype)
            dy_pred = dy_pred / dt_batch[:, None]
            dy_true = dy_true / dt_batch[:, None]

        return F.mse_loss(dy_pred, dy_true)

    def _frequency_domain_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        dt: Optional[TensorLike],
    ) -> torch.Tensor:
        """Compare log-amplitude spectra using rFFT."""
        pred_fft = torch.fft.rfft(y_pred, dim=-1)
        true_fft = torch.fft.rfft(y_true, dim=-1)

        pred_amp = torch.sqrt(pred_fft.real**2 + pred_fft.imag**2 + self.eps)
        true_amp = torch.sqrt(true_fft.real**2 + true_fft.imag**2 + self.eps)

        pred_log_amp = torch.log(pred_amp + self.eps)
        true_log_amp = torch.log(true_amp + self.eps)

        if self.fmin is not None or self.fmax is not None:
            if dt is None:
                raise ValueError("dt is required when fmin or fmax is used in the frequency loss.")

            # The current datasets generally have one constant dt. For a frequency
            # mask, use the first dt value and keep the same spectral bins for the batch.
            dt_batch = self._to_batch_vector(dt, y_pred.shape[0], y_pred.device, y_pred.dtype)
            dt_ref = dt_batch[0]
            freqs = torch.fft.rfftfreq(y_pred.shape[-1], d=float(dt_ref.detach().cpu()))
            freqs = freqs.to(device=y_pred.device, dtype=y_pred.dtype)

            mask = torch.ones_like(freqs, dtype=torch.bool)
            if self.fmin is not None:
                mask = mask & (freqs >= float(self.fmin))
            if self.fmax is not None:
                mask = mask & (freqs <= float(self.fmax))

            if not torch.any(mask):
                raise ValueError(
                    "The selected frequency band is empty. Check fmin, fmax, dt, and n_time."
                )

            pred_log_amp = pred_log_amp[:, mask]
            true_log_amp = true_log_amp[:, mask]

        return F.mse_loss(pred_log_amp, true_log_amp)

    def _energy_consistency_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        dt: Optional[TensorLike],
    ) -> torch.Tensor:
        """Compare log trace energy between prediction and reference."""
        if dt is None:
            raise ValueError("dt is required for the energy-consistency loss.")

        dt_batch = self._to_batch_vector(dt, y_pred.shape[0], y_pred.device, y_pred.dtype)

        energy_pred = torch.sum(y_pred**2, dim=-1) * dt_batch
        energy_true = torch.sum(y_true**2, dim=-1) * dt_batch

        log_energy_pred = torch.log(energy_pred + self.eps)
        log_energy_true = torch.log(energy_true + self.eps)

        return torch.mean((log_energy_pred - log_energy_true) ** 2)

    def _arrival_causality_loss(
        self,
        y_pred: torch.Tensor,
        dt: Optional[TensorLike],
        dx: Optional[TensorLike],
        dz: Optional[TensorLike],
        velocity_model: Optional[torch.Tensor],
        receiver_x: Optional[TensorLike],
        receiver_z: Optional[TensorLike],
        source_x: Optional[TensorLike],
        source_z: Optional[TensorLike],
    ) -> torch.Tensor:
        """
        Penalize predicted energy before the earliest physically plausible arrival.

        The minimum arrival time is computed as:

            t_arrival_min = t_source + d_sr / c_max

        This is a soft causality term. It does not force the exact first arrival.
        It only discourages strong energy before any direct wave could plausibly
        arrive under the maximum velocity of the current velocity model.
        """
        if dt is None or dx is None or dz is None:
            raise ValueError("dt, dx, and dz are required for the arrival / causality loss.")
        if velocity_model is None:
            raise ValueError("velocity_model is required for the arrival / causality loss.")
        if receiver_x is None or receiver_z is None or source_x is None or source_z is None:
            raise ValueError(
                "receiver_x, receiver_z, source_x, and source_z are required for the arrival / causality loss."
            )

        batch_size, n_time = y_pred.shape
        device = y_pred.device
        dtype = y_pred.dtype

        dt_batch = self._to_batch_vector(dt, batch_size, device, dtype)
        dx_batch = self._to_batch_vector(dx, batch_size, device, dtype)
        dz_batch = self._to_batch_vector(dz, batch_size, device, dtype)

        receiver_x_batch = self._to_batch_vector(receiver_x, batch_size, device, dtype)
        receiver_z_batch = self._to_batch_vector(receiver_z, batch_size, device, dtype)
        source_x_batch = self._to_batch_vector(source_x, batch_size, device, dtype)
        source_z_batch = self._to_batch_vector(source_z, batch_size, device, dtype)

        velocity_model = velocity_model.to(device=device, dtype=dtype)
        if velocity_model.shape[0] != batch_size:
            raise ValueError(
                "velocity_model batch dimension must match y_pred batch dimension. "
                f"Got velocity_model.shape[0]={velocity_model.shape[0]} and batch_size={batch_size}."
            )

        c_max = velocity_model.reshape(batch_size, -1).max(dim=1).values
        c_max = torch.clamp(c_max, min=self.eps)

        distance_sr = torch.sqrt(
            ((receiver_x_batch - source_x_batch) * dx_batch) ** 2
            + ((receiver_z_batch - source_z_batch) * dz_batch) ** 2
            + self.eps
        )

        if self.source_time_s is None:
            source_time = self.source_time_index * dt_batch
        else:
            source_time = torch.full_like(dt_batch, float(self.source_time_s))

        t_arrival_min = source_time + distance_sr / c_max
        t_limit = t_arrival_min - self.arrival_tolerance_s

        time_indices = torch.arange(n_time, device=device, dtype=dtype)
        time_axis = time_indices[None, :] * dt_batch[:, None]
        pre_arrival_mask = time_axis < t_limit[:, None]

        # Convert the boolean mask to a floating weight. This avoids indexing with
        # ragged masks and keeps the computation fully batched.
        pre_arrival_weight = pre_arrival_mask.to(dtype=dtype)
        numerator = torch.sum(pre_arrival_weight * y_pred**2, dim=-1)
        denominator = torch.sum(pre_arrival_weight, dim=-1).clamp_min(1.0)

        sample_loss = numerator / denominator
        return torch.mean(sample_loss)

    # ======================================================================
    # Utility methods
    # ======================================================================
    def _validate_weights(self) -> None:
        """Validate non-negative loss weights."""
        weights = {
            "alpha_time": self.alpha_time,
            "alpha_normalized": self.alpha_normalized,
            "alpha_derivative": self.alpha_derivative,
            "alpha_frequency": self.alpha_frequency,
            "alpha_energy": self.alpha_energy,
            "alpha_arrival": self.alpha_arrival,
        }
        negative_weights = {key: value for key, value in weights.items() if value < 0.0}
        if negative_weights:
            raise ValueError(f"Loss weights must be non-negative. Got: {negative_weights}")

        if self.arrival_tolerance_s < 0.0:
            raise ValueError("arrival_tolerance_s must be non-negative.")

        if self.fmin is not None and self.fmin < 0.0:
            raise ValueError("fmin must be non-negative when provided.")

        if self.fmax is not None and self.fmax <= 0.0:
            raise ValueError("fmax must be positive when provided.")

        if self.fmin is not None and self.fmax is not None and self.fmin >= self.fmax:
            raise ValueError("fmin must be smaller than fmax.")

    def _validate_traces(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Validate trace tensors and convert them to compatible floating tensors."""
        if not torch.is_tensor(y_pred):
            raise TypeError("y_pred must be a torch.Tensor.")
        if not torch.is_tensor(y_true):
            raise TypeError("y_true must be a torch.Tensor.")

        if y_pred.ndim != 2:
            raise ValueError(f"y_pred must have shape (batch_size, n_time). Got {tuple(y_pred.shape)}.")
        if y_true.ndim != 2:
            raise ValueError(f"y_true must have shape (batch_size, n_time). Got {tuple(y_true.shape)}.")
        if y_pred.shape != y_true.shape:
            raise ValueError(
                "y_pred and y_true must have the same shape. "
                f"Got {tuple(y_pred.shape)} and {tuple(y_true.shape)}."
            )

        if not torch.is_floating_point(y_pred):
            y_pred = y_pred.float()
        if not torch.is_floating_point(y_true):
            y_true = y_true.float()

        y_true = y_true.to(device=y_pred.device, dtype=y_pred.dtype)
        return y_pred, y_true

    def _to_batch_vector(
        self,
        value: TensorLike,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Convert a scalar or tensor-like value into a 1D batch tensor.

        Accepted inputs:
        - Python scalar
        - scalar tensor
        - tensor with shape (batch_size,)
        - tensor with shape (batch_size, 1)
        """
        if torch.is_tensor(value):
            tensor = value.to(device=device, dtype=dtype)
        else:
            tensor = torch.as_tensor(value, device=device, dtype=dtype)

        if tensor.ndim == 0:
            return tensor.repeat(batch_size)

        tensor = tensor.reshape(-1)

        if tensor.numel() == 1:
            return tensor.repeat(batch_size)

        if tensor.numel() != batch_size:
            raise ValueError(
                "Metadata tensor cannot be converted to a batch vector. "
                f"Expected 1 or {batch_size} values, got {tensor.numel()}."
            )

        return tensor

    def extra_repr(self) -> str:
        """Return a compact readable representation for print(criterion)."""
        return (
            f"alpha_time={self.alpha_time}, "
            f"alpha_normalized={self.alpha_normalized}, "
            f"alpha_derivative={self.alpha_derivative}, "
            f"alpha_frequency={self.alpha_frequency}, "
            f"alpha_energy={self.alpha_energy}, "
            f"alpha_arrival={self.alpha_arrival}, "
            f"source_time_index={self.source_time_index}, "
            f"source_time_s={self.source_time_s}, "
            f"arrival_tolerance_s={self.arrival_tolerance_s}, "
            f"use_dt_in_derivative={self.use_dt_in_derivative}, "
            f"fmin={self.fmin}, "
            f"fmax={self.fmax}"
        )
