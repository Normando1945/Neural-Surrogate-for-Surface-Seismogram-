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


                .PY FOR LOSS FUCTION (PSEUDO PHYSICS INFORMED).

Masked pseudo-physics-informed trace-level loss for receiver-conditioned
surface seismogram emulation.

This version is masked, duration-aware, arrival-aware, and robust against
near-zero traces. It does not modify the neural-network architecture; it only
replaces the training criterion.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedPseudoPhysicsInformedTraceLoss_v2(nn.Module):
    """
    Duration-aware and arrival-aware masked loss for receiver-conditioned
    one-trace seismogram prediction.
    """

    def __init__(
        self,
        alpha_time: float = 0.05,
        alpha_relative: float = 0.30,
        alpha_correlation: float = 1.00,
        alpha_normalized: float = 0.50,
        alpha_derivative: float = 0.10,
        alpha_energy: float = 0.50,
        alpha_frequency: float = 0.00,
        alpha_arrival: float = 0.00,
        alpha_silence: float = 1.00,
        alpha_partial_time: float = 0.10,
        alpha_partial_energy: float = 0.25,
        alpha_partial_silence: float = 0.50,
        active_rms_threshold: float = 0.50,
        significant_abs_threshold: float = 1.0e-4,
        significant_rel_threshold: float = 0.05,
        min_significant_duration_s: float = 0.05,
        min_post_arrival_duration_s: float = 0.05,
        signal_window_padding_samples: int = 8,
        relative_power_floor: float = 1.0,
        source_time_index: Optional[int] = 100,
        source_time_s: Optional[float] = None,
        arrival_tolerance_s: float = 0.0,
        use_dt_in_derivative: bool = False,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__()

        self.alpha_time = float(alpha_time)
        self.alpha_relative = float(alpha_relative)
        self.alpha_correlation = float(alpha_correlation)
        self.alpha_normalized = float(alpha_normalized)
        self.alpha_derivative = float(alpha_derivative)
        self.alpha_energy = float(alpha_energy)
        self.alpha_frequency = float(alpha_frequency)
        self.alpha_arrival = float(alpha_arrival)
        self.alpha_silence = float(alpha_silence)
        self.alpha_partial_time = float(alpha_partial_time)
        self.alpha_partial_energy = float(alpha_partial_energy)
        self.alpha_partial_silence = float(alpha_partial_silence)

        self.active_rms_threshold = float(active_rms_threshold)
        self.significant_abs_threshold = float(significant_abs_threshold)
        self.significant_rel_threshold = float(significant_rel_threshold)
        self.min_significant_duration_s = float(min_significant_duration_s)
        self.min_post_arrival_duration_s = float(min_post_arrival_duration_s)
        self.signal_window_padding_samples = int(signal_window_padding_samples)
        self.relative_power_floor = float(relative_power_floor)

        self.source_time_index = None if source_time_index is None else int(source_time_index)
        self.source_time_s = None if source_time_s is None else float(source_time_s)
        self.arrival_tolerance_s = float(arrival_tolerance_s)
        self.use_dt_in_derivative = bool(use_dt_in_derivative)
        self.fmin = None if fmin is None else float(fmin)
        self.fmax = None if fmax is None else float(fmax)
        self.eps = float(eps)

        if self.source_time_s is None and self.source_time_index is None:
            raise ValueError("Either source_time_index or source_time_s must be provided.")
        if self.active_rms_threshold < 0.0:
            raise ValueError("active_rms_threshold must be non-negative.")
        if self.significant_abs_threshold < 0.0:
            raise ValueError("significant_abs_threshold must be non-negative.")
        if self.significant_rel_threshold < 0.0:
            raise ValueError("significant_rel_threshold must be non-negative.")
        if self.min_significant_duration_s < 0.0:
            raise ValueError("min_significant_duration_s must be non-negative.")
        if self.min_post_arrival_duration_s < 0.0:
            raise ValueError("min_post_arrival_duration_s must be non-negative.")
        if self.signal_window_padding_samples < 0:
            raise ValueError("signal_window_padding_samples must be non-negative.")
        if self.relative_power_floor <= 0.0:
            raise ValueError("relative_power_floor must be positive.")

    def forward_from_metadata(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        velocity_model: torch.Tensor,
        metadata_batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the loss using the metadata dictionary returned by the dataset."""
        y_pred_2d = self._as_trace_batch(y_pred, name="y_pred")
        batch_size = int(y_pred_2d.shape[0])
        device = y_pred_2d.device

        dt = self._metadata_to_batch_tensor(metadata_batch["dt"], batch_size, device)
        dx = self._metadata_to_batch_tensor(metadata_batch["dx"], batch_size, device)
        dz = self._metadata_to_batch_tensor(metadata_batch["dz"], batch_size, device)
        receiver_x = self._metadata_to_batch_tensor(metadata_batch["receiver_x"], batch_size, device)
        receiver_z = self._metadata_to_batch_tensor(metadata_batch["receiver_z"], batch_size, device)
        source_x = self._metadata_to_batch_tensor(metadata_batch["source_x"], batch_size, device)
        source_z = self._metadata_to_batch_tensor(metadata_batch["source_z"], batch_size, device)

        return self.forward(
            y_pred=y_pred,
            y_true=y_true,
            dt=dt,
            dx=dx,
            dz=dz,
            velocity_model=velocity_model,
            receiver_x=receiver_x,
            receiver_z=receiver_z,
            source_x=source_x,
            source_z=source_z,
        )

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        dt: torch.Tensor,
        dx: torch.Tensor,
        dz: torch.Tensor,
        velocity_model: torch.Tensor,
        receiver_x: torch.Tensor,
        receiver_z: torch.Tensor,
        source_x: torch.Tensor,
        source_z: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute total duration-aware masked pseudo-physics-informed loss."""
        y_pred = self._as_trace_batch(y_pred, name="y_pred")
        y_true = self._as_trace_batch(y_true, name="y_true")
        if y_pred.shape != y_true.shape:
            raise ValueError(f"y_pred and y_true must have the same shape, got {y_pred.shape} and {y_true.shape}.")

        batch_size, n_time = y_true.shape
        device = y_pred.device
        dtype = y_pred.dtype

        dt = self._ensure_batch_vector(dt, batch_size, device, dtype, name="dt")
        dx = self._ensure_batch_vector(dx, batch_size, device, dtype, name="dx")
        dz = self._ensure_batch_vector(dz, batch_size, device, dtype, name="dz")
        receiver_x = self._ensure_batch_vector(receiver_x, batch_size, device, dtype, name="receiver_x")
        receiver_z = self._ensure_batch_vector(receiver_z, batch_size, device, dtype, name="receiver_z")
        source_x = self._ensure_batch_vector(source_x, batch_size, device, dtype, name="source_x")
        source_z = self._ensure_batch_vector(source_z, batch_size, device, dtype, name="source_z")

        true_power = torch.mean(y_true ** 2, dim=1)
        true_rms_raw = torch.sqrt(true_power)
        true_rms = torch.sqrt(true_power + self.eps)
        pred_rms = torch.sqrt(torch.mean(y_pred ** 2, dim=1) + self.eps)

        max_abs_true = torch.max(torch.abs(y_true), dim=1).values
        significant_threshold = torch.maximum(
            torch.full_like(max_abs_true, self.significant_abs_threshold),
            self.significant_rel_threshold * max_abs_true,
        )
        raw_signal_mask = torch.abs(y_true) > significant_threshold[:, None]
        signal_mask = self._dilate_boolean_mask(raw_signal_mask, self.signal_window_padding_samples)
        significant_samples = signal_mask.sum(dim=1).to(dtype=dtype)
        significant_duration_s = significant_samples * dt

        t_arrival_min, post_arrival_duration_s = self._arrival_time_info(
            n_time=n_time,
            dt=dt,
            dx=dx,
            dz=dz,
            velocity_model=velocity_model,
            receiver_x=receiver_x,
            receiver_z=receiver_z,
            source_x=source_x,
            source_z=source_z,
        )

        has_energy = true_rms_raw > self.active_rms_threshold
        has_significant_duration = significant_duration_s >= self.min_significant_duration_s
        has_post_arrival_duration = post_arrival_duration_s >= self.min_post_arrival_duration_s

        active_mask = has_energy & has_significant_duration & has_post_arrival_duration
        partial_mask = has_energy & (~active_mask)
        inactive_mask = ~has_energy

        diff = y_pred - y_true
        time_per_trace = torch.mean(diff ** 2, dim=1)
        relative_per_trace = self._robust_relative_loss_per_trace(diff, true_power)

        normalized_pred = y_pred / pred_rms[:, None]
        normalized_true = y_true / true_rms[:, None]
        normalized_per_trace = torch.mean((normalized_pred - normalized_true) ** 2, dim=1)

        correlation_per_trace = self._correlation_loss_per_trace(y_pred, y_true)
        derivative_per_trace = self._derivative_loss_per_trace(y_pred, y_true, dt)
        energy_per_trace = self._energy_loss_per_trace(y_pred, y_true, dt)
        frequency_per_trace = self._frequency_loss_per_trace(y_pred, y_true, dt)
        arrival_per_trace = self._arrival_loss_per_trace(y_pred=y_pred, dt=dt, t_arrival_min=t_arrival_min)
        silence_per_trace = torch.mean(y_pred ** 2, dim=1)

        partial_time_per_trace = self._windowed_mean_per_trace(diff ** 2, signal_mask)
        partial_energy_per_trace = energy_per_trace
        partial_silence_per_trace = self._windowed_mean_per_trace(y_pred ** 2, ~signal_mask)

        loss_time = self._masked_mean(time_per_trace, active_mask)
        loss_relative = self._masked_mean(relative_per_trace, active_mask)
        loss_correlation = self._masked_mean(correlation_per_trace, active_mask)
        loss_normalized = self._masked_mean(normalized_per_trace, active_mask)
        loss_derivative = self._masked_mean(derivative_per_trace, active_mask)
        loss_energy = self._masked_mean(energy_per_trace, active_mask)
        loss_frequency = self._masked_mean(frequency_per_trace, active_mask)

        loss_arrival_active = self._masked_mean(arrival_per_trace, active_mask)
        loss_arrival_partial = self._masked_mean(arrival_per_trace, partial_mask)
        loss_arrival_inactive = self._masked_mean(arrival_per_trace, inactive_mask)
        loss_arrival = loss_arrival_active + loss_arrival_partial + loss_arrival_inactive

        loss_partial_time = self._masked_mean(partial_time_per_trace, partial_mask)
        loss_partial_energy = self._masked_mean(partial_energy_per_trace, partial_mask)
        loss_partial_silence = self._masked_mean(partial_silence_per_trace, partial_mask)
        loss_silence = self._masked_mean(silence_per_trace, inactive_mask)

        loss_active = (
            self.alpha_time * loss_time
            + self.alpha_relative * loss_relative
            + self.alpha_correlation * loss_correlation
            + self.alpha_normalized * loss_normalized
            + self.alpha_derivative * loss_derivative
            + self.alpha_energy * loss_energy
            + self.alpha_frequency * loss_frequency
            + self.alpha_arrival * loss_arrival_active
        )
        loss_partial = (
            self.alpha_partial_time * loss_partial_time
            + self.alpha_partial_energy * loss_partial_energy
            + self.alpha_partial_silence * loss_partial_silence
            + self.alpha_arrival * loss_arrival_partial
        )
        loss_inactive = self.alpha_silence * loss_silence + self.alpha_arrival * loss_arrival_inactive
        loss_total = loss_active + loss_partial + loss_inactive

        active_count = active_mask.sum().to(dtype=dtype)
        partial_count = partial_mask.sum().to(dtype=dtype)
        inactive_count = inactive_mask.sum().to(dtype=dtype)
        total_count = torch.tensor(float(batch_size), device=device, dtype=dtype)

        loss_terms = {
            "loss_total": loss_total.detach(),
            "loss_active": loss_active.detach(),
            "loss_partial": loss_partial.detach(),
            "loss_inactive": loss_inactive.detach(),
            "loss_time": loss_time.detach(),
            "loss_relative": loss_relative.detach(),
            "loss_correlation": loss_correlation.detach(),
            "loss_normalized": loss_normalized.detach(),
            "loss_derivative": loss_derivative.detach(),
            "loss_energy": loss_energy.detach(),
            "loss_frequency": loss_frequency.detach(),
            "loss_arrival": loss_arrival.detach(),
            "loss_arrival_active": loss_arrival_active.detach(),
            "loss_arrival_partial": loss_arrival_partial.detach(),
            "loss_arrival_inactive": loss_arrival_inactive.detach(),
            "loss_partial_time": loss_partial_time.detach(),
            "loss_partial_energy": loss_partial_energy.detach(),
            "loss_partial_silence": loss_partial_silence.detach(),
            "loss_silence": loss_silence.detach(),
            "active_count": active_count.detach(),
            "partial_count": partial_count.detach(),
            "inactive_count": inactive_count.detach(),
            "active_fraction": (active_count / total_count).detach(),
            "partial_fraction": (partial_count / total_count).detach(),
            "inactive_fraction": (inactive_count / total_count).detach(),
            "target_rms_mean": true_rms.mean().detach(),
            "pred_rms_mean": pred_rms.mean().detach(),
            "significant_duration_mean_s": significant_duration_s.mean().detach(),
            "post_arrival_duration_mean_s": post_arrival_duration_s.mean().detach(),
            "t_arrival_min_mean_s": t_arrival_min.mean().detach(),
        }
        return loss_total, loss_terms

    def _robust_relative_loss_per_trace(self, diff: torch.Tensor, true_power: torch.Tensor) -> torch.Tensor:
        mse = torch.mean(diff ** 2, dim=1)
        denominator = torch.clamp(true_power, min=self.relative_power_floor)
        return torch.log1p(mse / (denominator + self.eps))

    def _correlation_loss_per_trace(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        pred_centered = y_pred - y_pred.mean(dim=1, keepdim=True)
        true_centered = y_true - y_true.mean(dim=1, keepdim=True)
        numerator = torch.sum(pred_centered * true_centered, dim=1)
        denominator = torch.sqrt(torch.sum(pred_centered ** 2, dim=1) + self.eps) * torch.sqrt(
            torch.sum(true_centered ** 2, dim=1) + self.eps
        )
        corr = numerator / (denominator + self.eps)
        corr = torch.clamp(corr, min=-1.0, max=1.0)
        return 1.0 - corr

    def _derivative_loss_per_trace(self, y_pred: torch.Tensor, y_true: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        dy_pred = y_pred[:, 1:] - y_pred[:, :-1]
        dy_true = y_true[:, 1:] - y_true[:, :-1]
        if self.use_dt_in_derivative:
            dy_pred = dy_pred / (dt[:, None] + self.eps)
            dy_true = dy_true / (dt[:, None] + self.eps)
        return torch.mean((dy_pred - dy_true) ** 2, dim=1)

    def _energy_loss_per_trace(self, y_pred: torch.Tensor, y_true: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        energy_pred = torch.sum(y_pred ** 2, dim=1) * dt
        energy_true = torch.sum(y_true ** 2, dim=1) * dt
        return (torch.log(energy_pred + self.eps) - torch.log(energy_true + self.eps)) ** 2

    def _frequency_loss_per_trace(self, y_pred: torch.Tensor, y_true: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        spectrum_pred = torch.fft.rfft(y_pred, dim=1)
        spectrum_true = torch.fft.rfft(y_true, dim=1)
        amp_pred = torch.log(torch.abs(spectrum_pred) + self.eps)
        amp_true = torch.log(torch.abs(spectrum_true) + self.eps)
        if self.fmin is not None or self.fmax is not None:
            n_time = int(y_pred.shape[1])
            dt0 = float(dt[0].detach().cpu().item())
            freqs = torch.fft.rfftfreq(n_time, d=dt0).to(device=y_pred.device, dtype=y_pred.dtype)
            freq_mask = torch.ones_like(freqs, dtype=torch.bool)
            if self.fmin is not None:
                freq_mask = freq_mask & (freqs >= self.fmin)
            if self.fmax is not None:
                freq_mask = freq_mask & (freqs <= self.fmax)
            if torch.any(freq_mask):
                amp_pred = amp_pred[:, freq_mask]
                amp_true = amp_true[:, freq_mask]
        return torch.mean((amp_pred - amp_true) ** 2, dim=1)

    def _arrival_loss_per_trace(self, y_pred: torch.Tensor, dt: torch.Tensor, t_arrival_min: torch.Tensor) -> torch.Tensor:
        _, n_time = y_pred.shape
        dtype = y_pred.dtype
        time_axis = torch.arange(n_time, device=y_pred.device, dtype=dtype)[None, :] * dt[:, None]
        pre_arrival_mask = time_axis < (t_arrival_min - self.arrival_tolerance_s)[:, None]
        early_energy = (y_pred ** 2) * pre_arrival_mask.to(dtype=dtype)
        counts = pre_arrival_mask.sum(dim=1).to(dtype=dtype).clamp_min(1.0)
        return early_energy.sum(dim=1) / counts

    def _arrival_time_info(
        self,
        n_time: int,
        dt: torch.Tensor,
        dx: torch.Tensor,
        dz: torch.Tensor,
        velocity_model: torch.Tensor,
        receiver_x: torch.Tensor,
        receiver_z: torch.Tensor,
        source_x: torch.Tensor,
        source_z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = dt.dtype
        if velocity_model.ndim == 4:
            c_max = velocity_model.flatten(start_dim=1).max(dim=1).values
        elif velocity_model.ndim == 3:
            c_max = velocity_model.flatten(start_dim=1).max(dim=1).values
        else:
            raise ValueError(
                "velocity_model must have shape (batch, 1, nz, nx) or (batch, nz, nx), "
                f"got {velocity_model.shape}."
            )
        c_max = c_max.to(device=dt.device, dtype=dtype)
        distance_sr = torch.sqrt(((receiver_x - source_x) * dx) ** 2 + ((receiver_z - source_z) * dz) ** 2 + self.eps)
        if self.source_time_s is None:
            t_source = float(self.source_time_index) * dt
        else:
            t_source = torch.full_like(dt, fill_value=float(self.source_time_s))
        t_arrival_min = t_source + distance_sr / (c_max + self.eps)
        t_end = (float(n_time) - 1.0) * dt
        post_arrival_duration_s = torch.clamp(t_end - t_arrival_min, min=0.0)
        return t_arrival_min, post_arrival_duration_s

    def _dilate_boolean_mask(self, mask: torch.Tensor, padding_samples: int) -> torch.Tensor:
        if padding_samples <= 0:
            return mask
        kernel_size = 2 * int(padding_samples) + 1
        mask_float = mask.to(dtype=torch.float32).unsqueeze(1)
        dilated = F.max_pool1d(mask_float, kernel_size=kernel_size, stride=1, padding=int(padding_samples))
        return dilated.squeeze(1) > 0.0

    def _windowed_mean_per_trace(self, values: torch.Tensor, window_mask: torch.Tensor) -> torch.Tensor:
        if values.shape != window_mask.shape:
            raise ValueError(f"values and window_mask must have the same shape, got {values.shape} and {window_mask.shape}.")
        mask_float = window_mask.to(device=values.device, dtype=values.dtype)
        counts = mask_float.sum(dim=1).clamp_min(1.0)
        return (values * mask_float).sum(dim=1) / counts

    def _as_trace_batch(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if tensor.ndim == 3 and tensor.shape[1] == 1:
            tensor = tensor[:, 0, :]
        if tensor.ndim != 2:
            raise ValueError(f"{name} must have shape (batch, n_time), got {tensor.shape}.")
        return tensor

    def _metadata_to_batch_tensor(self, value: Any, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=device, dtype=dtype)
        else:
            tensor = torch.as_tensor(value, device=device, dtype=dtype)
        return self._ensure_batch_vector(tensor, batch_size, device, dtype, name="metadata value")

    def _ensure_batch_vector(self, value: Any, batch_size: int, device: torch.device, dtype: torch.dtype, name: str) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=device, dtype=dtype)
        else:
            tensor = torch.as_tensor(value, device=device, dtype=dtype)
        if tensor.ndim == 0:
            tensor = tensor.reshape(1).expand(batch_size)
        else:
            tensor = tensor.reshape(-1)
        if tensor.numel() == 1:
            tensor = tensor.expand(batch_size)
        if tensor.numel() != batch_size:
            raise ValueError(f"{name} must contain either 1 value or batch_size={batch_size} values, got {tensor.numel()} values.")
        return tensor

    def _masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if values.ndim != 1:
            raise ValueError(f"values must be one-dimensional, got {values.shape}.")
        if mask.ndim != 1:
            raise ValueError(f"mask must be one-dimensional, got {mask.shape}.")
        if values.shape[0] != mask.shape[0]:
            raise ValueError("values and mask must have the same length.")
        mask_float = mask.to(device=values.device, dtype=values.dtype)
        denominator = mask_float.sum().clamp_min(1.0)
        return (values * mask_float).sum() / denominator


# Backward-compatible alias if you import the class without the _v2 suffix.
MaskedPseudoPhysicsInformedTraceLoss = MaskedPseudoPhysicsInformedTraceLoss_v2


if __name__ == "__main__":
    torch.manual_seed(123)
    batch_size = 6
    n_time = 1500
    nz = 32
    nx = 32
    y_true = torch.zeros(batch_size, n_time)
    y_true[1, 1200:1220] = 0.5 * torch.sin(torch.linspace(0, 3.14, 20))
    y_true[2:, 250:650] = torch.randn(batch_size - 2, 400)
    y_pred = 0.05 * torch.randn(batch_size, n_time, requires_grad=True)
    velocity_model = 1500.0 + 2500.0 * torch.rand(batch_size, 1, nz, nx)
    metadata = {
        "dt": torch.full((batch_size,), 0.0006683375),
        "dx": torch.full((batch_size,), 10.0),
        "dz": torch.full((batch_size,), 10.0),
        "receiver_x": torch.tensor([20, 100, 200, 300, 350, 380]),
        "receiver_z": torch.tensor([10, 10, 10, 10, 10, 10]),
        "source_x": torch.tensor([200, 200, 200, 200, 200, 200]),
        "source_z": torch.tensor([100, 100, 100, 100, 100, 100]),
    }
    criterion = MaskedPseudoPhysicsInformedTraceLoss_v2(alpha_frequency=0.0, alpha_arrival=0.0)
    loss, terms = criterion.forward_from_metadata(y_pred, y_true, velocity_model, metadata)
    loss.backward()
    print("Smoke-test loss:", float(loss.detach().cpu()))
    print("Active fraction:", float(terms["active_fraction"].cpu()))
    print("Partial fraction:", float(terms["partial_fraction"].cpu()))
    print("Inactive fraction:", float(terms["inactive_fraction"].cpu()))
    print("Significant duration mean [s]:", float(terms["significant_duration_mean_s"].cpu()))
