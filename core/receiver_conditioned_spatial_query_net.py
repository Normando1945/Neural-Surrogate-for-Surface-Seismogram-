"""
Receiver-conditioned spatial-query neural network.

This module intentionally lives outside core_wp_2d_simul.py.  It provides a
stronger replacement for the global-latent receiver-conditioned network while
keeping the same learning target:

    velocity_model + receiver_coordinates -> one seismogram trace
"""

import numpy as np


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR = None
except Exception as exc:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = exc


if TORCH_AVAILABLE:

    class _FourierCoordinateEncoding(nn.Module):
        """Encode normalized coordinates with Fourier features."""

        def __init__(self, in_dim=2, num_bands=16, max_frequency=16.0, include_input=True):
            super().__init__()

            self.in_dim = int(in_dim)
            self.num_bands = int(num_bands)
            self.max_frequency = float(max_frequency)
            self.include_input = bool(include_input)

            if self.in_dim <= 0:
                raise ValueError("in_dim must be positive.")
            if self.num_bands <= 0:
                raise ValueError("num_bands must be positive.")
            if self.max_frequency <= 0.0:
                raise ValueError("max_frequency must be positive.")

            frequencies = torch.logspace(
                start=0.0,
                end=np.log10(self.max_frequency),
                steps=self.num_bands,
                dtype=torch.float32,
            )
            self.register_buffer("frequencies", frequencies, persistent=True)

            base_dim = self.in_dim if self.include_input else 0
            trig_dim = 2 * self.in_dim * self.num_bands
            self.out_dim = int(base_dim + trig_dim)

        def forward(self, coords):
            if coords.ndim != 2:
                raise ValueError(f"coords must have shape (batch, {self.in_dim}), got {coords.shape}")
            if coords.shape[1] != self.in_dim:
                raise ValueError(f"coords must have shape (batch, {self.in_dim}), got {coords.shape}")

            pieces = []
            if self.include_input:
                pieces.append(coords)

            coords_expanded = coords.unsqueeze(-1)
            freq_view = self.frequencies.view(1, 1, -1)
            angles = 2.0 * np.pi * coords_expanded * freq_view

            pieces.append(torch.sin(angles).reshape(coords.shape[0], -1))
            pieces.append(torch.cos(angles).reshape(coords.shape[0], -1))

            return torch.cat(pieces, dim=1)


    class _ConvNormAct(nn.Module):
        """Small convolutional block with GroupNorm and GELU."""

        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
            super().__init__()
            padding = kernel_size // 2
            groups = min(8, int(out_channels))

            self.block = nn.Sequential(
                nn.Conv2d(
                    int(in_channels),
                    int(out_channels),
                    kernel_size=int(kernel_size),
                    stride=int(stride),
                    padding=int(padding),
                    bias=False,
                ),
                nn.GroupNorm(groups, int(out_channels)),
                nn.GELU(),
            )

        def forward(self, x):
            return self.block(x)


    class _ResidualConvBlock(nn.Module):
        """Residual spatial block used after downsampling."""

        def __init__(self, channels, dropout=0.0):
            super().__init__()
            channels = int(channels)
            dropout = float(dropout)

            self.conv1 = _ConvNormAct(channels, channels, kernel_size=3, stride=1)
            self.conv2 = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(min(8, channels), channels),
            )
            self.act = nn.GELU()
            self.drop = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        def forward(self, x):
            residual = x
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.drop(x)
            return self.act(residual + x)


    class _MLPResidualBlock(nn.Module):
        """Residual MLP block for fused latent features."""

        def __init__(self, width, dropout=0.05):
            super().__init__()
            width = int(width)
            dropout = float(dropout)

            self.norm = nn.LayerNorm(width)
            self.fc1 = nn.Linear(width, width)
            self.act = nn.GELU()
            self.drop1 = nn.Dropout(dropout)
            self.fc2 = nn.Linear(width, width)
            self.drop2 = nn.Dropout(dropout)

        def forward(self, x):
            residual = x
            x = self.norm(x)
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop1(x)
            x = self.fc2(x)
            x = self.drop2(x)
            return residual + x


    class ReceiverConditionedSpatialQuerySeismogramNet(nn.Module):
        """
        Predict one seismogram trace from a velocity model and receiver query.

        Difference from the older global-latent model
        ---------------------------------------------
        The older receiver-conditioned network compresses the whole velocity
        model into one global vector before it sees the receiver coordinate.
        This class keeps a spatial feature map and samples that map at:

        - the receiver coordinate,
        - the source coordinate,
        - the midpoint between source and receiver.

        These spatial-query features are fused with global context and Fourier
        receiver features before decoding the output trace.
        """

        def __init__(
            self,
            n_time=1500,
            coord_dim=2,
            coord_num_bands=16,
            coord_max_frequency=16.0,
            model_latent_dim=512,
            coord_latent_dim=256,
            fusion_dim=1024,
            decoder_hidden_dim=2048,
            dropout=0.05,
            spatial_channels=256,
            source_coords=(0.5, 0.25),
            use_midpoint_query=True,
        ):
            super().__init__()

            self.n_time = int(n_time)
            self.coord_dim = int(coord_dim)
            self.coord_num_bands = int(coord_num_bands)
            self.coord_max_frequency = float(coord_max_frequency)
            self.model_latent_dim = int(model_latent_dim)
            self.coord_latent_dim = int(coord_latent_dim)
            self.fusion_dim = int(fusion_dim)
            self.decoder_hidden_dim = int(decoder_hidden_dim)
            self.dropout = float(dropout)
            self.spatial_channels = int(spatial_channels)
            self.use_midpoint_query = bool(use_midpoint_query)

            if self.n_time <= 0:
                raise ValueError("n_time must be positive.")
            if self.coord_dim != 2:
                raise ValueError("coord_dim must be 2 for (x, z) spatial queries.")
            if self.model_latent_dim <= 0:
                raise ValueError("model_latent_dim must be positive.")
            if self.coord_latent_dim <= 0:
                raise ValueError("coord_latent_dim must be positive.")
            if self.fusion_dim <= 0:
                raise ValueError("fusion_dim must be positive.")
            if self.decoder_hidden_dim <= 0:
                raise ValueError("decoder_hidden_dim must be positive.")
            if self.spatial_channels <= 0:
                raise ValueError("spatial_channels must be positive.")
            if self.dropout < 0.0 or self.dropout >= 1.0:
                raise ValueError("dropout must be in the range [0, 1).")

            if source_coords is None:
                self.source_coords = None
            else:
                source_coords = tuple(float(v) for v in source_coords)
                if len(source_coords) != 2:
                    raise ValueError("source_coords must contain two normalized values: (x, z).")
                self.source_coords = source_coords

            self.receiver_fourier_encoder = _FourierCoordinateEncoding(
                in_dim=self.coord_dim,
                num_bands=self.coord_num_bands,
                max_frequency=self.coord_max_frequency,
                include_input=True,
            )

            self.receiver_mlp = nn.Sequential(
                nn.Linear(self.receiver_fourier_encoder.out_dim, 128),
                nn.GELU(),
                nn.LayerNorm(128),
                nn.Dropout(self.dropout),
                nn.Linear(128, self.coord_latent_dim),
                nn.GELU(),
                nn.LayerNorm(self.coord_latent_dim),
            )

            self.spatial_encoder = nn.Sequential(
                _ConvNormAct(1, 32, kernel_size=5, stride=2),
                _ConvNormAct(32, 64, kernel_size=5, stride=2),
                _ResidualConvBlock(64, dropout=self.dropout * 0.5),
                _ConvNormAct(64, 128, kernel_size=3, stride=2),
                _ResidualConvBlock(128, dropout=self.dropout * 0.5),
                _ConvNormAct(128, self.spatial_channels, kernel_size=3, stride=2),
                _ResidualConvBlock(self.spatial_channels, dropout=self.dropout * 0.5),
                _ResidualConvBlock(self.spatial_channels, dropout=self.dropout * 0.5),
            )

            query_count = 1
            if self.source_coords is not None:
                query_count += 1
                if self.use_midpoint_query:
                    query_count += 1

            local_feature_dim = self.spatial_channels * query_count

            self.global_head = nn.Sequential(
                nn.Linear(self.spatial_channels, self.model_latent_dim),
                nn.GELU(),
                nn.LayerNorm(self.model_latent_dim),
                nn.Dropout(self.dropout),
            )

            self.local_head = nn.Sequential(
                nn.Linear(local_feature_dim, self.model_latent_dim),
                nn.GELU(),
                nn.LayerNorm(self.model_latent_dim),
                nn.Dropout(self.dropout),
            )

            self.film_gamma = nn.Linear(self.coord_latent_dim, self.model_latent_dim)
            self.film_beta = nn.Linear(self.coord_latent_dim, self.model_latent_dim)

            fusion_input_dim = 2 * self.model_latent_dim + self.coord_latent_dim

            self.pre_fusion = nn.Sequential(
                nn.Linear(fusion_input_dim, self.fusion_dim),
                nn.GELU(),
                nn.LayerNorm(self.fusion_dim),
                nn.Dropout(self.dropout),
            )

            self.fusion_block_1 = _MLPResidualBlock(self.fusion_dim, dropout=self.dropout)
            self.fusion_block_2 = _MLPResidualBlock(self.fusion_dim, dropout=self.dropout)

            self.trace_decoder = nn.Sequential(
                nn.Linear(self.fusion_dim, self.decoder_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(self.decoder_hidden_dim),
                nn.Dropout(self.dropout),
                nn.Linear(self.decoder_hidden_dim, self.decoder_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(self.decoder_hidden_dim),
                nn.Dropout(self.dropout),
                nn.Linear(self.decoder_hidden_dim, self.n_time),
            )

        def _grid_from_coords(self, coords):
            coords = torch.clamp(coords, 0.0, 1.0)
            grid = coords * 2.0 - 1.0
            return grid.view(coords.shape[0], 1, 1, 2)

        def _sample_spatial_features(self, feature_map, coords):
            grid = self._grid_from_coords(coords)
            sampled = F.grid_sample(
                feature_map,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            return sampled[:, :, 0, 0]

        def _source_coord_batch(self, receiver_coords):
            if self.source_coords is None:
                return None
            source = receiver_coords.new_tensor(self.source_coords)
            return source.view(1, 2).expand(receiver_coords.shape[0], -1)

        def forward(self, x_model, receiver_coords):
            if x_model.ndim != 4:
                raise ValueError(f"x_model must have shape (batch, 1, nz, nx), got {x_model.shape}")
            if x_model.shape[1] != 1:
                raise ValueError(f"x_model must have shape (batch, 1, nz, nx), got {x_model.shape}")
            if receiver_coords.ndim != 2:
                raise ValueError(f"receiver_coords must have shape (batch, 2), got {receiver_coords.shape}")
            if receiver_coords.shape[1] != 2:
                raise ValueError(f"receiver_coords must have shape (batch, 2), got {receiver_coords.shape}")

            feature_map = self.spatial_encoder(x_model)
            global_feature = torch.mean(feature_map, dim=(2, 3))

            local_features = [self._sample_spatial_features(feature_map, receiver_coords)]

            source_coords = self._source_coord_batch(receiver_coords)
            if source_coords is not None:
                local_features.append(self._sample_spatial_features(feature_map, source_coords))

                if self.use_midpoint_query:
                    midpoint_coords = 0.5 * (receiver_coords + source_coords)
                    local_features.append(self._sample_spatial_features(feature_map, midpoint_coords))

            local_feature = torch.cat(local_features, dim=1)

            receiver_encoded = self.receiver_fourier_encoder(receiver_coords)
            receiver_latent = self.receiver_mlp(receiver_encoded)

            global_latent = self.global_head(global_feature)
            local_latent = self.local_head(local_feature)

            gamma = self.film_gamma(receiver_latent)
            beta = self.film_beta(receiver_latent)

            conditioned_global = global_latent * (1.0 + gamma) + beta
            conditioned_local = local_latent * (1.0 + gamma) + beta

            fused_latent = torch.cat(
                [conditioned_global, conditioned_local, receiver_latent],
                dim=1,
            )
            fused_latent = self.pre_fusion(fused_latent)
            fused_latent = self.fusion_block_1(fused_latent)
            fused_latent = self.fusion_block_2(fused_latent)

            return self.trace_decoder(fused_latent)


    class EnergyWeightedMSELoss(nn.Module):
        """
        MSE loss that gives more weight to energetic parts of the target trace.

        This keeps the loss compatible with y_pred/y_true only, so it can replace
        nn.MSELoss in the existing training loop without needing dt_batch.
        """

        def __init__(self, base_weight=1.0, energy_weight=5.0, normalize_weights=True, eps=1e-6):
            super().__init__()

            self.base_weight = float(base_weight)
            self.energy_weight = float(energy_weight)
            self.normalize_weights = bool(normalize_weights)
            self.eps = float(eps)

            if self.base_weight < 0.0:
                raise ValueError("base_weight must be non-negative.")
            if self.energy_weight < 0.0:
                raise ValueError("energy_weight must be non-negative.")
            if self.eps <= 0.0:
                raise ValueError("eps must be positive.")

        def forward(self, y_pred, y_true):
            abs_true = torch.abs(y_true)
            peak = torch.amax(abs_true, dim=-1, keepdim=True).clamp_min(self.eps)
            weights = self.base_weight + self.energy_weight * (abs_true / peak)

            if self.normalize_weights:
                weights = weights / weights.mean(dim=-1, keepdim=True).clamp_min(self.eps)

            return torch.mean(weights * (y_pred - y_true) ** 2)

else:

    class ReceiverConditionedSpatialQuerySeismogramNet:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"PyTorch is not available in this environment. Original error: {TORCH_IMPORT_ERROR}"
            )


    class EnergyWeightedMSELoss:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"PyTorch is not available in this environment. Original error: {TORCH_IMPORT_ERROR}"
            )
