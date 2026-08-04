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


.PY FOR VALIDATING RECIVER CONDITIONED CNN (FOURIER ONET) RESULTS WIDTH SHUFFLED COORDINATES.
"""
#---------------- Basic Libraries ----------------
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import time as tm

#---------------- Import all classes ----------------
from core import *

#---------------- Route of file ----------------
project_root = Path(__file__).resolve().parent.parent
#---------------- Checkpoints directory ----------------
checkpoints_dir = project_root / "checkpoints"                                                  # Define the checkpoints directory path

#---------------- Safe Folder ----------------
Safefolder = "0. Test"

#---------------- CheckPoint Folder ----------------
Checkfolder = "Test"

#---------------- Network folder -------------------
Network = project_root /'Network_Arq'/'NEARQ_TEST_8_batch_90_receptors_3 epoch'

#---------------- Active plots ---------------------
loss_dashboard = 1
review_metric_summary = 1
review_prediction = 1
spectrogram = 1
radar = 1
correlation = 1
trace_plot = 1


#---------------- Define Loss path ----------------
loss_dir = checkpoints_dir /Checkfolder/"receiver_conditioned_cnn_loss_history.npz"
print("\n" + "=" * 120)
print("Loss history path:")
print(loss_dir)
if not loss_dir.exists():
    raise FileNotFoundError(f"Loss history file not found: {loss_dir}")
print("=" * 120)
print("\n")

#---------------- Define paths for receiver-conditioned model review ----------------
best_model_path = checkpoints_dir /Checkfolder/"best_receiver_conditioned_cnn.pth"              # Define best trained receiver-conditioned checkpoint path
h5_path = project_root / "data" / "raw" / "dataset_surface_seismograms.h5"                      # Define HDF5 dataset path

#---------------- Define path for .xlsx Network Arch --------------------------------
xlsx_path = Network / "ai_surface_parameter_count_layer_order_detail.xlsx"

print("\n" + "=" * 120)
print("Best receiver-conditioned model path:")
print(best_model_path)
print("\nHDF5 file path:")
print(h5_path)
if not best_model_path.exists():
    raise FileNotFoundError(f"Best model file not found: {best_model_path}")

if not h5_path.exists():
    raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
print("=" * 120)
print("\n")

#---------------- Define common results directory ----------------
results_dir = project_root / "outputs" / "Results_Conditioned_Recivers_FourierOnet" / Safefolder  # Define dedicated results directory for this review workflow
results_dir.mkdir(parents=True, exist_ok=True)                                                    # Create directory if it does not exist

#---------------- Recivers to review ----------------
recivers = [0, 19, 29, 49, 59]
sample_to_view = 1                                                                              # Validation sample index to review in detail
recivers_trace = [9, 19, 39, 59]                                                                  # Trace indices to review 
sample_trace = 9                                                                                  # Validation sample index for trace plotting 


########################################################################################################################
########################################################################################################################
###################################### checkpoint-normalized reviewer ##################################################
########################################################################################################################
########################################################################################################################

class ReceiverConditionedShuffledCoordsReviewerWithCheckpointNormalization(ReceiverConditionedShuffledCoordsReviewer):
    """
    Local reviewer that preserves the original 5.2 workflow but applies the same
    velocity-model normalization used during training when the checkpoint says so.

    Why this is necessary
    ---------------------
    The dataset must keep normalize_x=False so that the velocity model remains in
    physical units for plotting and interpretation. However, the trained 3.3 model
    receives a normalized velocity model inside the neural network:

        x_model_input = (x_batch - velocity_mean_for_model) / velocity_std_for_model

    Therefore, during review, the model input must be normalized before inference
    if the checkpoint stores normalize_model_input=True.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.normalize_model_input = bool(self.checkpoint.get("normalize_model_input", False))
        self.velocity_mean_for_model = float(self.checkpoint.get("velocity_mean_for_model", 3000.0))
        self.velocity_std_for_model = float(self.checkpoint.get("velocity_std_for_model", 1500.0))

        print("\n" + "=" * 120)
        print("Checkpoint velocity-input normalization used by reviewer")
        print(f"normalize_model_input   : {self.normalize_model_input}")
        print(f"velocity_mean_for_model : {self.velocity_mean_for_model}")
        print(f"velocity_std_for_model  : {self.velocity_std_for_model}")
        print("=" * 120)
        print("\n")

    def _normalize_velocity_for_model(self, x_batch):
        """
        Normalize the velocity model only for the neural network.

        The unnormalized velocity model is still used for plotting because the
        HDF5 dataset is loaded with normalize_x=False.
        """
        if not self.normalize_model_input:
            return x_batch

        return (x_batch - self.velocity_mean_for_model) / self.velocity_std_for_model

    def _predict_full_gather(self, x_true_tensor, receiver_coords_np):
        """
        Predict a full receiver gather while respecting checkpoint normalization.
        """
        n_receivers = int(receiver_coords_np.shape[0])                                           # Read number of receiver queries

        x_batch_physical = x_true_tensor.unsqueeze(0).repeat(n_receivers, 1, 1, 1).to(self.device) # Repeat physical velocity model
        x_batch_model = self._normalize_velocity_for_model(x_batch_physical)                     # Normalize only for the CNN encoder

        receiver_coord_batch = torch.from_numpy(receiver_coords_np).float().to(self.device)      # Convert receiver coordinates to tensor

        with torch.no_grad():                                                                    # Disable gradients during inference
            y_pred = self.model(
                x_batch_model,                                                                   # Normalized or physical input depending on checkpoint
                receiver_coord_batch,
            )

        return y_pred.detach().cpu().numpy()                                                     # Return predicted gather with shape (n_receivers, n_time)

    def run_all(
        self,
        sample_index=0,
        shuffle_seed=42,
        receivers_to_plot_traces=None,
        save=True,
        plot=1,
        plot_metric_summary=1,
        plot_prediction_review=1,
    ):
        result_dict = self.evaluate_sample(                                                       # Evaluate one sample using correct and shuffled coordinates
            sample_index=sample_index,
            shuffle_seed=shuffle_seed,
        )

        if plot == 1 and plot_metric_summary == 1:
            self.plot_metric_summary(                                                             # Plot metric summary only when enabled
                result_dict=result_dict,
                save=save,
            )

        if plot == 1 and plot_prediction_review == 1:
            self.plot_prediction_review(                                                          # Plot trace review only when enabled
                result_dict=result_dict,
                receivers_to_plot=receivers_to_plot_traces,
                save=save,
            )

        return result_dict, self.last_metrics_df, self.last_overall_summary_df                    # Return reviewed outputs


########################################################################################################################
########################################################################################################################
############################################# loss-history reviewer  ###################################################
########################################################################################################################
########################################################################################################################

loss_reviewer = ReceiverConditionedLossHistoryReviewer(                                           # Create receiver-conditioned loss-history reviewer
    loss_history_path=loss_dir,                                                                   # Pass NPZ history path
    outputs_dir=project_root / "outputs" / "Results_Conditioned_Recivers_FourierOnet" / Safefolder,            # Pass outputs directory
)
if loss_dashboard == 1:
    loss_reviewer.run_all()                                                                       # Execute the full loss-history review


########################################################################################################################
########################################################################################################################
################################################## model review  #######################################################
########################################################################################################################
########################################################################################################################

recivers_to_plot_traces = recivers                                                               # Receiver indices to plot in the trace review figure

#---------------- Create reviewer ----------------
conditioned_reviewer = ReceiverConditionedShuffledCoordsReviewerWithCheckpointNormalization(       # Create reviewer with checkpoint-aware velocity normalization
    h5_path=h5_path,                                                                              # Pass HDF5 dataset path
    best_model_path=best_model_path,                                                              # Pass best trained checkpoint path
    split="val",                                                                                  # Review validation split
    normalize_x=False,                                                                            # Keep raw velocity models for plotting; reviewer normalizes model input if checkpoint requires it
    normalize_y=False,                                                                            # Keep raw traces
    device=None,                                                                                  # Use CUDA automatically if available
    outputs_dir=results_dir,                                                                      # Save figures in the dedicated results folder
)

#---------------- Run reviewer ----------------
result_dict, metrics_df, overall_summary_df = conditioned_reviewer.run_all(                       # Execute full model review
    sample_index=sample_to_view,                                                                  # Validation sample index to review
    shuffle_seed=42,                                                                              # Reproducible shuffle seed
    receivers_to_plot_traces=recivers_to_plot_traces,                                             # Receivers to plot in the trace-review figure
    save=True,                                                                                    # Save generated figures
    plot_metric_summary=review_metric_summary,                                                     # Enable/disable class metric-summary plot
    plot_prediction_review=review_prediction,                                                       # Enable/disable class prediction-review plot
)

print('='*120)
print('Overall summary')
print(metrics_df.columns)
print(metrics_df.head())
print(metrics_df)
print('='*120)


########################################################################################################################
########################################################################################################################
############################################ spectrogram comparator ####################################################
########################################################################################################################
########################################################################################################################

spectrogram_comparator = ReceiverConditionedCNNSpectrogramComparator(                             # Create spectrogram comparator from the already computed review results
    result_dict=result_dict,                                                                      # Reuse the result_dict returned by conditioned_reviewer.run_all()
    NFFT=128,                                                                                     # Spectrogram window length
    noverlap=96,                                                                                  # Number of overlapping samples
    cmap_main="viridis",                                                                          # Colormap for simulated and predicted spectrograms
    cmap_diff="magma",                                                                            # Colormap for absolute dB-difference spectrograms
    db_floor=-120.0,                                                                              # Minimum dB floor
    outputs_dir=results_dir,                                                                      # Save figures in the same review folder
)

spectrogram_figure_path = results_dir / f"4.spectrogram_sample_{sample_to_view}.svg"              # Use a short file name to avoid Windows path-length issues

spectrogram_result_dict = spectrogram_comparator.plot_spectrogram_comparison(                     # Generate spectrogram comparison figure
    receivers_to_plot= recivers,                                                                  # Receivers to include in the spectrogram review
    fmax=None,                                                                                    # Use the full frequency range
    common_color_scale=True,                                                                      # Use one common color scale for simulated/predicted panels
    common_diff_scale=True,                                                                       # Use one common color scale for difference panels
    save=True,                                                                                    # Save the generated figure
    figure_path=spectrogram_figure_path,                                                          # Pass short output path explicitly
    plot = spectrogram
)

print(result_dict.keys())
print("y_true shape:", result_dict["y_true"].shape)
print("y_pred_correct shape:", result_dict["y_pred_correct"].shape)
print("y_pred_shuffled shape:", result_dict["y_pred_shuffled"].shape)

########################################################################################################################
########################################################################################################################
################################# Radar View Correlation, RSME, Energy Error ###########################################
########################################################################################################################
########################################################################################################################

selected_receivers = [9, 19, 29, 39, 49, 59, 69, 79, 89]
text_size = 18

df_plot = metrics_df[
    metrics_df["receiver_id"].isin(selected_receivers)
].copy()

labels = [f"R{int(r + 1)}" for r in df_plot["receiver_id"]]

metrics_to_plot = {
    "Correlation": (
        df_plot["correlation_correct"].values,
        df_plot["correlation_shuffled"].values,
    ),
    "RMSE [%]": (
        df_plot["rmse_correct"].values,
        df_plot["rmse_shuffled"].values,
    ),
    "Energy error [%]": (
        df_plot["total_energy_relative_error_correct"].values,
        df_plot["total_energy_relative_error_shuffled"].values,
    ),
    "Arrival-time error [s]": (
        df_plot["peak_arrival_time_abs_error_s_correct"].values,
        df_plot["peak_arrival_time_abs_error_s_shuffled"].values,
    ),
}

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
angles = np.concatenate([angles, angles[:1]])

if radar == 1:
    fig = plt.figure(figsize=(10, 10))
    # fig.suptitle('Receiver-wise TIER 4 Performance Summary', fontsize=text_size + 1, color=(0,0,1))
    for i, (title, (values_correct, values_shuffled)) in enumerate(metrics_to_plot.items(), start=1):
        values_correct = np.asarray(values_correct, dtype=float)
        values_shuffled = np.asarray(values_shuffled, dtype=float)
        values_correct = np.concatenate([values_correct, values_correct[:1]])
        values_shuffled = np.concatenate([values_shuffled, values_shuffled[:1]])
        ax = fig.add_subplot(2, 2, i, polar=True)
        ax.plot(angles,values_correct, ls='-',lw=1.0,marker='o',markeredgecolor=(0,0,1),markerfacecolor=(0,0,1),markersize=5,color=(0,0,1),label='Correct')
        ax.fill(angles,values_correct,color=(0,0,1),alpha=0.15)

        ax.plot(
            angles,
            values_shuffled,
            ls='--',
            lw=1.0,
            marker='s',
            markeredgecolor=(0.4,0.4,0.4),
            markerfacecolor=(1,1,1),
            markersize=5,
            color=(0.4,0.4,0.4),
            label='Shuffled'
        )

        ax.fill(
            angles,
            values_shuffled,
            color=(0.4,0.4,0.4),
            alpha=0.08
        )

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=text_size)
        ax.tick_params(axis="y", labelsize=text_size)
        ax.set_title(title, color=(0,0,1), fontsize=text_size + 1)

        if i == 1:
            ax.legend(
                loc='upper right',
                bbox_to_anchor=(1.35, 1.15),
                fontsize=text_size - 2,
                frameon=False
            )

    fig.tight_layout()

    fig_path = results_dir / f"5.Receiverwise_radar_metrics_correct_vs_shuffled_sample_{sample_to_view}.svg"
    fig_path_pdf = results_dir / f"fig10.pdf"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(fig_path_pdf, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.show()

    print(f"Saved radar figure to: {fig_path}")

else:
    pass


########################################################################################################################
########################################################################################################################
############################### True-vs-predicted receiver similarity matrices #########################################
########################################################################################################################
########################################################################################################################

text_size = 18
cmap_name = "Blues"   # Options: "viridis", "magma", "plasma", "Spectral", "coolwarm", "cividis"

y_true = result_dict["y_true"]
y_pred_correct = result_dict["y_pred_correct"]
y_pred_shuffled = result_dict["y_pred_shuffled"]


def cross_receiver_correlation_matrix(y_ref, y_pred):
    # y_ref = np.asarray(y_ref, dtype=float)
    # y_pred = np.asarray(y_pred, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)[:45]
    y_pred = np.asarray(y_pred, dtype=float)[:45]

    n_ref = y_ref.shape[0]
    n_pred = y_pred.shape[0]

    corr_matrix = np.zeros((n_ref, n_pred), dtype=float)

    for i in range(n_ref):
        for j in range(n_pred):
            corr_matrix[i, j] = np.corrcoef(y_ref[i], y_pred[j])[0, 1]
    return corr_matrix

corr_true_vs_correct = cross_receiver_correlation_matrix(y_true,y_pred_correct)
corr_true_vs_shuffled = cross_receiver_correlation_matrix(y_true,y_pred_shuffled)

if correlation == 1:
    fig, axes = plt.subplots(2,1,figsize=(6, 10))
    plots = [(corr_true_vs_correct, "Waveform Correlation Matrix: Ground Truth vs. Correct Query"),(corr_true_vs_shuffled, "Waveform Correlation Matrix: Ground Truth vs. Shuffled Query")]
    for ax, (matrix, title) in zip(axes, plots):
        im = ax.imshow(matrix,cmap=cmap_name,vmin=-1,vmax=1,aspect="equal", origin="upper")
        ax.set_title(title, fontsize=text_size + 1,color=(0, 0, 1))
        ax.set_xlabel("Predicted receiver index",fontsize=text_size)
        ax.set_ylabel("Ground Truth receiver index",fontsize=text_size)
        ax.tick_params(axis="both",labelsize=text_size)
        cbar = fig.colorbar(im, ax=ax,fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=text_size)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    fig.tight_layout()
    fig_path = results_dir / f"6.Waveform Correlation Matrix_ReceiverIndex_sample_{sample_to_view}.svg"
    fig_path_pdf = results_dir / f"fig11.pdf"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight",pad_inches=0.02)
    plt.savefig(fig_path_pdf, dpi=200, bbox_inches="tight",pad_inches=0.02)
    plt.show()
    print(f"Saved figure to: {fig_path}")
else:
    pass


########################################################################################################################
########################################################################################################################
################################################# Trace Plot ###########################################################
########################################################################################################################
########################################################################################################################

text_size = 20

trace_sample_index = sample_trace
trace_receivers_to_plot = recivers_trace

if trace_plot == 1:
    with h5py.File(h5_path, "r") as h5f:
        n_h5_samples = int(h5f["outputs/surface_seismograms"].shape[0])
        n_h5_receivers = int(h5f["outputs/surface_seismograms"].shape[1])

        trace_gather = h5f["outputs/surface_seismograms"][trace_sample_index].astype(np.float32)
        dt_dataset = h5f["metadata/dt"]
        receiver_x = h5f["metadata/receiver_x"][...].astype(int)
        receiver_z = h5f["metadata/receiver_z"][...].astype(int)

    step_axis = np.arange(trace_gather.shape[1], dtype=int)
    selected_traces = trace_gather[trace_receivers_to_plot]
    trace_scale = float(np.percentile(np.abs(selected_traces), 99.0))
    if trace_scale < 1e-20:
        trace_scale = 1.0

    trace_offsets = np.arange(len(trace_receivers_to_plot), dtype=float)
    trace_spacing = 1.0
    wiggle_gain = 0.42 * trace_spacing / trace_scale
    
    print("\n" + "="*120)
    print(f'Number of Samples = {n_h5_samples}')
    print(f'Number of Recivers in each Sample = {n_h5_receivers}')
    print(f'Plot Trace = {trace_sample_index }')
    print(f'Number of Calculation Steps = {len(step_axis)}')
    print(f'Steps for plot = {step_axis}')
    print("="*120)

    fig, ax = plt.subplots(figsize=(5.2, 10.0))

    for local_id, receiver_id in enumerate(trace_receivers_to_plot):
        normalized_trace = trace_gather[receiver_id] * wiggle_gain
        x_trace = trace_offsets[local_id] + normalized_trace
        ax.plot(x_trace,step_axis,color=(0,0,0),linewidth=1.0)
        # ax.axvline(trace_offsets[local_id],color=(0.5,0.5,0.5),linewidth=0.65,alpha=0.7)

    ax.set_title(f"Seismogram | sample {trace_sample_index + 1}",fontsize=text_size, color=(0,0,1))
    ax.set_xticklabels([f"R{receiver_id + 1}" for receiver_id in trace_receivers_to_plot],fontsize= text_size)
    ax.set_ylabel("Calculation step",fontsize=text_size)
    ax.set_xlabel("Receiver",fontsize=text_size)
    ax.tick_params(axis="both",labelsize=text_size)
    
    ax.set_xlim(trace_offsets[0] - 0.8,trace_offsets[-1] + 0.8)
    ax.set_ylim(step_axis[-1],step_axis[0])
    ax.set_xticks(trace_offsets)
    ax.grid(True,axis="y")
    fig.tight_layout()

    fig_path = results_dir / f"7.Trace_plot_sample_{trace_sample_index + 1}.svg"
    plt.savefig(fig_path,dpi=200,bbox_inches="tight",pad_inches=0.02)
    plt.show()
    print(f"Saved HDF5 trace plot to: {fig_path}")
else:
    pass


########################################################################################################################
################################ Trainable parameter density matrix ####################################################
########################################################################################################################
########################################################################################################################
text_size = 18

import warnings

# These warnings are not critical. They only mean that Excel-specific formatting is ignored by pandas/openpyxl.
warnings.filterwarnings("ignore", message="Unknown extension is not supported")
warnings.filterwarnings("ignore", message="Conditional Formatting extension is not supported")


detail_df = pd.read_excel( xlsx_path, sheet_name="Layer-by-Layer Order", header=3, engine="openpyxl")
block_df = pd.read_excel( xlsx_path, sheet_name="Block Subtotals",  header=2, engine="openpyxl")
# Clean column names
detail_df.columns = detail_df.columns.astype(str).str.strip()
block_df.columns = block_df.columns.astype(str).str.strip()
print("\nDetail sheet columns:")
print(detail_df.columns)
print("\nBlock subtotal sheet columns:")
print(block_df.columns)
# Validate required columns before continuing
required_detail_cols = [
    "Order",
    "Stage",
    "Branch / Flow",
    "Module / Layer",
    "Operation",
    "Total Params",
    "Parameters (M)"
]
required_block_cols = [
    "Order",
    "Functional Block",
    "Parameters",
    "Parameters (M)",
    "Share of Total"
]
missing_detail_cols = [col for col in required_detail_cols if col not in detail_df.columns]
missing_block_cols = [col for col in required_block_cols if col not in block_df.columns]
if len(missing_detail_cols) > 0:
    raise ValueError(f"Missing columns in 'Layer-by-Layer Order': {missing_detail_cols}")
if len(missing_block_cols) > 0:
    raise ValueError(f"Missing columns in 'Block Subtotals': {missing_block_cols}")
# Clean empty rows
detail_df = detail_df.dropna(subset=["Order"]).copy()
block_df = block_df.dropna(subset=["Order"]).copy()
# Numeric conversion
detail_df["Order"] = pd.to_numeric(detail_df["Order"], errors="coerce").astype(int)
detail_df["Total Params"] = pd.to_numeric(detail_df["Total Params"], errors="coerce").fillna(0)
detail_df["Parameters (M)"] = pd.to_numeric(detail_df["Parameters (M)"], errors="coerce").fillna(0)
block_df["Order"] = pd.to_numeric(block_df["Order"], errors="coerce").astype(int)
block_df["Parameters"] = pd.to_numeric(block_df["Parameters"], errors="coerce").fillna(0)
block_df["Parameters (M)"] = pd.to_numeric(block_df["Parameters (M)"], errors="coerce").fillna(0)
block_df["Share of Total"] = pd.to_numeric(block_df["Share of Total"], errors="coerce").fillna(0)

detail_df = detail_df.sort_values("Order").reset_index(drop=True)
block_df = block_df.sort_values("Order").reset_index(drop=True)

orders = detail_df["Order"].to_numpy()
layer_params_M = detail_df["Parameters (M)"].to_numpy()

total_params = block_df["Parameters"].sum()
total_params_M = total_params / 1e6

########################################################################################################################
# Plot
########################################################################################################################
fig, axes = plt.subplots(
    2,
    1,
    figsize=(10, 10),
    gridspec_kw={"height_ratios": [1.00, 1.25]}
)
ax0 = axes[0]
ax1 = axes[1]

########################################################################################################################
# (a) Layer-wise cumulative parameter concentration
########################################################################################################################

layer_sort_idx = np.argsort(layer_params_M)[::-1]
layer_params_M_sorted = layer_params_M[layer_sort_idx]
layer_rank = np.arange(1, len(layer_params_M_sorted) + 1)
layer_cumulative_share = 100.0 * np.cumsum(layer_params_M_sorted) / np.sum(layer_params_M_sorted)

ax0.plot(
    layer_rank,
    layer_cumulative_share,
    color=(0.18, 0.38, 0.58),
    lw=1.8,
    marker="o",
    markersize=3.8,
    markerfacecolor=(1, 1, 1),
    markeredgecolor=(0.18, 0.38, 0.58),
    markeredgewidth=0.9
)

ax0.fill_between(
    layer_rank,
    layer_cumulative_share,
    color=(0.18, 0.38, 0.58),
    alpha=0.10
)

for y_ref in [50, 80, 90]:
    ax0.axhline(
        y_ref,
        color=(0.55, 0.55, 0.55),
        lw=0.6,
        ls="--",
        alpha=0.45
    )

annotate_ranks = [1, 5, 10]
annotate_ranks = [rank for rank in annotate_ranks if rank <= len(layer_rank)]

for rank in annotate_ranks:
    idx = rank - 1
    if rank == 10:
        xytext = (-8, -34)
        ha = "right"
        va = "top"
    else:
        xytext = (4, 10)
        ha = "left"
        va = "bottom"

    ax0.annotate(
        f"Top {rank}\n{layer_cumulative_share[idx]:.1f}%",
        xy=(layer_rank[idx], layer_cumulative_share[idx]),
        xytext=xytext,
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=text_size - 2,
        color=(0, 0, 0),
        bbox=dict(boxstyle="round,pad=0.20", fc=(1, 1, 1), ec=(0.65, 0.65, 0.65), lw=0.5, alpha=0.85)
    )

ax0.set_title(
    "(a) Layer concentration",
    fontsize=text_size + 1,
    color=(0, 0, 1),
    pad=8
)
ax0.set_xlabel("Top-ranked layers", fontsize=text_size)
ax0.set_ylabel("Cumulative parameters [%]", fontsize=text_size)

xticks = np.arange(1, len(layer_rank) + 1, 5)
ax0.set_xticks(xticks)
ax0.set_xticklabels(xticks, fontsize=text_size - 2, rotation=0)

ax0.tick_params(axis="both", labelsize=text_size)
ax0.tick_params(axis="x", length=0)
ax0.grid(
    axis="both",
    linestyle="-",
    linewidth=0.5,
    alpha=0.25
)
ax0.set_axisbelow(True)
ax0.set_xlim(1, len(layer_rank))
ax0.set_ylim(0, 105)

ax0.spines["top"].set_visible(False)
ax0.spines["right"].set_visible(False)
ax0.spines["left"].set_visible(False)
ax0.spines["bottom"].set_color((0.35, 0.35, 0.35))


########################################################################################################################
# (b) Functional block parameter totals
########################################################################################################################

block_plot_df = block_df.copy()
block_plot_df["Share (%)"] = 100.0 * block_plot_df["Share of Total"]
block_plot_df = block_plot_df.sort_values("Parameters (M)", ascending=True).reset_index(drop=True)

block_names_plot = block_plot_df["Functional Block"].astype(str).to_numpy()
block_params_M_plot = block_plot_df["Parameters (M)"].to_numpy()
block_share_plot = block_plot_df["Share (%)"].to_numpy()

block_label_dic = {
    "Velocity model encoder": "Velocity encoder",
    "Receiver encoding": "Receiver enc.",
}

block_names_plot = np.asarray([
    block_label_dic.get(block_name, block_name)
    for block_name in block_names_plot
])

y_pos = np.arange(len(block_names_plot))
max_block_params_M = np.max(block_params_M_plot)

bar_colors = [
    (0.18, 0.38, 0.58) if value_M == max_block_params_M else (0.72, 0.78, 0.82)
    for value_M in block_params_M_plot
]

ax1.barh(
    y_pos,
    block_params_M_plot,
    height=0.62,
    color=bar_colors,
    edgecolor=(0.20, 0.20, 0.20),
    linewidth=0.45
)

ax1.set_title(
    "(b) Functional block budget",
    fontsize=text_size + 1,
    color=(0, 0, 1),
    pad=8
)

ax1.set_xlabel(
    "Trainable parameters [M]",
    fontsize=text_size
)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(block_names_plot, fontsize=text_size)

ax1.tick_params(axis="both", labelsize=text_size)
ax1.tick_params(axis="y", length=0)

label_offset = max_block_params_M * 0.025

for i, (value_M, share) in enumerate(zip(block_params_M_plot, block_share_plot)):

    if value_M >= 0.01:
        label = f"{value_M:.2f} M  ({share:.1f}%)"
    elif value_M > 0:
        label = f"<0.01 M  ({share:.1f}%)"
    else:
        label = "0 M"

    ax1.text(
        value_M + label_offset,
        i,
        label,
        va="center",
        ha="left",
        fontsize=text_size,
        color=(0, 0, 0)
    )

ax1.grid(
    axis="x",
    linestyle="-",
    linewidth=0.5,
    alpha=0.25
)

ax1.set_axisbelow(True)

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.spines["bottom"].set_color((0.35, 0.35, 0.35))

ax1.set_xlim(
    0,
    max_block_params_M * 1.32
)


########################################################################################################################
# General layout and save
########################################################################################################################

# fig.suptitle(
#     f"Trainable Parameter Distribution of the Receiver-Query Neural Surrogate "
#     f"({total_params_M:.2f} M parameters)",
#     fontsize=text_size + 3,
#     fontweight="bold",
#     y=0.995
# )

fig.subplots_adjust(
    left=0.23,
    right=0.96,
    bottom=0.09,
    top=0.94,
    hspace=0.50
)

fig_path_svg = results_dir / "8.Trainable_Parameter_Profile_and_Block_Budget.svg"
fig_path_pdf = results_dir / "fig5.pdf"

plt.savefig(fig_path_svg, dpi=200, bbox_inches="tight", pad_inches=0.02)
plt.savefig(fig_path_pdf, dpi=200, bbox_inches="tight", pad_inches=0.02)

plt.show()

print(f"Saved SVG figure to: {fig_path_svg}")
print(f"Saved PDF figure to: {fig_path_pdf}")
print(f"Total trainable parameters: {total_params:,} ({total_params_M:.2f} M)")
