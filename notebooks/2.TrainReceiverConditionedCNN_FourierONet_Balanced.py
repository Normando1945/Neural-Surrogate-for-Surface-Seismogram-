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

SIMULATION OF FINITE DIFFERENCE METHOD OF 2D ACOUSTIC WAVE EQUATION, IN HETEROGENEOUS MEDIUM
                    AI SURFACE SEISMOGRAM EMULATION WORKFLOW
                  Velocity Models -> Wavefield -> Seismograms

                 Author: Msc. Ing. Carlos Andrés Celi Sánchez


    .PY FOR TRAINING THE FULL CNN WITH CONDITIONED RECEIVER, USING FOURIER ONET.
    BALANCED RECEIVER-QUERY SAMPLING VERSION FOR ACTIVE / WEAK / INACTIVE TRACES.

Purpose
-------
This file keeps the same general output structure as 3.2 so that the results
can still be reviewed by the downstream 5.2 workflow. The two key changes are:

    1. The CNN receives a normalized velocity model:
           x_model_input = (x_batch - 3000.0) / 1500.0

       while the loss still receives the physical velocity model x_batch.

    2. The training DataLoader uses a WeightedRandomSampler instead of naive
       shuffle=True. The sampler remains random, but it controls the expected
       proportion of active, weak, and inactive traces in each training epoch.
"""

#---------------- Basic Libraries ----------------
import sys
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

import time as tm

#---------------- Route of H5 file ----------------
project_root = Path(__file__).resolve().parent.parent

#---------------- Import all classes ----------------
from core import *

#---------------- Define H5 path ----------------
h5_path = project_root / "data" / "raw" / "dataset_surface_seismograms.h5"

print("\n" + "=" * 120)
print("HDF5 file path:")
print(h5_path)

if not h5_path.exists():
    raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

print("=" * 120)
print("\n")

#---------------- Select device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n" + "=" * 120)
print("Selected device:")
print(device)
print("=" * 120)
print("\n")

# #---------------- Output directories ----------------
outputs_dir = project_root / "outputs"                                                          # Define the outputs directory path
outputs_dir.mkdir(parents=True, exist_ok=True)                                                  # Create the outputs directory if it does not exist

#---------------- Checkpoints directory ----------------
checkpoints_dir = project_root / "checkpoints"                                                  # Define the checkpoints directory path
checkpoints_dir.mkdir(parents=True, exist_ok=True)                                              # Create the checkpoints directory if it does not exist

#---------------- Define model checkpoint paths ----------------
folder = "Test"
(checkpoints_dir / folder).mkdir(parents=True, exist_ok=True)
best_model_path = checkpoints_dir /folder/"best_receiver_conditioned_cnn.pth"                   # Define the path used to save the best receiver conditioned model
last_model_path = checkpoints_dir /folder/"last_receiver_conditioned_cnn.pth"                   # Define the path used to save the last receiver conditioned model

#---------------- Define receiver conditioned loss-history path ----------------
loss_history_path = checkpoints_dir /folder/"receiver_conditioned_cnn_loss_history.npz"         # Define the file path used to save the receiver conditioned loss history

print("\n" + "=" * 120)                                                                         # Print separator line
print("\nReceiver conditioned loss-history path:")                                              # Print loss-history file title
print(loss_history_path)                                                                        # Print loss-history file path
print("\nBest receiver conditioned model path:")                                                # Print best-model checkpoint title
print(best_model_path)                                                                          # Print best-model checkpoint path
print("\nLast receiver conditioned model path:")                                                # Print last-model checkpoint title
print(last_model_path)                                                                          # Print last-model checkpoint path
print("=" * 120)                                                                                # Print separator line
print("\n")                                                                                     # Print blank line


########################################################################################################################
########################################################################################################################
########################################## Receiver-Query samples per batch ############################################
########################################################################################################################
########################################################################################################################

batch_size_query = 8                                                                            # Number of receiver-query samples per batch
numberEpoch = 3                                                                                # Number of epochs to train the receiver-conditioned CNN


########################################################################################################################
########################################################################################################################
############################################ Balanced sampler setup ####################################################
########################################################################################################################
########################################################################################################################

#---------------- Model-input normalization ----------------
# The network receives a normalized velocity field, but the loss receives the original physical velocity field.
normalize_model_input = True                                                                     # Normalize velocity models before the CNN encoder
velocity_mean_for_model = 3000.0                                                                 # Velocity mean used for model-input normalization
velocity_std_for_model = 1500.0                                                                  # Velocity standard deviation used for model-input normalization

#---------------- Balanced random sampling by target-trace RMS ----------------
use_balanced_train_sampler = True                                                                # Replace naive shuffle with weighted random sampling
sampler_inactive_rms_threshold = 0.05                                                            # RMS below this value is treated as nearly silent
sampler_active_rms_threshold = 0.50                                                              # RMS equal or above this value is treated as active
sampler_active_fraction = 0.85                                                                   # Expected sampling fraction for active traces
sampler_weak_fraction = 0.10                                                                     # Expected sampling fraction for weak / partial traces
sampler_inactive_fraction = 0.05                                                                 # Expected sampling fraction for inactive traces
sampler_replacement = True                                                                       # Sample with replacement to preserve random balanced batches
sampler_random_seed = 12345                                                                      # Seed for the weighted random sampler


def normalize_velocity_for_model(x_batch):                                                       # Normalize velocity input only for the neural network
    """
    Normalize the velocity model before it enters the CNN branch.

    Important:
    x_batch must remain in physical velocity units when passed to the loss function,
    because the loss uses the physical velocity model for arrival-time and distance-aware terms.
    """
    if not normalize_model_input:                                                                # Return the original physical field when normalization is disabled
        return x_batch                                                                           # Keep velocity in physical units

    return (x_batch - velocity_mean_for_model) / velocity_std_for_model                          # Normalize velocity field for the CNN encoder


def _get_dataset_receiver_ids(dataset):                                                          # Recover the selected receiver ids used by the dataset
    """
    Return the receiver ids used by HDF5ReceiverQueryTorchDataset.

    The current full-training configuration uses receiver_ids=None, so all receivers
    are selected. This helper keeps the code robust if a receiver subset is used later.
    """
    if hasattr(dataset, "selected_receiver_ids"):                                                # Preferred attribute if available
        return np.asarray(dataset.selected_receiver_ids, dtype=int)                              # Return selected receiver ids

    if hasattr(dataset, "receiver_ids"):                                                         # Alternative attribute name used by some dataset implementations
        receiver_ids = getattr(dataset, "receiver_ids")                                         # Read receiver_ids attribute

        if receiver_ids is not None:                                                             # Use it only if it is not None
            return np.asarray(receiver_ids, dtype=int)                                           # Return selected receiver ids

    return np.arange(int(dataset.n_receivers_total), dtype=int)                                  # Fallback: all receivers are selected


def compute_query_target_rms_from_hdf5(h5_path, dataset):                                        # Compute target RMS for every receiver-query sample
    """
    Compute target RMS for each query in the same order expected by the dataset.

    For the common receiver-query ordering, the query order is:
        sample_0 receiver_0,
        sample_0 receiver_1,
        ...
        sample_1 receiver_0,
        ...
    """
    split_name = str(dataset.split) if hasattr(dataset, "split") else "train"                    # Read dataset split if available
    receiver_ids = _get_dataset_receiver_ids(dataset)                                            # Receiver ids used by the dataset

    rms_values = []                                                                              # Store target RMS values per query

    with h5py.File(h5_path, "r") as h5f:                                                         # Open HDF5 database
        split_sample_ids = h5f[f"splits/{split_name}_ids"][:].astype(int)                        # Read split sample ids
        surface_seismograms = h5f["outputs/surface_seismograms"]                                 # HDF5 dataset with full seismograms

        for sample_id in split_sample_ids:                                                       # Loop over samples in split order
            traces = surface_seismograms[int(sample_id), receiver_ids, :].astype(np.float32)      # Read selected receiver traces for this sample
            sample_rms = np.sqrt(np.mean(traces ** 2, axis=1))                                   # Compute RMS per receiver trace
            rms_values.extend(sample_rms.tolist())                                               # Append in receiver-query order

    rms_values = np.asarray(rms_values, dtype=np.float64)                                        # Convert to NumPy array

    if len(rms_values) != len(dataset):                                                          # Validate that RMS vector matches dataset length
        raise ValueError(
            f"Computed RMS vector length {len(rms_values)} does not match dataset length {len(dataset)}."
        )

    return rms_values                                                                            # Return target RMS per query


def build_balanced_query_sampler(target_rms_values):                                             # Build weighted random sampler by trace class
    """
    Build a WeightedRandomSampler so that training batches remain random but are not
    dominated by inactive or nearly silent traces.
    """
    target_rms_values = np.asarray(target_rms_values, dtype=np.float64)                          # Ensure NumPy format

    inactive_mask = target_rms_values < float(sampler_inactive_rms_threshold)                    # Nearly silent traces
    weak_mask = (target_rms_values >= float(sampler_inactive_rms_threshold)) & (target_rms_values < float(sampler_active_rms_threshold)) # Weak traces
    active_mask = target_rms_values >= float(sampler_active_rms_threshold)                       # Active traces

    class_masks = {
        "active": active_mask,                                                                   # Active trace mask
        "weak": weak_mask,                                                                       # Weak / partial trace mask
        "inactive": inactive_mask,                                                              # Inactive trace mask
    }

    target_fractions = {
        "active": float(sampler_active_fraction),                                                # Desired active fraction
        "weak": float(sampler_weak_fraction),                                                    # Desired weak fraction
        "inactive": float(sampler_inactive_fraction),                                            # Desired inactive fraction
    }

    available_fractions_sum = sum(target_fractions[name] for name, mask in class_masks.items() if int(mask.sum()) > 0) # Sum fractions for available classes

    if available_fractions_sum <= 0.0:                                                           # Stop if no class has samples
        raise ValueError("No samples were available to build the balanced sampler.")

    weights = np.zeros_like(target_rms_values, dtype=np.float64)                                 # Initialize sampling weights

    for class_name, mask in class_masks.items():                                                 # Loop over trace classes
        class_count = int(mask.sum())                                                            # Count traces in the current class

        if class_count == 0:                                                                     # Skip empty classes
            continue                                                                             # Continue to next class

        normalized_fraction = target_fractions[class_name] / available_fractions_sum             # Renormalize if a class is missing
        weights[mask] = normalized_fraction / float(class_count)                                 # Assign inverse-count class-balanced weights

    weights = weights / weights.sum()                                                            # Normalize weights for numerical stability
    weights_tensor = torch.as_tensor(weights, dtype=torch.double)                                # Convert to tensor required by WeightedRandomSampler

    sampler_generator = torch.Generator()                                                        # Create sampler random generator
    sampler_generator.manual_seed(int(sampler_random_seed))                                      # Seed sampler for reproducibility

    sampler = WeightedRandomSampler(                                                             # Create weighted random sampler
        weights=weights_tensor,                                                                  # Sampling weights per query
        num_samples=len(weights_tensor),                                                         # Keep one epoch equal to the original train-query count
        replacement=bool(sampler_replacement),                                                   # Sample with replacement for balanced random draws
        generator=sampler_generator,                                                             # Reproducible random generator
    )

    class_counts = {                                                                             # Store diagnostic class counts
        "active": int(active_mask.sum()),
        "weak": int(weak_mask.sum()),
        "inactive": int(inactive_mask.sum()),
    }

    class_fractions = {                                                                          # Store diagnostic class fractions
        "active": float(active_mask.mean()),
        "weak": float(weak_mask.mean()),
        "inactive": float(inactive_mask.mean()),
    }

    return sampler, class_counts, class_fractions                                                # Return sampler and class-count summary


########################################################################################################################
########################################################################################################################
################################################### Training dataset ###################################################
########################################################################################################################
########################################################################################################################

#---------------- Create training receiver-query dataset ----------------
dataset_train = HDF5ReceiverQueryTorchDataset(                                                  # Create the receiver-query training dataset
    h5_path=h5_path,                                                                            # Path to the HDF5 database
    split="train",                                                                              # Use the training split
    receiver_ids=None,                                                                          # Use all available receivers from the dense receiver bank
    normalize_x=False,                                                                          # Keep velocity models in physical units; normalize only before the model
    normalize_y=False,                                                                          # Keep target traces in physical amplitude units
    normalize_receiver_coords=True,                                                             # Use normalized receiver coordinates in [0, 1]
    return_metadata=True,                                                                       # Return metadata to keep the same loader structure
)

#---------------- Create training dataloader ----------------
if use_balanced_train_sampler:                                                                   # Use balanced random sampling for the training dataset
    train_target_rms_values = compute_query_target_rms_from_hdf5(                                # Compute RMS for every train query
        h5_path=h5_path,                                                                         # Path to HDF5 database
        dataset=dataset_train,                                                                   # Receiver-query training dataset
    )

    train_sampler, train_class_counts, train_class_fractions = build_balanced_query_sampler(train_target_rms_values) # Build balanced weighted sampler

    train_loader = DataLoader(                                                                   # Create DataLoader using the balanced random sampler
        dataset_train,
        batch_size=batch_size_query,
        sampler=train_sampler,                                                                   # Sampler already performs randomization
        shuffle=False,                                                                           # Must be False when a sampler is provided
        num_workers=0,
    )

else:                                                                                            # Fallback to standard random shuffle
    train_target_rms_values = None                                                               # No RMS vector used
    train_class_counts = {"active": -1, "weak": -1, "inactive": -1}                             # Placeholder class counts
    train_class_fractions = {"active": -1.0, "weak": -1.0, "inactive": -1.0}                    # Placeholder class fractions

    train_loader = DataLoader(                                                                   # Create standard shuffled DataLoader
        dataset_train,
        batch_size=batch_size_query,
        shuffle=True,
        num_workers=0,
    )

#---------------- Print training dataset information ----------------
print("\n" + "=" * 120)                                                                         # Print separator line
print("Training receiver-query dataset created successfully")                                   # Print dataset creation message
print(f"Train query samples       : {len(dataset_train)}")                                      # Print total number of receiver-query samples
print(f"Velocity model shape      : {dataset_train.x_shape}")                                   # Print shape of the velocity model
print(f"Full seismogram shape     : {dataset_train.y_shape}")                                   # Print original full seismogram shape stored in HDF5
print(f"Total receivers in HDF5   : {dataset_train.n_receivers_total}")                         # Print total number of receivers stored in HDF5
print(f"Selected receivers        : {dataset_train.n_selected_receivers}")                      # Print number of receivers selected in this dataset
print(f"Time samples per trace    : {dataset_train.n_time}")                                    # Print number of time samples in each trace
print(f"Batch size Query          : {batch_size_query}")                                        # Print batch size used for training
print(f"Balanced sampler enabled  : {use_balanced_train_sampler}")                              # Print balanced sampler flag
print(f"Sampler active count      : {train_class_counts['active']} ({train_class_fractions['active']:.3f})") # Print active traces
print(f"Sampler weak count        : {train_class_counts['weak']} ({train_class_fractions['weak']:.3f})")     # Print weak traces
print(f"Sampler inactive count    : {train_class_counts['inactive']} ({train_class_fractions['inactive']:.3f})") # Print inactive traces
print(f"Sampler target fractions  : active={sampler_active_fraction}, weak={sampler_weak_fraction}, inactive={sampler_inactive_fraction}") # Print target sampler fractions
print(f"Velocity normalization    : {normalize_model_input}")                                   # Print model-input normalization flag
print(f"Velocity mean / std       : {velocity_mean_for_model} / {velocity_std_for_model}")       # Print normalization constants
print("=" * 120)                                                                                # Print separator line
print("\n")                                                                                     # Print blank line


########################################################################################################################
########################################################################################################################
###############################################  Validation dataset  ###################################################
########################################################################################################################
########################################################################################################################

#---------------- Create validation receiver-query dataset ----------------
dataset_val = HDF5ReceiverQueryTorchDataset(                                                    # Create the validation receiver-query dataset
    h5_path=h5_path,                                                                            # Path to the HDF5 database
    split="val",                                                                                # Use the validation split
    receiver_ids=None,                                                                          # Use all available receivers from the dense receiver bank
    normalize_x=False,                                                                          # Keep velocity models in physical units
    normalize_y=False,                                                                          # Keep target traces in physical units
    normalize_receiver_coords=True,                                                             # Use normalized receiver coordinates in [0, 1]
    return_metadata=True,                                                                       # Return metadata to keep the same loader structure
)

#---------------- Create validation dataloader ----------------
val_loader = DataLoader(                                                                        # Create DataLoader for the validation dataset
    dataset_val,
    batch_size=batch_size_query,                                                                # Use the same batch size used for training
    shuffle=False,                                                                              # Do not shuffle validation data
    num_workers=0,
)

print("\n" + "=" * 120)                                                                         # Print separator line
print("Validation receiver-query dataset created successfully")                                 # Print dataset creation message
print(f"Validation query samples  : {len(dataset_val)}")                                        # Print total number of validation query samples
print("=" * 120)                                                                                # Print separator line
print("\n")                                                                                     # Print blank line


########################################################################################################################
########################################################################################################################
###############################################  Model hyperparameters  ################################################
########################################################################################################################
########################################################################################################################

coord_dim = 2                                                                                   # Receiver coordinate dimension = (x, z)
coord_num_bands = 16                                                                            # Number of Fourier bands for receiver-coordinate encoding
coord_max_frequency = 16.0                                                                      # Maximum Fourier frequency used in coordinate encoding
model_latent_dim = 512                                                                          # Latent dimension of the CNN velocity-model branch
coord_latent_dim = 256                                                                          # Latent dimension of the receiver-coordinate branch
fusion_dim = 1024                                                                               # Latent dimension after branch fusion
decoder_hidden_dim = 2048                                                                       # Hidden width of the temporal decoder
dropout = 0.05                                                                                  # Use nonzero dropout
learning_rate = 1e-4                                                                            # Learning rate for Adam optimizer
num_epochs = numberEpoch                                                                        # Number of full passes through the train dataset

print("\n" + "=" * 120)                                                                         # Print separator line
print("Receiver-conditioned model hyperparameters")                                             # Print hyperparameter section title
print(f"coord_dim               : {coord_dim}")                                                 # Print coordinate dimension
print(f"coord_num_bands         : {coord_num_bands}")                                           # Print number of Fourier bands
print(f"coord_max_frequency     : {coord_max_frequency}")                                       # Print maximum Fourier frequency
print(f"model_latent_dim        : {model_latent_dim}")                                          # Print model latent dimension
print(f"coord_latent_dim        : {coord_latent_dim}")                                          # Print receiver latent dimension
print(f"fusion_dim              : {fusion_dim}")                                                # Print fusion latent dimension
print(f"decoder_hidden_dim      : {decoder_hidden_dim}")                                        # Print decoder hidden dimension
print(f"dropout                 : {dropout}")                                                   # Print dropout value
print(f"learning_rate           : {learning_rate}")                                             # Print learning rate
print(f"num_epochs              : {num_epochs}")                                                # Print number of epochs
print("=" * 120)                                                                                # Print separator line
print("\n")                                                                                     # Print blank line


########################################################################################################################
########################################################################################################################
################################################  Create model  ########################################################
########################################################################################################################
########################################################################################################################

#---------------- Define output trace length ----------------
n_time = dataset_train.n_time                                                                   # Read output trace length directly from the dataset

#---------------- Create receiver-conditioned model ----------------
model = BaselineReceiverConditionedSeismogramNet(                                               # Create the receiver-conditioned baseline network
    n_time=n_time,                                                                              # Set output trace length
    coord_dim=coord_dim,                                                                        # Set coordinate dimension
    coord_num_bands=coord_num_bands,                                                            # Set number of Fourier bands
    coord_max_frequency=coord_max_frequency,                                                    # Set maximum Fourier frequency
    model_latent_dim=model_latent_dim,                                                          # Set CNN-branch latent dimension
    coord_latent_dim=coord_latent_dim,                                                          # Set receiver-branch latent dimension
    fusion_dim=fusion_dim,                                                                      # Set fusion latent dimension
    decoder_hidden_dim=decoder_hidden_dim,                                                      # Set decoder hidden dimension
    dropout=dropout,                                                                            # Set dropout value
).to(device)                                                                                    # Move the model to CPU or GPU

print("\n" + "=" * 120)                                                                         # Print separator line
print("Receiver-conditioned model created successfully")                                        # Print model creation message
print(model)                                                                                    # Print model architecture
print("=" * 120)                                                                                # Print separator line
print("\n")                                                                                     # Print blank line


########################################################################################################################
########################################################################################################################
###############################################  Loss and optimizer  ###################################################
########################################################################################################################
########################################################################################################################

#---------------- Define loss function ----------------
criterion = MaskedPseudoPhysicsInformedTraceLoss_v2(
    alpha_time=0.05,
    alpha_relative=0.30,
    alpha_correlation=1.00,
    alpha_normalized=0.50,
    alpha_derivative=0.10,
    alpha_energy=1.00,
    alpha_frequency=0.00,
    alpha_arrival=0.00,
    alpha_silence=1.00,
    alpha_partial_time=0.10,
    alpha_partial_energy=0.50,
    alpha_partial_silence=0.50,
    active_rms_threshold=0.50,
    significant_abs_threshold=1.0e-4,
    significant_rel_threshold=0.05,
    min_significant_duration_s=0.05,
    min_post_arrival_duration_s=0.05,
    signal_window_padding_samples=8,
    relative_power_floor=1.0,
    source_time_index=100,
)

print("\n" + "=" * 120)                                                                         # Print separator line
print("Loss function created successfully")                                                     # Print confirmation message
print(criterion)                                                                                # Print selected loss function
print("=" * 120)                                                                                # Print separator line
print("\n")                                                                                     # Print blank line

#---------------- Define optimizer ----------------
optimizer = torch.optim.Adam(                                                                   # Create Adam optimizer for the receiver-conditioned model
    model.parameters(),                                                                         # Optimize all trainable model parameters
    lr=learning_rate,                                                                           # Use the selected learning rate
    weight_decay=0.0,                                                                           # Disable weight decay
)

print("\n" + "=" * 120)                                                                         # Print separator line
print("Optimizer created successfully")                                                         # Print confirmation message
print(optimizer)                                                                                # Print optimizer configuration
print("=" * 120)                                                                                # Print separator line
print("\n")                                                                                     # Print blank line


########################################################################################################################
########################################################################################################################
#################################################  TRAINING  ###########################################################
########################################################################################################################
########################################################################################################################

best_val_loss = float("inf")                                                                    # Initialize best validation loss tracker
best_epoch = -1                                                                                 # Initialize best epoch tracker

train_losses = []                                                                               # Store average training loss for each epoch
val_losses = []                                                                                 # Store average validation loss for each epoch

train_batch_losses = []                                                                         # Store sampled training-batch losses across all epochs
train_batch_steps = []                                                                          # Store sampled global batch-step indices
train_batch_epochs = []                                                                         # Store epoch index for each sampled training-batch loss
train_batch_step_in_epoch = []                                                                  # Store batch index inside epoch for each sampled training-batch loss
epoch_times_sec = []                                                                            # Store elapsed time per epoch in seconds
global_train_step = 0                                                                           # Global counter of training batches across all epochs

t_start = tm.time()                                                                             # Store training start time

print("\n" + "=" * 120)                                                                         # Print separator line
print("Start full receiver-conditioned training")                                               # Print section title
print(f"Train query samples         : {len(dataset_train)}")                                    # Print number of train queries
print(f"Validation query samples    : {len(dataset_val)}")                                      # Print number of validation queries
print(f"Batch size                  : {batch_size_query}")                                      # Print training batch size
print(f"Number of epochs            : {num_epochs}")                                            # Print total number of epochs
print(f"Balanced sampler             : {use_balanced_train_sampler}")                            # Print balanced sampler flag
print(f"Velocity input normalized    : {normalize_model_input}")                                 # Print velocity normalization flag
print("=" * 120)                                                                                # Print separator line
print("\n")                                                                                     # Print blank line

#-------------------------------------------------------------------------------------------------------------------------------------------#
train_batch_log_stride = 10                                                                     # Number of training steps between saving training-batch losses
print_interval = 10                                                                             # Number of batches between printing training status updates
#-------------------------------------------------------------------------------------------------------------------------------------------#

for epoch in range(num_epochs):                                                                 # Loop over epochs

    ####################################################################################################################
    ################################################## Train phase #####################################################
    ####################################################################################################################

    epoch_start_time = tm.time()                                                                # Store start time of the current epoch

    model.train()                                                                               # Put the model in training mode
    running_train_loss = 0.0                                                                    # Initialize train-loss accumulator

    for batch_idx, (x_batch, receiver_coord_batch, y_trace_batch, metadata_batch) in enumerate(train_loader, start=1):  # Loop over training batches
        x_batch = x_batch.to(device=device, dtype=torch.float32)                                                       # Move velocity-model batch to CPU or GPU
        receiver_coord_batch = receiver_coord_batch.to(device=device, dtype=torch.float32)                             # Move receiver-coordinate batch to CPU or GPU
        y_trace_batch = y_trace_batch.to(device=device, dtype=torch.float32)                                           # Move target-trace batch to CPU or GPU

        optimizer.zero_grad()                                                                                        # Reset gradients before backpropagation

        x_model_input = normalize_velocity_for_model(x_batch)                                                       # Normalize velocity model only for the CNN encoder

        y_pred = model(                                                                                              # Run forward pass for the current training batch
            x_model_input,                                                                                           # Normalized velocity-model batch for the neural network
            receiver_coord_batch,                                                                                    # Input receiver-coordinate batch
        )

        loss, loss_terms = criterion.forward_from_metadata(                                                          # Compute masked TIER 4 loss
            y_pred=y_pred,                                                                                           # Predicted traces
            y_true=y_trace_batch,                                                                                    # Ground-truth traces
            velocity_model=x_batch,                                                                                  # Physical velocity model used by the loss
            metadata_batch=metadata_batch,                                                                           # Metadata with dt, dx, dz, source and receiver coordinates
        )

        if not torch.isfinite(loss):                                                                                 # Check for numerical instability
            raise ValueError(
                f"Loss is not finite at epoch {epoch + 1}. Current loss value: {loss.item()}"
            )

        loss.backward()                                                                                              # Compute gradients through backpropagation

        torch.nn.utils.clip_grad_norm_(                                                                              # Clip gradients for stability
            model.parameters(),                                                                                      # Apply clipping to all model parameters
            max_norm=5.0,                                                                                            # Maximum allowed gradient norm
        )

        optimizer.step()                                                                                             # Update model parameters

        running_train_loss += loss.item() * x_batch.size(0)                                                          # Accumulate total train loss weighted by batch size

        global_train_step += 1                                                                                       # Increase global training-batch counter
        if global_train_step == 1 or global_train_step % train_batch_log_stride == 0:                                # Save sampled training-batch loss
            train_batch_losses.append(float(loss.item()))                                                            # Store sampled training-batch loss
            train_batch_steps.append(int(global_train_step))                                                         # Store sampled global training-batch step
            train_batch_epochs.append(int(epoch + 1))                                                                # Store 1-based epoch index
            train_batch_step_in_epoch.append(int(batch_idx))                                                         # Store batch index inside current epoch

        if batch_idx % print_interval == 0:                                                                          # Print training status every selected number of batches
            print(
                f"Epoch [{epoch + 1:03d}/{num_epochs:03d}] | "
                f"Train batch [{batch_idx:05d}/{len(train_loader):05d}] | "
                f"Total = {loss.item():.6f} | "
                f"Active = {loss_terms['loss_active'].item():.6f} | "
                f"Partial = {loss_terms['loss_partial'].item():.6f} | "
                f"Inactive = {loss_terms['loss_inactive'].item():.6f} | "
                f"Rel = {loss_terms['loss_relative'].item():.6f} | "
                f"CorrLoss = {loss_terms['loss_correlation'].item():.6f} | "
                f"CorrCoef = {1.0 - loss_terms['loss_correlation'].item():.6f} | "  
                f"Norm = {loss_terms['loss_normalized'].item():.6f} | "
                f"Energy = {loss_terms['loss_energy'].item():.6f} | "
                f"Silence = {loss_terms['loss_silence'].item():.6f} | "
                f"ActiveFrac = {loss_terms['active_fraction'].item():.3f} | "
                f"PartialFrac = {loss_terms['partial_fraction'].item():.3f} | "
                f"InactiveFrac = {loss_terms['inactive_fraction'].item():.3f} | "
                f"SigDur = {loss_terms['significant_duration_mean_s'].item():.4f} s",
                flush=True,
            )

    epoch_train_loss = running_train_loss / len(dataset_train)                                  # Compute sampled average training loss
    train_losses.append(epoch_train_loss)                                                       # Store average training loss

    ####################################################################################################################
    ################################################ Validation phase ##################################################
    ####################################################################################################################

    model.eval()                                                                                # Put the model in evaluation mode
    running_val_loss = 0.0                                                                      # Initialize validation-loss accumulator

    with torch.no_grad():                                                                       # Disable gradient computation during validation
        for x_batch, receiver_coord_batch, y_trace_batch, metadata_batch in val_loader:          # Loop over validation batches
            x_batch = x_batch.to(device=device, dtype=torch.float32)                            # Move velocity-model batch to CPU or GPU
            receiver_coord_batch = receiver_coord_batch.to(device=device, dtype=torch.float32)  # Move receiver-coordinate batch to CPU or GPU
            y_trace_batch = y_trace_batch.to(device=device, dtype=torch.float32)                # Move target-trace batch to CPU or GPU

            x_model_input = normalize_velocity_for_model(x_batch)                               # Normalize velocity model only for the CNN encoder

            y_pred = model(                                                                     # Run forward pass for the current validation batch
                x_model_input,                                                                  # Normalized velocity-model batch for the neural network
                receiver_coord_batch,                                                           # Input receiver-coordinate batch
            )

            loss, loss_terms = criterion.forward_from_metadata(                                 # Compute masked validation TIER 4 loss
                y_pred=y_pred,                                                                  # Predicted traces
                y_true=y_trace_batch,                                                           # Ground-truth traces
                velocity_model=x_batch,                                                         # Physical velocity model used by the loss
                metadata_batch=metadata_batch,                                                  # Metadata with dt, dx, dz, source and receiver coordinates
            )

            if not torch.isfinite(loss):                                                        # Check for numerical instability
                raise ValueError(
                    f"Validation loss is not finite at epoch {epoch + 1}. Current loss value: {loss.item()}"
                )

            running_val_loss += loss.item() * x_batch.size(0)                                   # Accumulate total validation loss weighted by batch size

    epoch_val_loss = running_val_loss / len(dataset_val)                                        # Compute average validation loss over full validation dataset
    val_losses.append(epoch_val_loss)                                                           # Store average validation loss

    epoch_end_time = tm.time()                                                                  # Store end time of current epoch
    epoch_times_sec.append(float(epoch_end_time - epoch_start_time))                            # Store elapsed time per epoch

    ####################################################################################################################
    ################################################ Save best model ###################################################
    ####################################################################################################################

    if epoch_val_loss < best_val_loss:                                                          # Check whether current validation loss is best so far
        best_val_loss = epoch_val_loss                                                          # Update best validation loss
        best_epoch = epoch + 1                                                                  # Store 1-based epoch index

        torch.save(                                                                             # Save best receiver-conditioned checkpoint
            {
                "epoch": int(best_epoch),                                                       # Save best epoch number
                "model_state_dict": model.state_dict(),                                         # Save model weights
                "optimizer_state_dict": optimizer.state_dict(),                                 # Save optimizer state
                "train_loss": float(epoch_train_loss),                                          # Save train loss at best epoch
                "val_loss": float(epoch_val_loss),                                              # Save validation loss at best epoch
                "n_time": int(n_time),                                                          # Save output trace length
                "coord_dim": int(coord_dim),                                                    # Save coordinate dimension
                "coord_num_bands": int(coord_num_bands),                                        # Save number of Fourier bands
                "coord_max_frequency": float(coord_max_frequency),                              # Save maximum Fourier frequency
                "model_latent_dim": int(model_latent_dim),                                      # Save model latent dimension
                "coord_latent_dim": int(coord_latent_dim),                                      # Save receiver latent dimension
                "fusion_dim": int(fusion_dim),                                                  # Save fusion latent dimension
                "decoder_hidden_dim": int(decoder_hidden_dim),                                  # Save decoder hidden dimension
                "dropout": float(dropout),                                                      # Save dropout used in the model
                "normalize_model_input": bool(normalize_model_input),                            # Save velocity normalization flag
                "velocity_mean_for_model": float(velocity_mean_for_model),                       # Save velocity normalization mean
                "velocity_std_for_model": float(velocity_std_for_model),                         # Save velocity normalization standard deviation
                "use_balanced_train_sampler": bool(use_balanced_train_sampler),                  # Save balanced sampler flag
                "sampler_inactive_rms_threshold": float(sampler_inactive_rms_threshold),         # Save inactive threshold
                "sampler_active_rms_threshold": float(sampler_active_rms_threshold),             # Save active threshold
                "sampler_active_fraction": float(sampler_active_fraction),                       # Save active sampling fraction
                "sampler_weak_fraction": float(sampler_weak_fraction),                           # Save weak sampling fraction
                "sampler_inactive_fraction": float(sampler_inactive_fraction),                   # Save inactive sampling fraction
            },
            best_model_path,                                                                    # Output file path for best checkpoint
        )

    print(                                                                                      # Print one-line epoch summary
        f"Epoch [{epoch + 1:03d}/{num_epochs:03d}] | "
        f"Train Loss = {epoch_train_loss:.6f} | "
        f"Val Loss = {epoch_val_loss:.6f} | "
        f"Best Val = {best_val_loss:.6f}"
    )


########################################################################################################################
########################################################################################################################
################################################### Save last model ####################################################
########################################################################################################################
########################################################################################################################

torch.save(                                                                                     # Save final receiver-conditioned checkpoint
    {
        "epoch": int(num_epochs),                                                               # Save last epoch number
        "model_state_dict": model.state_dict(),                                                 # Save final model weights
        "optimizer_state_dict": optimizer.state_dict(),                                         # Save final optimizer state
        "train_losses": np.array(train_losses, dtype=float),                                    # Save complete train-loss history
        "val_losses": np.array(val_losses, dtype=float),                                        # Save complete validation-loss history
        "best_epoch": int(best_epoch),                                                          # Save best epoch index
        "best_val_loss": float(best_val_loss),                                                  # Save best validation loss
        "n_time": int(n_time),                                                                  # Save output trace length
        "coord_dim": int(coord_dim),                                                            # Save coordinate dimension
        "coord_num_bands": int(coord_num_bands),                                                # Save number of Fourier bands
        "coord_max_frequency": float(coord_max_frequency),                                      # Save maximum Fourier frequency
        "model_latent_dim": int(model_latent_dim),                                              # Save model latent dimension
        "coord_latent_dim": int(coord_latent_dim),                                              # Save receiver latent dimension
        "fusion_dim": int(fusion_dim),                                                          # Save fusion latent dimension
        "decoder_hidden_dim": int(decoder_hidden_dim),                                          # Save decoder hidden dimension
        "dropout": float(dropout),                                                              # Save dropout used in the model
        "normalize_model_input": bool(normalize_model_input),                                   # Save velocity normalization flag
        "velocity_mean_for_model": float(velocity_mean_for_model),                              # Save velocity normalization mean
        "velocity_std_for_model": float(velocity_std_for_model),                                # Save velocity normalization standard deviation
        "use_balanced_train_sampler": bool(use_balanced_train_sampler),                         # Save balanced sampler flag
        "sampler_inactive_rms_threshold": float(sampler_inactive_rms_threshold),                # Save inactive threshold
        "sampler_active_rms_threshold": float(sampler_active_rms_threshold),                    # Save active threshold
        "sampler_active_fraction": float(sampler_active_fraction),                              # Save active sampling fraction
        "sampler_weak_fraction": float(sampler_weak_fraction),                                  # Save weak sampling fraction
        "sampler_inactive_fraction": float(sampler_inactive_fraction),                          # Save inactive sampling fraction
    },
    last_model_path,                                                                            # Output file path for last checkpoint
)

np.savez(                                                                                       # Save the train/validation loss history to compressed NumPy file
    loss_history_path,                                                                          # Output file path
    train_losses=np.array(train_losses, dtype=float),                                           # Save full training-loss history per epoch
    val_losses=np.array(val_losses, dtype=float),                                               # Save full validation-loss history per epoch
    train_batch_losses=np.array(train_batch_losses, dtype=float),                               # Save sampled training-batch losses
    train_batch_steps=np.array(train_batch_steps, dtype=int),                                   # Save global batch-step indices
    train_batch_epochs=np.array(train_batch_epochs, dtype=int),                                 # Save epoch index for each sampled training-batch loss
    train_batch_step_in_epoch=np.array(train_batch_step_in_epoch, dtype=int),                   # Save batch index inside epoch for each sampled training-batch loss
    epoch_times_sec=np.array(epoch_times_sec, dtype=float),                                     # Save elapsed time per epoch
    best_epoch=int(best_epoch),                                                                 # Save best epoch index
    best_val_loss=float(best_val_loss),                                                         # Save best validation loss
    num_epochs=int(num_epochs),                                                                 # Save number of epochs
    batch_size_query=int(batch_size_query),                                                     # Save batch size used during training
    normalize_model_input=bool(normalize_model_input),                                          # Save velocity normalization flag
    velocity_mean_for_model=float(velocity_mean_for_model),                                     # Save velocity normalization mean
    velocity_std_for_model=float(velocity_std_for_model),                                       # Save velocity normalization standard deviation
    use_balanced_train_sampler=bool(use_balanced_train_sampler),                                # Save balanced sampler flag
    sampler_inactive_rms_threshold=float(sampler_inactive_rms_threshold),                       # Save inactive RMS threshold
    sampler_active_rms_threshold=float(sampler_active_rms_threshold),                           # Save active RMS threshold
    sampler_active_fraction=float(sampler_active_fraction),                                     # Save active sampling fraction
    sampler_weak_fraction=float(sampler_weak_fraction),                                         # Save weak sampling fraction
    sampler_inactive_fraction=float(sampler_inactive_fraction),                                 # Save inactive sampling fraction
)

print("\n" + "=" * 120)                                                                         # Print separator line
print("Receiver-conditioned checkpoints and loss history saved successfully")                   # Print confirmation message
print(f"Best model path        : {best_model_path}")                                            # Print best checkpoint path
print(f"Last model path        : {last_model_path}")                                            # Print last checkpoint path
print(f"Loss history path      : {loss_history_path}")                                          # Print loss-history path
print("=" * 120)                                                                                # Print separator line
print("\n")                                                                                     # Print blank line

t_end = tm.time()                                                                               # Store training end time
elapsed_time = t_end - t_start                                                                  # Compute total elapsed training time

print("\n" + "=" * 120)                                                                         # Print separator line
print("Total computation time")                                                                 # Print timing title
print(f"Computation Time in seconds = {elapsed_time:.2f}")                                      # Print total training time in seconds
print(f"Computation Time in minutes = {elapsed_time / 60:.2f}")                                 # Print total training time in minutes
print(f"Computation Time in hours   = {elapsed_time / 3600:.4f}")                               # Print total training time in hours
print("=" * 120)                                                                                # Print separator line
print("\n")                                                                                     # Print blank line
