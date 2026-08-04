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

                    AI SURFACE SEISMOGRAM EMULATION WORKFLOW
                  Velocity Models -> Wavefield -> Seismograms

                        Author: MSc. Ing. Carlos Celi
                                                

            .PY FOR REVIEW SPLIT DATA AND PREPARE TO PYTHORCH (USING QUERY).
"""
#---------------- Basic Libraries ----------------
import sys
from pathlib import Path
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

#---------------- Import all classes ----------------
from core import *

# ---------------- Define HDF5 path ----------------
project_root = Path.cwd()
h5_path = project_root / "data" / "raw" / "dataset_surface_seismograms.h5"

print("\n" + "=" * 120)
print("HDF5 file path:")
print(h5_path)

if not h5_path.exists():
    raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

print("=" * 120)
print("\n")

# ---------------- Create datasets ----------------
dataset_train = HDF5ReceiverQueryTorchDataset(
    h5_path=h5_path,
    split="train",
    receiver_ids=None,
    normalize_x=False,
    normalize_y=False,
    normalize_receiver_coords=True,
    return_metadata=True,
)

dataset_val = HDF5ReceiverQueryTorchDataset(
    h5_path=h5_path,
    split="val",
    receiver_ids=None,
    normalize_x=False,
    normalize_y=False,
    normalize_receiver_coords=True,
    return_metadata=True,
)

dataset_test = HDF5ReceiverQueryTorchDataset(
    h5_path=h5_path,
    split="test",
    receiver_ids=None,
    normalize_x=False,
    normalize_y=False,
    normalize_receiver_coords=True,
    return_metadata=True,
)

# ---------------- Print general dataset information ----------------
print("\n" + "=" * 120)
print("Receiver-query dataset inspection")
print("=" * 120)
print(f"Train query samples      : {len(dataset_train)}")
print(f"Validation query samples : {len(dataset_val)}")
print(f"Test query samples       : {len(dataset_test)}")
print(f"Velocity model shape     : {dataset_train.x_shape}")
print(f"Full seismogram shape    : {dataset_train.y_shape}")
print(f"Total receivers in HDF5  : {dataset_train.n_receivers_total}")
print(f"Selected receivers       : {dataset_train.n_selected_receivers}")
print(f"Time samples per trace   : {dataset_train.n_time}")
print("=" * 120)
print("\n")

# ---------------- Read one query sample ----------------
#--------------------------------------------------------
requested_query_index = 3869                                                                  # Keep the preferred inspection query index when the dataset is large enough
if len(dataset_train) == 0:                                                                    # Validate that the training receiver-query dataset is not empty
    raise ValueError("dataset_train is empty; no receiver-query sample can be inspected.")     # Stop with a clear message if there are no train queries
query_index = min(requested_query_index, len(dataset_train) - 1)                               # Clamp the requested query index to the available dataset range
if query_index != requested_query_index:                                                       # Notify the user when the preferred query index was outside the current dataset
    print(f"Requested query_index={requested_query_index} is out of range; using query_index={query_index}.")  # Print the safe replacement index
#--------------------------------------------------------
x_tensor, receiver_coord_tensor, y_trace_tensor, metadata = dataset_train[query_index]
print("\n" + "=" * 120)
print("One receiver-query sample loaded successfully")
print("=" * 120)
print(f"Query index                    : {query_index}")
print(f"Mapped sample_id               : {metadata['sample_id']}")
print(f"Receiver id (0-based)          : {metadata['receiver_id']}")
print(f"Receiver id (1-based)          : {metadata['receiver_id_1_based']}")
print(f"Receiver x                     : {metadata['receiver_x']}")
print(f"Receiver z                     : {metadata['receiver_z']}")
print(f"Receiver coord raw             : {metadata['receiver_coord_raw']}")
print(f"Receiver coord used            : {metadata['receiver_coord_used']}")
print(f"Source x                       : {metadata['source_x']}")
print(f"Source z                       : {metadata['source_z']}")
print(f"dt                             : {metadata['dt']}")
print(f"dx                             : {metadata['dx']}")
print(f"dz                             : {metadata['dz']}")
print(f"Model type                     : {metadata['model_type']}")
print("-" * 120)
print(f"x_tensor shape                 : {x_tensor.shape}")
print(f"receiver_coord_tensor shape    : {receiver_coord_tensor.shape}")
print(f"y_trace_tensor shape           : {y_trace_tensor.shape}")
print(f"x_tensor dtype                 : {x_tensor.dtype}")
print(f"receiver_coord_tensor dtype    : {receiver_coord_tensor.dtype}")
print(f"y_trace_tensor dtype           : {y_trace_tensor.dtype}")
print("=" * 120)
print("\n")

# ---------------- Basic numerical inspection ----------------
x_np = x_tensor[0].numpy()
receiver_coord_np = receiver_coord_tensor.numpy()
y_trace_np = y_trace_tensor.numpy()

print("\n" + "=" * 120)
print("Numerical inspection")
print("=" * 120)
print(f"x min                          : {x_np.min()}")
print(f"x max                          : {x_np.max()}")
print(f"receiver_coord min/max         : {receiver_coord_np.min()} / {receiver_coord_np.max()}")
print(f"y_trace min                    : {y_trace_np.min()}")
print(f"y_trace max                    : {y_trace_np.max()}")
print("=" * 120)
print("\n")



#########################################################################################################################################################################
#########################################################################################################################################################################
############################## Preare dataloaders for pyTorch training using the created datasets width batch size of "n" size and QUERY ################################
#########################################################################################################################################################################
#########################################################################################################################################################################

# ---------------- Create DataLoader ----------------
batch_size = 8

train_loader = DataLoader(
    dataset_train,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    dataset_val,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
)

# ---------------- Read one batch ----------------
x_batch, receiver_coord_batch, y_trace_batch, metadata_batch = next(iter(train_loader))

print("\n" + "=" * 120)
print("Receiver-query DataLoader batch inspection")
print("=" * 120)
print(f"x_batch shape                    : {x_batch.shape}")
print(f"receiver_coord_batch shape       : {receiver_coord_batch.shape}")
print(f"y_trace_batch shape              : {y_trace_batch.shape}")
print(f"x_batch dtype                    : {x_batch.dtype}")
print(f"receiver_coord_batch dtype       : {receiver_coord_batch.dtype}")
print(f"y_trace_batch dtype              : {y_trace_batch.dtype}")
print("-" * 120)
print(f"sample_id batch                  : {metadata_batch['sample_id']}")
print(f"receiver_id batch                : {metadata_batch['receiver_id']}")
print(f"receiver_x batch                 : {metadata_batch['receiver_x']}")
print(f"receiver_z batch                 : {metadata_batch['receiver_z']}")
print(f"source_x batch                   : {metadata_batch['source_x']}")
print(f"source_z batch                   : {metadata_batch['source_z']}")
print("=" * 120)
print("\n")
