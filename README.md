# Neural Surrogate for Surface Seismogram Emulation

<p align="center">
  <img src="outputs/Results_Conditioned_Recivers_FourierOnet/0.%20Test/3.receiver_conditioned_cnn_prediction_review_sample_1_Mapped_sample_6.svg" alt="Receiver-conditioned seismic-trace prediction review" width="100%">
</p>

This repository contains the computational workflow for emulating surface seismograms from heterogeneous two-dimensional velocity models. It combines finite-difference acoustic-wave simulations with a receiver-query neural surrogate: a velocity model and an explicit receiver location are used to predict one seismic trace.

The repository accompanies ongoing research. The associated manuscript is not yet published; this README intentionally describes the software workflow and its reproducible settings without reproducing unpublished manuscript text, figures, or conclusions.

## What is this repository for?

The workflow creates synthetic acoustic-wave simulations, stores them in HDF5 format, prepares receiver-query samples for PyTorch, trains a receiver-conditioned CNN with Fourier-encoded coordinates, and reviews the resulting predictions.

The four active executables are intended to be run in this order:

```text
0.DataGeneratorFromSeed.ipynb
        -> data/raw/dataset_surface_seismograms.h5
1.PrepareDataPyTorch_width_Query.py
        -> receiver-query and split inspection
2.TrainReceiverConditionedCNN_FourierONet_Balanced.py
        -> checkpoints/Test/
3.ReviewReceiverConditionedCNN_FourierONet_shuffled_coords_CORRECTED.py
        -> outputs/Results_Conditioned_Recivers_FourierOnet/
```

## Before you start

The project has been prepared for Python 3.10 or newer. You will need:

- Git and Python 3.10+.
- PyTorch. A CUDA-capable NVIDIA GPU is strongly recommended for the larger experiment, although the quick test can run on CPU.
- HDF5 support through `h5py`.
- NumPy, Pandas, SciPy, Matplotlib, IPython, and Pillow.
- PyVista, PyVistaQt, PySide6, and QtPy. These are required by the current `core` package, including when animations are disabled.
- Sufficient storage for generated HDF5 data, checkpoints, and figures. The 1,000-model configuration is substantially more demanding than the quick test.

> **Important:** install PyTorch first using the command recommended by the [official PyTorch installer](https://pytorch.org/get-started/locally/) for your operating system and CUDA version. Do not assume that one CUDA wheel fits every machine.

## How do I get set up?

Clone the repository and create an isolated environment.

```bash
git clone <repository-url>
cd Neural-Surrogate-for-Surface-Seismogram-

python -m venv .venv
```

Activate the environment.

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux or macOS
source .venv/bin/activate
```

Install PyTorch following the official selector. For a CPU-only environment, a typical command is:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then install the libraries used by this workflow.

```bash
pip install numpy pandas scipy matplotlib ipython pillow h5py
pip install pyvista pyvistaqt PySide6 qtpy
```

Install the local package in editable mode and verify the environment.

```bash
pip install -e .
python -c "import torch, h5py, numpy, pandas, scipy, matplotlib, pyvista, pyvistaqt, qtpy; import core; print('Environment OK')"
```

## Usage

Run every command from the repository root. The notebook is opened with Jupyter; the other three files are ordinary Python scripts.

### 1. Quick test

The committed defaults are deliberately small:

| Setting | Default | Location |
|---|---:|---|
| Generated simulations | `n_samples = 10` | `0.DataGeneratorFromSeed.ipynb` |
| Wavefield animation | `anim = 0` | `0.DataGeneratorFromSeed.ipynb` |
| Training epochs | `numberEpoch = 3` | `2.TrainReceiverConditionedCNN_FourierONet_Balanced.py` |

These settings are a **smoke test only**. They confirm that data generation, HDF5 storage, receiver-query preparation, training, checkpoints, and figures work together. They must not be interpreted as scientific validation, a generalization study, or representative paper-scale results.

Run the workflow:

```bash
jupyter notebook notebooks/0.DataGeneratorFromSeed.ipynb
python notebooks/1.PrepareDataPyTorch_width_Query.py
python notebooks/2.TrainReceiverConditionedCNN_FourierONet_Balanced.py
python notebooks/3.ReviewReceiverConditionedCNN_FourierONet_shuffled_coords_CORRECTED.py
```

Execute the notebook cells in order before proceeding to the Python scripts.

### 2. Larger experiment configuration

For the intended larger-scale workflow, edit the following values before starting a new run:

```python
# notebooks/0.DataGeneratorFromSeed.ipynb
n_samples = 1000

# notebooks/2.TrainReceiverConditionedCNN_FourierONet_Balanced.py
numberEpoch = 80
```

This configuration requires substantially more compute time, memory, disk space, and checkpoint storage. Use a dedicated output/checkpoint folder for each run so that a larger experiment does not overwrite a quick-test run.

### 3. Optional wavefield animations

To display or save wavefield animations during data generation, change the notebook setting to:

```python
anim = 1
```

Animations are useful for visual inspection of propagation, but they increase runtime and require a working graphical environment. Keep `anim = 0` for unattended generation or a minimal smoke test.

## Output files

After a successful workflow, the main generated artifacts are:

- `data/raw/dataset_surface_seismograms.h5`: velocity models, surface seismograms, physical metadata, and train/validation/test split identifiers.
- `checkpoints/Test/best_receiver_conditioned_cnn.pth`: best validation checkpoint.
- `checkpoints/Test/last_receiver_conditioned_cnn.pth`: last training checkpoint.
- `checkpoints/Test/receiver_conditioned_cnn_loss_history.npz`: epoch and batch loss history.
- `outputs/Results_Conditioned_Recivers_FourierOnet/<run-folder>/`: loss dashboard, trace-prediction review, spectrogram comparison, receiver-wise metrics, correlation figures, and parameter-profile figures.

The review script also compares correct receiver coordinates against shuffled coordinates. This is a diagnostic check that the trained model uses spatial receiver information rather than merely producing a location-independent trace.

## Reproducibility notes

- Keep the generated HDF5 dataset, its split identifiers, the selected configuration, and the corresponding checkpoint together.
- Do not mix outputs from different runs in the same result folder.
- The quick-test defaults are intentionally too small for scientific claims. Use the larger configuration and an appropriately designed validation protocol for substantive experiments.
- GPU, CUDA, PyTorch, and operating-system differences can affect execution time and numerical details. Record the environment used for each substantial run.

## Project status

This repository accompanies ongoing research. The associated manuscript is not yet published. The code is provided to document and reproduce the computational workflow; published claims, formal citation information, and final archival artifacts will be added after publication.

## Author

Carlos A. Celi<br>
Pontificia Universidad Católica del Ecuador

## License

This project is distributed under the [MIT License](LICENSE).
