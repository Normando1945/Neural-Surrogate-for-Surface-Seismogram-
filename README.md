# AI Surface Seismogram Emulation: Accelerating Seismic Wave Simulation via Physics-Informed Neural Networks

**Author:** MSc. Ing. Carlos Andrés Celi Sánchez  
**Version:** 0.1.2  
**Status:** Active Research (2D Acoustic Wave Propagation)  
**Python:** ≥3.10

---

> **Disclaimer:** This README was structured and formatted with AI assistance (Claude) for clarity and presentation. 
> However, all scientific methodology, mathematical formulations, experimental design, code implementation, 
> research philosophy, and technical decisions are entirely the original work of the author. 
> The AI provided only editorial organization and writing refinement and code documentation.

---

## Table of Contents

1. [Scientific Motivation](#scientific-motivation)
2. [The Problem & Our Solution](#the-problem--our-solution)
3. [Methodology](#methodology)
   - [Wave Propagation Physics](#wave-propagation-physics)
   - [Data Generation Pipeline](#data-generation-pipeline)
   - [Neural Network Architectures](#neural-network-architectures)
   - [Physics-Informed Loss Functions](#physics-informed-loss-functions)
4. [Tiny Experiments & Validation Protocol](#tiny-experiments--validation-protocol)
5. [Project Status: Where We Are](#project-status-where-we-are)
6. [Future Directions](#future-directions)
7. [Project Organization](#project-organization)
8. [Key Components](#key-components)
9. [Installation & Usage](#installation--usage)

---

## Scientific Motivation

### Background: The Seismic Inverse Problem

Seismic imaging and hazard assessment rely fundamentally on forward modeling—predicting ground motion at surface receivers given a subsurface velocity structure. This forms the basis of:

- **Seismic inversion** (estimating velocity structures from observed seismograms)
- **Earthquake scenario planning** (predicting surface shaking from hypothetical ruptures)
- **Engineering seismology** (designing structures resilient to seismic waves)

Traditionally, this forward problem is solved via **Finite Difference Method (FDM)** solvers that numerically integrate the acoustic wave equation across discretized grids. While physically rigorous, FDM is computationally expensive:

- Single 2D simulation: **seconds to minutes** of CPU/GPU time
- Seismic inversion iterations: **thousands of forward simulations** required
- Parameter space exploration: **prohibitively expensive** for real-time applications

### The Neural Operator Paradigm Shift

Recent advances in **neural operator learning** demonstrate that convolutional neural networks can learn **implicit mappings** from velocity fields to seismic responses with orders-of-magnitude speedup:

- **Inference time:** Milliseconds (vs. seconds for FDM)
- **Generalization:** Trained on one velocity distribution, can predict on unseen models
- **Coupling with inversion:** Enables fast forward models in iterative inversion loops

Our work explores this paradigm, combining **data-driven learning** with **physics-informed constraints** to create fast, accurate seismogram emulators.

---

## The Problem & Our Solution

### The Forward Wave Equation

The acoustic wave equation in a heterogeneous 2D medium:

$$\frac{\partial^2 u}{\partial t^2} = c^2(x,z) \nabla^2 u + f(x,z,t)$$

where:
- $u(x,z,t)$ = pressure wavefield (displacement in acoustic media)
- $c(x,z)$ = velocity field (spatially heterogeneous)
- $\nabla^2 u$ = Laplacian (spatial derivatives)
- $f(x,z,t)$ = source function (Ricker wavelet, typically)

**Challenges:**
1. Requires solving a PDE for every new velocity model
2. Stability constraints (CFL conditions) limit time-step size
3. Spatial discretization (dx, dz ~5-10m) demands fine grids for high frequencies

### Our Approach: Receiver-Conditioned Neural Surrogate

Rather than predicting seismograms for all receivers simultaneously, we train a **receiver-conditioned network**:

$$\hat{\mathbf{s}} = \mathcal{N}_\theta(V, r_i)$$

where:
- $\mathcal{N}_\theta$ = trained neural network with parameters $\theta$
- $V$ = 2D velocity field (input)
- $r_i$ = receiver coordinates (spatial query)
- $\hat{\mathbf{s}}$ = predicted seismogram trace at receiver $i$

**Key innovation:** Encode receiver coordinates using **Fourier features**—this allows the network to learn spatially-varying seismic responses without hardcoding receiver positions.

---

## Methodology

### Wave Propagation Physics

#### FDM Implementation

We solve the acoustic wave equation using **2nd-order central difference schemes**:

$$u_{i,k}^{n+1} = 2u_{i,k}^n - u_{i,k}^{n-1} + r^2(u_{i+1,k}^n + u_{i-1,k}^n + u_{i,k+1}^n + u_{i,k-1}^n - 4u_{i,k}^n)$$

where:
- Stability: $r = c_{\max} \Delta t / \Delta h \leq 1/\sqrt{2}$ (CFL condition)
- $\Delta t$ = temporal step, $\Delta h$ = spatial step
- $c_{\max}$ = maximum velocity in domain

**Source:** Ricker wavelet with peak frequency $f_p$ (typically 10-30 Hz for surface waves)

#### Velocity Models

We explore two velocity model classes:

1. **Simple 2D Models** (Gaussian anomalies):
   $$v(x,z) = v_0 + \sum_i A_i \exp\left(-\frac{(x-x_i)^2 + (z-z_i)^2}{2\sigma^2}\right)$$

2. **Layered Models** (Realistic subsurface):
   - Horizontal layers with irregular interfaces
   - Velocity gradients within layers
   - Anomalies embedded in layered structure

### Data Generation Pipeline

```
Velocity Model → FDM Simulation → Surface Recordings → HDF5 Dataset
      ↓                ↓              ↓                    ↓
  v(x,z)          u(x,z,t)      seismic traces      PyTorch DataLoader
```

#### Stage 1: Velocity Model Generation
- Generate random velocity models from specified distributions
- Store as 2D grids (domain: 1000m × 1000m, resolution: ~5m)

#### Stage 2: FDM Simulation
- Discretize wave equation on regular grid
- Apply absorbing boundaries (PML, currently exploring alternatives)
- Record pressure at surface receivers (0-100 receivers per simulation)
- Duration: typically 2-4 seconds (seismic timescale)

#### Stage 3: Seismogram Extraction & Preprocessing
- Extract surface seismogram matrix: **(receivers × time samples)**
- Normalize by receiver amplitude or energy
- Compute spectrograms: time-frequency decomposition (STFT with 512-sample windows)

#### Stage 4: HDF5 Archival
- Store simulations efficiently in HDF5 format
- Metadata tracking: hyperparameters, model seeds, simulation details
- Enable random access for training without loading entire dataset into memory

**Current dataset scale:**
- Tiny experiments: 61-100 simulations (for sanity checks, overfitting tests)
- Medium scale: 500-1000 simulations (typical training)
- Receiver count: 5-90 per simulation

### Neural Network Architectures

#### Evolution of Approaches

We've progressively refined our architecture through ablation studies:

**Version 1: Baseline CNN Encoder-Decoder**
```
Velocity Field (2D) → CNN Encoder → Latent Code → CNN Decoder → Seismogram Matrix
```
- **Problem:** No explicit spatial awareness of receiver location
- **Result:** Decent for single-receiver or all-receivers, but doesn't scale to arbitrary receiver sets

**Version 2: Fourier Receiver-Conditioned CNN**
```
Velocity Field → CNN Encoder → Latent (512)
                                   ↓
                          Fourier Feature Encoding of r_i → 32D  
                                   ↓ (Concatenate)
                          Fusion Network (1024D) → MLP Decoder → Seismogram Trace
```

**Current Architecture (Version 3): Receiver-Conditioned Spatial Query Network**

The network learns a spatial basis and queries it via receiver coordinates:

1. **Spatial Encoding** (CNN encoder on velocity field):
   - Input: $V \in \mathbb{R}^{200 \times 200}$ (velocity grid)
   - Output: Feature map $\phi \in \mathbb{R}^{C \times H \times W}$ where $C=256$

2. **Receiver Encoding** (Fourier positional encoding):
   $$\mathbf{e}_i = [\sin(2^0\pi r_i), \cos(2^0\pi r_i), \sin(2^1\pi r_i), \cos(2^1\pi r_i), \ldots]$$
   - 16 frequency bands per coordinate (total 64D)
   - Allows network to learn multi-scale spatial patterns

3. **Spatial Query** (bilinear interpolation in feature space):
   - Query encoded features at receiver location
   - Fuse with encoded receiver position

4. **Decoding** (multi-layer MLP):
   - Latent fusion (1024D) → Hidden layers (2048D) → Seismogram trace (2000+ time samples)
   - Dropout 0.05 for regularization

#### Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| **Fourier encoding** | Captures multi-scale spatial patterns without hardcoding coordinates |
| **Receiver conditioning** | Enables efficient batching across variable receiver counts |
| **Separate encoders** | Decouples model property learning from spatial query mechanics |
| **Bilinear interpolation** | Smooth spatial queries, differentiable |
| **High-dim latent (1024)** | Necessary for accurate time-series (2000+ time samples) |

### Physics-Informed Loss Functions

#### Standard MSE Loss (Baseline)

$$\mathcal{L}_{MSE} = \frac{1}{N}\sum_i ||s_i - \hat{s}_i||_2^2$$

- **Problem:** Weights all time samples equally, but physics priorities are different
- **Issues:** Models often underpredict high-frequency content and late arrivals

#### Version 1: Pseudo-Physics-Informed Trace Loss

We incorporate physical constraints at the trace level:

$$\mathcal{L}_{physics} = \mathcal{L}_{MSE} + \lambda_1 \mathcal{L}_{amplitude} + \lambda_2 \mathcal{L}_{energy}$$

where:

- **Amplitude term:** Penalizes predicted peak displacement deviations
  $$\mathcal{L}_{amplitude} = ||A(s) - A(\hat{s})||_2$$

- **Energy term:** Penalizes integral of squared amplitude (seismic energy)
  $$\mathcal{L}_{energy} = ||\int s^2 dt - \int \hat{s}^2 dt||_2$$

- **Weights:** $\lambda_1 = 0.1, \lambda_2 = 0.2$ (empirically tuned)

**Result:** Improved phase accuracy but still struggles with late arrivals and duration prediction.

#### Version 2: Masked Pseudo-Physics-Informed Trace Loss (Current Standard) ✓

Our **latest innovation** addresses duration and arrival-time misprediction:

$$\mathcal{L}_{masked} = \mathcal{L}_{MSE} + \lambda_1 \mathcal{L}_{active} + \lambda_2 \mathcal{L}_{onset} + \lambda_3 \mathcal{L}_{duration}$$

**Active Region Masking:**
- Identify where signal RMS exceeds threshold (0.50)
- Only apply physics constraints to *active windows* (where seismic energy concentrated)
- Reduces weight on noise regions

**Onset Detection:**
- Penalize arrival time mismatch (when signal first becomes active)
- Uses gradient-based first-motion picking

**Duration Constraint:**
- Enforce that predicted signal duration ≈ true signal duration
- Prevents model from "stretching" or "compressing" events

**Result:** Significantly improved generalization and arrival-time prediction on unseen velocity models.

---

## Tiny Experiments & Validation Protocol

### Philosophy: "Fail Fast, Learn Quickly"

Rather than immediately scaling to thousands of simulations, we employ a **tiered validation strategy** using small, controlled experiments. This approach:

1. **Verifies code correctness** without expensive computations
2. **Validates architectural choices** through targeted ablations
3. **Tests new loss functions** before committing to large-scale training
4. **Detects generalization issues** early
5. **Provides rapid feedback** on research hypotheses

### Tiny Experiment Categories

#### 1. Overfitting Tests (Validation on Training Data)

**Purpose:** Verify that the neural architecture can *learn* the mapping at all

**Configuration:**
- **Dataset:** 61-100 simulations with random seed
- **Velocity models:** Simple Gaussian anomalies or layered structures
- **Receivers:** Fixed count (typically 5, 10, or 90)
- **Training duration:** 300 epochs with early stopping
- **Batch size:** 8-16
- **Success metric:** Achieve <5% relative error on training set

**Example Experiment 5.3.1:**
```
Tiny Overfit with Layered Models
├── Simulations: 61 (small but diverse)
├── Receivers per sim: 90 (tests receiver conditioning)
├── Velocity models: Layered (realistic subsurface)
├── Loss: MaskedPseudoPhysicsInformedTraceLoss_v2 (latest version)
├── Epochs: 300
├── Batch: 8
└── Goal: "If model can't overfit, architecture is broken"
```

**Expected behavior:**
- Training loss decreases smoothly
- Loss curves show clear learning signal
- Final RMS error < 5% on training set
- If this fails → fundamental issue with architecture or loss function

#### 2. Generalization Tests (Train/Validation Split)

**Purpose:** Verify that learned patterns generalize to *unseen* velocity models

**Configuration:**
- **Total simulations:** 61 split into train (45) + validation (16)
- **Velocity models:** Generated from same distribution but different random seeds
- **Success metric:** Validation error ≤ 2× training error (no catastrophic overfitting)

**Example workflow:**
```python
# Tiny generalization test
train_dataset = load_simulations(seed=42, count=45)
val_dataset = load_simulations(seed=43, count=16)  # Different seed!

# Train on tiny set
model = train(train_dataset, epochs=300)

# Evaluate on validation (unseen velocity models)
train_error = evaluate(model, train_dataset)
val_error = evaluate(model, val_dataset)

# Success: val_error / train_error < 2.0
assert val_error / train_error < 2.0, "Severe overfitting detected"
```

#### 3. Loss Function Ablation (Isolated Tests)

**Purpose:** Validate that physics constraints actually help (or hurt)

**Configuration:**
- **Same 61-simulation dataset**
- **Train multiple models:**
  - Model A: MSE loss only
  - Model B: Pseudo-Physics loss (v1)
  - Model C: Masked Pseudo-Physics loss (v2)
- **Compare:** Generalization error, arrival-time accuracy, spectral alignment

**Expected results:**
```
Loss Type                    | Train Error | Val Error | Arrival Error
------------------------------|-------------|-----------|---------------
MSE (Baseline)               | 4.2%        | 8.5%      | 120 ms
Pseudo-Physics v1            | 5.1%        | 6.8%      | 95 ms
Masked Pseudo-Physics v2 ✓   | 5.3%        | 5.9%      | 45 ms (better!)
```

Physics constraints should:
- Slightly *increase* training error (more constraints = harder optimization)
- *Decrease* validation error (physics knowledge helps generalization)
- Improve arrival-time prediction

#### 4. Shuffled Coordinate Tests (Spatial Reasoning)

**Purpose:** Verify that Fourier encoding truly learns spatial information

**Configuration:**
- **Train model** on 61 simulations with correct receiver coordinates
- **Test on shuffled coordinates:** Randomly permute receiver positions
- **Expected behavior:**
  - Correct coordinates → low error
  - Shuffled coordinates → high error (confirms model uses spatial info)

**Why this matters:**
```python
# If model ignores receiver position (bad):
pred_correct_coords = model.predict(v, r=[100, 200])  # error: 6%
pred_shuffled_coords = model.predict(v, r=[450, 850]) # error: 6% (same!)
# → Model is just averaging, not learning spatial patterns

# If model respects receiver position (good):
pred_correct_coords = model.predict(v, r=[100, 200])  # error: 6%
pred_shuffled_coords = model.predict(v, r=[450, 850]) # error: 45% (much worse!)
# → Model learned that receiver location matters
```

### Tiny Experiment Checkpoints

We maintain checkpoints from key tiny experiments:

| Experiment | Date | Models | Receivers | Epochs | Loss Function | Status | Notebook |
|-----------|------|--------|-----------|--------|---------------|--------|----------|
| Tiny v1 (simple) | Apr 11 | 61 | 5 | 300 | MSE | ✓ works | 5.0 |
| Tiny v2 (baseline+shuffle) | Apr 12 | 61 | 10 | 300 | MSE | ✓ works | 5.1 |
| Tiny v3 (receiver-conditioned) | May 11 | 61 | 90 | 300 | Pseudo-Physics | ✓ works | 5.2 |
| **Tiny v4 (layered+masked)** | May 13 | 61 | 90 | 300 | Masked v2 | 🔬 in progress | 5.3.1 |

### Interpretation: Reading Tiny Experiment Results

#### Success Indicators ✓

```
Training Curve:
├── Loss decreases smoothly (not noisy)
├── Final training error < 5%
├── No divergence or NaN values
├── Early stopping triggers around epoch 200-250

Generalization:
├── Validation error ≤ 2× training error
├── No sharp divergence after epoch 100
├── Physics constraints improve val error

Spatial Awareness:
├── Correct coords: 4-6% error
├── Shuffled coords: 30-50% error
└── Ratio > 5× indicates spatial learning
```

#### Warning Indicators ⚠️

```
If you see this:
├── Training error plateaus at > 10% → Architecture too simple
├── Validation error > 3× training error → Severe overfitting
├── Loss is noisy/unstable → Learning rate too high
├── NaN after first batch → Numerical instability in loss function
├── Shuffled/correct coords perform similarly → Fourier encoding broken
└── → Stop and debug before scaling to 1000 simulations
```

### Why Tiny Experiments Save Time

**Cost analysis:**
- Tiny (61 sims × 300 epochs): ~1-2 hours wall time
- Medium (500 sims × 300 epochs): ~20 hours wall time
- Failed medium experiment: 20 hours wasted + learned nothing

By validating with tiny experiments first:
1. Catch architectural bugs quickly
2. Confirm loss function improvements *before* committing time
3. Measure generalization risk early
4. Document validated configurations for reproducibility

### Current Tiny Experiment: 5.3.1

**Status:** In progress (as of May 14, 2026)

**Setup:**
```python
# 5.3.1: TinyOverfitLayered_MaskedPseudoPhysicsInformedTraceLoss_nModels_90Receivers

numberModels = 20  # Layered velocity models from dataset (positions 0-19)
receivers_per_sim = 90  # All available receivers, high receiver density
velocity_type = 'layered'  # Realistic geological structures
normalize_model_input = True  # Normalize velocity: mean=3000, std=1500

# Training parameters
batch_size_query = 8  # Receiver-query batch size
n_overfit_steps = 30000  # Total optimization steps (not epochs)
learning_rate = 1.0e-4  # Conservative LR for multi-model tiny overfit
gradient_clip_max_norm = 5.0  # Stability control

# Loss function: MaskedPseudoPhysicsInformedTraceLoss_v2 with:
# α_time=0.05, α_relative=0.30, α_correlation=1.00, α_normalized=0.50,
# α_derivative=0.10, α_energy=0.50, α_arrival=0.00, α_silence=1.00,
# α_partial_time=0.10, α_partial_energy=0.25, α_partial_silence=0.50
active_rms_threshold = 0.50  # Active region detection
```

**Validation checklist:**
- [ ] Loss decreases smoothly with stable gradients
- [ ] Training correlation > 0.95 on tiny subset (20 models × 90 receivers)
- [ ] No NaN/divergence in loss terms
- [ ] Predicted traces align with true seismograms visually
- [ ] Architecture can memorize multi-model layered set

**If successful:** Confirms architecture + loss handle layered models  
**If failed:** Debug model-to-model generalization bottleneck

---

## Project Status: Where We Are

### Current Research Phase: "Validation & Scaling"

As of **May 14, 2026**, we are:

#### ✓ Completed Milestones

1. **Data generation pipeline** (fully functional)
   - FDM solver validated against reference solutions
   - HDF5 dataset creation optimized for large-scale experiments
   - Velocity model generation (simple + layered)

2. **Baseline CNN model** (working)
   - Encoder-decoder architecture functioning
   - MSE loss training stable
   - Can achieve <10% relative error on training set

3. **Receiver-conditioned architecture** (working)
   - Fourier feature encoding implemented
   - Multi-receiver training functional
   - Shows clear spatial awareness (verified via shuffled-coordinate ablation)

4. **Physics-informed loss functions** (two versions)
   - Version 1 (Pseudo-Physics): Implemented, modest improvements
   - Version 2 (Masked Pseudo-Physics): Latest, promising results
   - Ablation studies confirm physics constraints help generalization

5. **Visualization & Analysis Tools**
   - PyVista 3D wavefield viewer
   - Spectrogram comparisons (predicted vs. actual)
   - Loss history tracking and analysis
   - Shuffled-coordinate ablation studies

#### 🔬 Current Experiments

**Experiment 5.3.1:** *Tiny Overfit with Layered Models*
- **Setup:** 61 simulations, 90 receivers/sim, layered velocity models
- **Loss:** MaskedPseudoPhysicsInformedTraceLoss_v2
- **Status:** Training in progress, early results promising
- **Goal:** Validate that masked loss prevents overfitting on new velocity model class

#### ⚠️ Known Limitations & Open Questions

1. **Generalization across velocity distributions**
   - Trained on simple Gaussian anomalies
   - Limited testing on realistic geological structures
   - Layered models still being validated

2. **Receiver array flexibility**
   - Training on fixed receiver counts (5-90)
   - Unclear how model behaves with untrained receiver configurations
   - May need interpolation or more robust conditioning

3. **Frequency content**
   - Current experiments: 10-30 Hz bandwidth
   - Higher frequencies require finer grids (more expensive)
   - Band-limited training may not capture full seismic spectrum

4. **Physics constraint weighting**
   - Loss function weights ($\lambda_1, \lambda_2, \lambda_3$) chosen empirically
   - No systematic ablation on weight sensitivity
   - May need adaptive weighting schemes

5. **Scale to 3D**
   - Current work is 2D acoustic (P-waves only)
   - Extension to 3D or elastic (P+S) waves not yet explored
   - Computational cost will increase significantly

---

## Future Directions

### Short-term (Next 3 months)

1. **Stabilize masked loss function**
   - Complete experiments 5.3.1 (layered models)
   - Quantify generalization gains vs. standard MSE
   - Compare onset/duration metrics on held-out test set

2. **Systematic loss function ablation**
   - Isolate contribution of each physics term
   - Optimize weight hyperparameters ($\lambda_i$)
   - Test adaptive weighting strategies

3. **Extended receiver configurations**
   - Train with varying receiver counts per simulation
   - Test on receiver arrays unseen during training (interpolation/extrapolation)
   - Measure robustness of Fourier encoding

4. **Benchmark against FDM**
   - Direct speed comparisons (neural net vs. FDM solver)
   - Accuracy-speed tradeoff curves
   - Computational cost analysis

### Medium-term (3-6 months) — Paper Focus

1. **Comprehensive validation**
   - Generalization tests on independent velocity distributions
   - Real seismic data comparison (if available)
   - Error quantification (RMS, correlation, spectral metrics)

2. **Physics-informed learning analysis**
   - Visualize learned feature representations
   - Interpret what network learns about wave propagation
   - Theoretical analysis of why physics constraints help

3. **Competing architectures**
   - Operator networks (DeepONet, FNO)
   - Attention-based models for spatial queries
   - Hybrid FDM-neural approaches

4. **Practical applications**
   - Integration with seismic inversion workflows
   - Real-time hazard assessment demo
   - Efficient parameter space exploration

### Long-term (6+ months)

1. **Extension to 3D acoustic**
2. **Elastic wave propagation** (P+S, anisotropy)
3. **Integration with inversion** (inverse problem learning)
4. **Deployment considerations** (ONNX export, inference optimization)

---

## Project Organization

```
ai_surface_seismogram_emulation/
│
├── core/                          # Main Python package
│   ├── __init__.py               # Module exports
│   ├── core_wp_2d_simul.py       # Core FDM + neural network code (6,215 lines)
│   │   ├── FFt_src               # Ricker wavelet source generation
│   │   ├── FDM simulator         # Acoustic wave equation solver
│   │   ├── HDF5 dataset writers  # Data archival
│   │   └── Neural network classes (CNN, MLP, encoders, decoders)
│   │
│   ├── velocity_models.py        # Simple Gaussian anomaly generation
│   ├── layered_velocity_models.py # Realistic layered velocity fields
│   ├── receiver_conditioned_spatial_query_net.py # Latest spatial query architecture
│   ├── PseudoPhysicsInformedTraceLoss.py # Version 1 physics loss
│   ├── MaskedPseudoPhysicsInformedTraceLoss_v2.py # Version 2 (current standard)
│   └── pyvista_wave_propagation_qt_viewer.py # 3D visualization
│
├── notebooks/                     # Jupyter scripts (research workflow)
│   ├── 0.DataGeneratorFromSeed.ipynb       # Generate FDM simulations
│   ├── 1.H5_dataBase_inspection.ipynb      # Validate generated data
│   ├── 2.0/2.1.PrepareDataPyTorch.py      # Create PyTorch datasets
│   ├── 3.0/3.1/3.2.TrainCNN*.py           # Training scripts (CNN variants)
│   ├── 4.0.BaselineCNNPerformance.py      # Performance evaluation
│   ├── 5.0/5.1/5.2/5.3*.py                # Model review & analysis
│   ├── 6.0.Papers_pyVista.py              # Publication-quality visualizations
│   ├── 7.ONNX_safe.py                     # Model export for deployment
│   └── 8.Tensor_visualizer.py             # Debug tensor operations
│
├── data/                          # Raw/processed velocity models & simulations
│   ├── velocity_models/          # Generated velocity field arrays
│   └── seismograms/              # Raw FDM output (HDF5)
│
├── checkpoints/                   # Trained model weights
│   ├── baseline_cnn/
│   ├── receiver_conditioned/
│   └── layered_models/
│
├── outputs/                       # Training results
│   ├── loss_histories/           # .npz files with training curves
│   ├── predictions/              # Model inference outputs
│   └── metrics/                  # Performance evaluation reports
│
├── Papers/                        # Reference literature (12 papers)
│   └── [Neural operators, seismic inversion, FNOs, etc.]
│
├── figures/                       # Architecture diagrams, publication plots
└── setup.py, requirements.txt    # Package metadata & dependencies
```

---

## Key Components

### Core FDM Solver (`core/core_wp_2d_simul.py`)

**FFt_src (Ricker Wavelet)**
```python
source = FFt_src(peak_freq=15.0, duration=2.0, dt=0.001)
# Generates Ricker wavelet:
# r(t) = (1 - 2π²f²t²) exp(-π²f²t²)
```

**Wave Propagation Simulation**
- 2nd order finite difference discretization
- Stability: CFL ≤ 1/√2
- Absorbing boundaries (currently PML-like)
- Efficiency: ~0.5 seconds per simulation (2000m × 2000m, 5m resolution, 4sec duration)

**HDF5 Dataset Creation**
```python
writer = HDF5SurfaceSeismogramWriter("dataset.h5")
writer.add_simulation(
    velocity_field=v,      # shape: (200, 200)
    seismograms=seis,      # shape: (n_receivers, n_time_samples)
    metadata={'seed': 42, 'freq': 15.0, ...}
)
```

### Neural Network Inference

**Receiver-Conditioned Prediction**
```python
from core import ReceiverConditionedSpatialQuerySeismogramNet

net = ReceiverConditionedSpatialQuerySeismogramNet(
    latent_dim=512,
    fusion_dim=1024,
    decoder_hidden=2048
)

# Predict seismogram for receiver at (x=100m, z=0m)
velocity_field = torch.tensor(v, dtype=torch.float32)
receiver_coords = torch.tensor([[100.0, 0.0]], dtype=torch.float32)
predicted_seis = net(velocity_field.unsqueeze(0), receiver_coords)
# Output shape: (1, 2000) — seismogram trace
```

### Physics-Informed Loss

```python
from core import MaskedPseudoPhysicsInformedTraceLoss_v2

loss_fn = MaskedPseudoPhysicsInformedTraceLoss_v2(
    active_rms_threshold=0.50,
    lambda_onset=0.1,
    lambda_duration=0.2
)

loss = loss_fn(predicted_seis, true_seis)
```

---

## Installation & Usage

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 1.13 (for neural networks; gracefully degraded if not installed)
- NumPy ≥ 1.23
- SciPy ≥ 1.9
- h5py ≥ 3.7 (HDF5 support)
- Matplotlib ≥ 3.6 (visualization)

Optional:
- PyVista + PySide6/qtpy (3D interactive visualization)
- ONNX ≥ 1.16 (model export for deployment)

### Installation

```bash
# Clone repository
git clone <repo_url>
cd ai_surface_seismogram_emulation

# Install dependencies (uncomment in requirements.txt first)
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Quick Start: Generate Data

```python
from core import VelocityModel2DGenerator, HDF5SurfaceSeismogramWriter
import numpy as np

# Generate random velocity model
gen = VelocityModel2DGenerator(
    domain_size=1000.0,  # meters
    resolution=5.0,      # meters per cell
    base_velocity=2500.0,
    anomaly_amplitude=500.0
)

v = gen.generate()  # shape: (200, 200)

# Simulate (currently requires manual FDM call)
# See notebooks/0.DataGeneratorFromSeed.ipynb for full example
```

### Training a Receiver-Conditioned Model

See `notebooks/3.2.TrainReceiverConditionedCNN_FourierONet.py` for full example.

```python
# Pseudo-code
dataset = HDF5ReceiverQueryTorchDataset("data/dataset.h5", n_receivers=90)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = ReceiverConditionedSpatialQuerySeismogramNet(...)
loss_fn = MaskedPseudoPhysicsInformedTraceLoss_v2(...)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

for epoch in range(300):
    for batch in dataloader:
        v, r, s_true = batch
        s_pred = model(v, r)
        loss = loss_fn(s_pred, s_true)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## References & Theoretical Foundation

The work builds on:

1. **Neural Operators for PDEs:**
   - Fourier Neural Operators (FNO) — Li et al. (2021)
   - DeepONet — Lu et al. (2021)

2. **Physics-Informed Neural Networks:**
   - PINNs — Raissi et al. (2019)
   - Hybrid data-driven + physics approaches

3. **Seismic Wave Propagation:**
   - Finite difference methods for elastic/acoustic waves
   - Absorbing boundary conditions
   - Spectral analysis of seismic traces

4. **Inverse Problems:**
   - Full-waveform inversion (FWI)
   - Receiver-conditioned learning for spatial queries

---

## Contact & Attribution

**Research:** MSc. Ing. Carlos Andrés Celi Sánchez

This work represents ongoing research in **physics-informed machine learning** for seismic applications. All feedback, questions, and technical discussions are welcome.

---

**Last Updated:** May 14, 2026  
**Status:** Active Research, v0.1.2
