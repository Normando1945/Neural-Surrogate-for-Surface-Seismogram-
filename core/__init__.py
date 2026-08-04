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


                     .PY FOR Python package for 2D 
                       wave propagation simulations
                    using Finite Difference Method (FDM)
                             and CNN Training.
"""
from .core_wp_2d_simul import (
    FFt_src,
    animation2D_FDM,
    HDF5SurfaceSeismogramWriter,
    HDF5PyVistaSnapshotWriter,
    SurfaceSeismogramSpectrograms,
    HDF5TimeFrequencyComparator,
    HDF5SurfaceSeismogramTorchDataset,
    HDF5ReceiverQueryTorchDataset,
    TORCH_AVAILABLE,
)

from .velocity_models import VelocityModel2DGenerator
from .layered_velocity_models import LayeredVelocityModel2DGenerator
from .pyvista_wave_propagation_qt_viewer import PyVistaWavePropagationQtViewer
from .PseudoPhysicsInformedTraceLoss import PseudoPhysicsInformedTraceLoss          # nuevo loss, pero no funciono :(
from .MaskedPseudoPhysicsInformedTraceLoss_v2 import MaskedPseudoPhysicsInformedTraceLoss_v2
from .classes_plots_paper import *

__all__ = [
    "FFt_src",
    "animation2D_FDM",
    "HDF5SurfaceSeismogramWriter",
    "HDF5PyVistaSnapshotWriter",
    "SurfaceSeismogramSpectrograms",
    "VelocityModel2DGenerator",
    "LayeredVelocityModel2DGenerator",
    "PyVistaWavePropagationQtViewer",
    "HDF5TimeFrequencyComparator",
    "HDF5SurfaceSeismogramTorchDataset",
    "HDF5ReceiverQueryTorchDataset",
    "PseudoPhysicsInformedTraceLoss",
    "MaskedPseudoPhysicsInformedTraceLoss_v2",
    "Keys_h5_file",
    "PlotEpochLoss",
    "PlotBatchLoss",
    "CFLsimul_plot"
]

if TORCH_AVAILABLE:
    from .core_wp_2d_simul import (
        FourierFeatureReceiverEncoding,
        ReceiverConditionedMLPBlock,
        BaselineReceiverConditionedSeismogramNet,
        BaselineCNNEncoderDecoder,
        BaselineCNNHybridSpectralLoss,
        BaselineCNNPerformanceEvaluator,
        BaselineCNNSpectrogramComparator,
        BaselineCNNBandpassComparator,
        ReceiverConditionedLossHistoryReviewer,
        ReceiverConditionedShuffledCoordsReviewer,
        ReceiverConditionedCNNSpectrogramComparator,
    )
    from .receiver_conditioned_spatial_query_net import (
        ReceiverConditionedSpatialQuerySeismogramNet,
        EnergyWeightedMSELoss,
    )

    __all__ += [
        "FourierFeatureReceiverEncoding",
        "ReceiverConditionedMLPBlock",
        "BaselineReceiverConditionedSeismogramNet",
        "ReceiverConditionedSpatialQuerySeismogramNet",
        "EnergyWeightedMSELoss",
        "BaselineCNNEncoderDecoder",
        "BaselineCNNHybridSpectralLoss",
        "BaselineCNNPerformanceEvaluator",
        "BaselineCNNSpectrogramComparator",
        "BaselineCNNBandpassComparator",
        "ReceiverConditionedLossHistoryReviewer",
        "ReceiverConditionedShuffledCoordsReviewer",
        "ReceiverConditionedCNNSpectrogramComparator",
    ]
