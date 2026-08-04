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
    SurfaceSeismogramSpectrograms,
    HDF5TimeFrequencyComparator,
    HDF5SurfaceSeismogramTorchDataset,
    HDF5ReceiverQueryTorchDataset,
    BaselineCNNEncoderDecoder,
    BaselineCNNHybridSpectralLoss,
    BaselineCNNPerformanceEvaluator,
    BaselineCNNSpectrogramComparator,
    BaselineCNNBandpassComparator,
    
    FourierFeatureReceiverEncoding,
    ReceiverConditionedMLPBlock,
    BaselineReceiverConditionedSeismogramNet,
    
    ReceiverConditionedLossHistoryReviewer,
    
    ReceiverConditionedShuffledCoordsReviewer,
    
    ReceiverConditionedCNNSpectrogramComparator,
    
)

from .velocity_models import (
    VelocityModel2DGenerator,
)

__all__ = [
    "FFt_src",
    "animation2D_FDM",
    "HDF5SurfaceSeismogramWriter",
    "SurfaceSeismogramSpectrograms",
    "VelocityModel2DGenerator",
    "HDF5TimeFrequencyComparator",
    "HDF5SurfaceSeismogramTorchDataset",
    "HDF5ReceiverQueryTorchDataset",
    "BaselineCNNEncoderDecoder",
    "BaselineCNNHybridSpectralLoss",
    "BaselineCNNPerformanceEvaluator",
    "BaselineCNNSpectrogramComparator",
    "BaselineCNNBandpassComparator",
    
    "FourierFeatureReceiverEncoding",
    "ReceiverConditionedMLPBlock",
    "BaselineReceiverConditionedSeismogramNet",
    
    "ReceiverConditionedLossHistoryReviewer",
    
    "ReceiverConditionedShuffledCoordsReviewer",
    
    "ReceiverConditionedCNNSpectrogramComparator",
    
]

__version__ = "0.1.2"