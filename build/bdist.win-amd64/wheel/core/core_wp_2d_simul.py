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


                    .PY FOR CLASS SIMULATION AND AI CLASSES.
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.animation import PillowWriter
from IPython.display import display
from matplotlib.animation import FFMpegWriter
import os
from pathlib import Path
import h5py
import time

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR = None

except Exception as e:
    torch = None
    nn = None
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = e

    class Dataset:
        pass

##############################################################################################################################################
##############################################################################################################################################
#################################################         FFt_src       ######################################################################
##############################################################################################################################################
##############################################################################################################################################

class FFt_src:
    """
    This class processes a source signal using the Fast Fourier Transform (FFT) method.

    Parameters:
    src (numpy array): The source time function.
    dt (float): The time step between samples.
    nt (int): The number of time samples.
    record (str, optional): A label for the record. Default is 'Source Time Function'.

    Attributes:
    signal (numpy array): The source time function.
    dt (float): The time step between samples.
    record (str): A label for the record.
    nt (int): The number of time samples.
    """
    def __init__(self, src, dt, nt, record = 'Source Time Function'):
        self.signal = src
        self.dt = dt
        self.record = record
        self.nt = nt
    
    def fft_src(self):
        """
        Calculates the Fast Fourier Transform (FFT) of the source signal.

        Returns:
        frequencies (numpy array): The array of frequencies.
        fft (numpy array): The array of amplitudes.
        RESULT (pandas DataFrame): A DataFrame containing the frequencies and their corresponding amplitudes.
        """
        signal = self.signal
        dt = self.dt
        nt = self.nt
        
        time = np.linspace(0 * dt, nt * dt, nt)
        fft = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(len(signal), dt)
        idx = np.argsort(frequencies)
        frequencies = frequencies[idx]
        fft = fft[idx]

        mask = frequencies > 0
        positive_frequencies = frequencies[mask]
        positive_amplitudes = fft[len(positive_frequencies)+1:len(fft)]

        frequencies = positive_frequencies
        fft = positive_amplitudes[0:len(frequencies)]

        max_amp_idx = np.argmax(np.abs(fft))
        corresponding_freq = frequencies[max_amp_idx]
        corresponding_amp = np.abs(fft[max_amp_idx])
        F = pd.DataFrame(frequencies, columns= ['Frequencies'])
        FFT = pd.DataFrame(np.abs(fft), columns= ['Amplitudes'])
        RESULT = pd.concat([F, FFT], axis=1, ignore_index=False)

        #_______ Plotting _______
        fig, ax = plt.subplots(2,1, figsize=(16, 9))                                                                 

        ax[0].plot(time, signal, color=(0, 0, 1), marker='o', markersize=0,                               
                    markerfacecolor='w', markeredgewidth=1, linewidth=1, alpha=0.6) # plot source time function
        ax[0].set_title(f'Source Time Function, {self.record}', fontsize=12, color=(0, 0, 1))
        ax[0].set_xlim(time[0], time[-1])
        ax[0].set_xlabel('Time (s)', fontsize=10)
        ax[0].set_ylabel('Amplitude', fontsize=10)
        ax[0].grid(which='both', axis='x', linestyle='--', alpha=0.7)    
        
        ax[1].semilogx(frequencies, np.abs(fft), color=(0, 0, 1), marker='o', markersize=0,                               
                    markerfacecolor='w', markeredgewidth=1, linewidth=1, alpha=0.6) # plot frequency and amplitude                                     
        ax[1].semilogx(corresponding_freq, corresponding_amp, color=(0, 0, 0), marker='o', markersize=5,                  
                    markerfacecolor=(0, 0, 0), markeredgewidth=1, linewidth=1, alpha=1)                                 
        ax[1].text(corresponding_freq*1.05, corresponding_amp, f'{corresponding_freq:.2f} [Hz]',                           
                fontsize=10, color=(0, 0, 0), verticalalignment='bottom')
        ax[1].set_title(f'Frequency and Amplitude [FFT], {self.record}', fontsize=12, color=(0, 0, 1))                                        
        ax[1].set_xlabel('Frequency [Hz]', rotation=0, fontsize=10)                                                            
        ax[1].set_ylabel('Amplitude', rotation=90, fontsize=10)                                                                
        ax[1].set_xlim([min(frequencies), max(frequencies)])                                                                  
        ax[1].grid(which='both', axis='x', linestyle='--', alpha=0.7)                                                     
        plt.yticks([])
        plt.tight_layout()

        return frequencies, fft, RESULT


##############################################################################################################################################
##############################################################################################################################################
#############################################       Finite Difference Method       ###########################################################
##############################################################################################################################################
##############################################################################################################################################
class animation2D_FDM:
    """
    This class simulates and animates 2D acoustic wave propagation
    using the finite difference method.

    In addition to visualization, it now returns a structured results
    dictionary suitable for HDF5 dataset construction.
    """

    def __init__(
        self,
        sample,
        nx,
        nz,
        dx,
        dt,
        nt,
        model_type,        # Velocity model label
        c,                 # 2D array (nz, nx) or scalar
        isx,
        isz,               # source indices
        irx,
        irz,               # receiver indices arrays/lists
        src,               # source time function
        idisp=10,          # update interval
        nop=3,             # 3 or 5 point operators
        fsc=3.0,           # scale factor to plot seismograph
        # cmap=plt.cm.Grays,
        cmap='gray',
        show=True,         # show animation in notebook
        save=False,        # save video
        video_name=None,   # output mp4 name
        fps=10,            # video fps
        dpi=120,           # video dpi
        bitrate=1800,      # video bitrate
        dz=None,           # optional dz for metadata storage
        sample_metadata=None
    ):
        self.sample = sample
        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dx if dz is None else dz
        self.dt = dt
        self.nt = nt
        self.c = c
        self.isx = isx
        self.isz = isz
        self.irx = np.array(irx, dtype=int)
        self.irz = np.array(irz, dtype=int)
        self.src = src
        self.idisp = idisp
        self.nop = nop
        self.fsc = fsc
        self.model_type = model_type
        self.cmap = cmap
        self.show = show
        self.save = save
        self.video_name = video_name
        self.fps = fps
        self.dpi = dpi
        self.bitrate = bitrate
        self.sample_metadata = {} if sample_metadata is None else dict(sample_metadata)

    def animate(self):
        """
        Simulates and animates 2D wave propagation using FDM.

        Returns
        -------
        results : dict
            Dictionary containing:
            - velocity_model
            - surface_seismograms
            - dx, dz, dt
            - nx, nz, nt
            - source_x, source_z
            - receiver_x, receiver_z
            - model_type
            - sample_metadata
        """
        sample = self.sample
        fsc = self.fsc
        nx, nz = self.nx, self.nz
        dx, dz, dt, nt = self.dx, self.dz, self.dt, self.nt
        isx, isz = self.isx, self.isz
        irx, irz = self.irx, self.irz
        src = self.src
        idisp = self.idisp
        nop = self.nop
        show = self.show
        save = self.save
        fps = self.fps
        dpi = self.dpi
        bitrate = self.bitrate

        repo_root = Path(__file__).resolve().parent.parent
        videos_dir = repo_root / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------
        # Initialize fields
        # -------------------------
        p = np.zeros((nz, nx), dtype=float)
        pold = np.zeros((nz, nx), dtype=float)
        pnew = np.zeros((nz, nx), dtype=float)

        pxx = np.zeros((nz, nx), dtype=float)
        pzz = np.zeros((nz, nx), dtype=float)

        # -------------------------
        # Velocity model
        # -------------------------
        if np.isscalar(self.c):
            c = np.full((nz, nx), float(self.c))
        else:
            c = np.array(self.c, dtype=float)
            if c.shape != (nz, nx):
                raise ValueError(f"c must have shape (nz, nx) = ({nz}, {nx}), got {c.shape}")

        # -------------------------
        # Receivers for seismograms
        # -------------------------
        nrec = len(irx)
        seis = np.zeros((nrec, nt), dtype=float)
        ir = np.arange(nrec)

        # -------------------------
        # Courant info
        # -------------------------
        cmax = float(np.max(c))
        print("Courant Criterion eps :")
        print(cmax * dt / dx)

        # -------------------------
        # Plot
        # -------------------------
        v = float(max(abs(np.min(src)), abs(np.max(src)))) if np.size(src) else 1.0
        if v == 0:
            v = 1.0

        t = np.arange(nt) * dt

        fig, (ax0, ax1, ax2) = plt.subplots(
            1, 3,
            figsize=(30, 8),
            gridspec_kw={'width_ratios': [1.7, 1.7, 3]},
            constrained_layout=True
        )

        fig.suptitle(
            f"2D Acoustic Wave Propagation in a Heterogeneous Medium, FINITE DIFFERENCE METHOD, nop = {nop}, Sample = {sample}",
            fontsize=18,
            fontweight='bold',
            color=(0, 0, 1)
        )

        # --- Velocity model ---
        im0 = ax0.imshow(c, cmap='Spectral', aspect="auto")
        ax0.set_title(f'Velocity Model, Model = {self.model_type}')
        ax0.set_xlabel('ix')
        ax0.set_ylabel('iz')
        ax0.text(
            0.01, 0.02, "by Carlos Celi",
            transform=ax0.transAxes,
            fontsize=12,
            fontweight='bold',
            ha='left',
            va='bottom',
            alpha=0.8,
            color=(0, 0, 0)
        )
        cbar0 = fig.colorbar(im0, ax=ax0, pad=0.01, fraction=0.03)
        cbar0.set_label('Velocity')

        # --- Wavefield ---
        im = ax1.imshow(
            pnew,
            interpolation="nearest",
            animated=True,
            vmin=-v,
            vmax=+v,
            cmap=self.cmap,
            origin="upper",
            aspect="auto"
        )

        ax1.scatter(irx, irz, marker='^', s=60, linewidths=1.0, color=(0, 0, 1))
        for k in range(len(irx)):
            ax1.text(
                irx[k], irz[k] * 0.8, f'ST{k+1}',
                ha='center', va='bottom', fontweight='bold', color=(0, 0, 1)
            )

        ax1.scatter([isx], [isz], marker='*', s=150, color=(0, 0, 0))
        ax1.text(
            float(isx) * 1.05, float(isz), 'Source',
            ha='left', va='center', fontweight='bold', color=(0, 0, 0)
        )

        cbar = fig.colorbar(im, ax=ax1, pad=0.01, fraction=0.03)
        cbar.set_label("Pressure Amplitude")

        ax1.set_xlabel("ix")
        ax1.set_ylabel("iz")
        ax1.set_title("2D Wave Propagation")

        # --- Seismograms ---
        offset = fsc * v
        offsets = np.arange(nrec) * offset
        seis_lines = []

        for k in range(nrec):
            ln, = ax2.plot(
                t[:1],
                np.zeros(1) + offsets[k],
                color=(0, 0, 0),
                linewidth=1.0,
                alpha=0.9
            )
            seis_lines.append(ln)

        time_line = ax2.axvline(
            0.0,
            linewidth=3.0,
            color=[0, 0, 1],
            linestyle='--',
            alpha=0.6
        )

        ax2.set_title("Seismograms")
        ax2.set_xlim(t[0], t[-1])
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        ax2.set_yticks(offsets)
        ax2.set_yticklabels([f"ST{k+1}" for k in range(nrec)])
        ax2.grid(True, alpha=0.25)

        # -------------------------
        # Display handle
        # -------------------------
        handle = None
        if show:
            handle = display(fig, display_id=True)

        # -------------------------
        # Video writer
        # -------------------------
        writer = None
        video_path = None

        if save:
            writer = FFMpegWriter(
                fps=fps,
                metadata=dict(artist="MSc. Ing. Carlos Celi"),
                bitrate=bitrate
            )

            if self.video_name is None:
                file_name = f"wave_propagation_2D_FD_{self.model_type}.mp4"
            else:
                file_name = Path(self.video_name).name

            video_path = videos_dir / file_name

        # -------------------------
        # Helper for plot update
        # -------------------------
        def update_plot(it):
            im.set_data(pnew)
            ax1.set_title(f"2D Wave Propagation | it = {it} | max(P) = {pnew.max():.3e}")

            ti = t[:it + 1]
            for k, ln in enumerate(seis_lines):
                ln.set_data(ti, seis[k, :it + 1] + offsets[k])

            time_line.set_xdata([t[it], t[it]])

            if save:
                writer.grab_frame()

            if show:
                handle.update(fig)

        # -------------------------
        # Simulation loop
        # -------------------------
        def run_simulation():
            nonlocal p, pold, pnew, pxx, pzz

            for it in range(nt):

                if nop == 3:
                    # second derivative with respect to x (columns)
                    for i in range(1, nx - 1):
                        pzz[:, i] = p[:, i + 1] - 2.0 * p[:, i] + p[:, i - 1]

                    # second derivative with respect to z (rows)
                    for j in range(1, nz - 1):
                        pxx[j, :] = p[j - 1, :] - 2.0 * p[j, :] + p[j + 1, :]

                elif nop == 5:
                    for i in range(2, nx - 2):
                        pzz[:, i] = (
                            (-1.0 / 12.0) * p[:, i + 2]
                            + (4.0 / 3.0) * p[:, i + 1]
                            - (5.0 / 2.0) * p[:, i]
                            + (4.0 / 3.0) * p[:, i - 1]
                            - (1.0 / 12.0) * p[:, i - 2]
                        )

                    for j in range(2, nz - 2):
                        pxx[j, :] = (
                            (-1.0 / 12.0) * p[j + 2, :]
                            + (4.0 / 3.0) * p[j + 1, :]
                            - (5.0 / 2.0) * p[j, :]
                            + (4.0 / 3.0) * p[j - 1, :]
                            - (1.0 / 12.0) * p[j - 2, :]
                        )
                else:
                    raise ValueError("nop must be 3 or 5")

                # scale by dx^2
                pxx = pxx / (dx ** 2)
                pzz = pzz / (dx ** 2)

                # time extrapolation
                pnew = 2.0 * p - pold + (dt ** 2) * (c ** 2) * (pxx + pzz)

                # source
                pnew[isz, isx] += src[it]

                # seismograms
                seis[ir, it] = pnew[irz[ir], irx[ir]]

                # plot update
                if (it % idisp) == 0:
                    update_plot(it)

                # remap time levels
                pold, p = p, pnew.copy()

                # reset derivatives
                pxx.fill(0.0)
                pzz.fill(0.0)

        # -------------------------
        # Run simulation
        # -------------------------
        if save:
            with writer.saving(fig, str(video_path), dpi=dpi):
                run_simulation()
            os.system(f'xdg-open "{video_path}"')
        else:
            run_simulation()

        # Optional close if no display is needed
        if not show:
            plt.close(fig)

        # -------------------------
        # Structured output for HDF5 writing
        # -------------------------
        results = {
            "velocity_model": c.copy(),
            "surface_seismograms": seis.copy(),
            "dx": float(dx),
            "dz": float(dz),
            "dt": float(dt),
            "nx": int(nx),
            "nz": int(nz),
            "nt": int(nt),
            "source_x": int(isx),
            "source_z": int(isz),
            "receiver_x": irx.copy(),
            "receiver_z": irz.copy(),
            "model_type": str(self.model_type),
            "sample_metadata": dict(self.sample_metadata),
        }

        return results


##############################################################################################################################################
##############################################################################################################################################
########################################################       HD5 Maker       ###############################################################
##############################################################################################################################################
##############################################################################################################################################
class HDF5SurfaceSeismogramWriter:
    """
    Class to create and append samples into a single HDF5 dataset file.

    Expected result dictionary keys from animation2D_FDM.animate():
    - velocity_model
    - surface_seismograms
    - dx, dz, dt
    - nx, nz, nt
    - source_x, source_z
    - receiver_x, receiver_z
    - model_type
    - sample_metadata
    """

    def __init__(self, h5_path, compression="gzip", compression_opts=4, float_dtype=np.float32):
        self.h5_path = Path(h5_path)
        self.compression = compression
        self.compression_opts = compression_opts
        self.float_dtype = float_dtype

        self.expected_sample_metadata = [
            "background_velocity",
            "anomaly_center_x",
            "anomaly_center_z",
            "anomaly_radius_x",
            "anomaly_radius_z",
            "anomaly_velocity_contrast",
        ]

    def _ensure_groups(self, h5f):
        h5f.require_group("inputs")
        h5f.require_group("outputs")
        h5f.require_group("metadata")
        h5f.require_group("splits")

    def _append_expandable_dataset(self, group, name, sample_array, dtype=None):
        sample_array = np.asarray(sample_array, dtype=dtype)

        if name not in group:
            maxshape = (None,) + sample_array.shape
            group.create_dataset(
                name,
                data=sample_array[np.newaxis, ...],
                maxshape=maxshape,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )
        else:
            dset = group[name]
            new_size = dset.shape[0] + 1
            dset.resize((new_size,) + dset.shape[1:])
            dset[-1] = sample_array

    def _append_scalar_dataset(self, group, name, value, dtype=None):
        value_array = np.asarray([value], dtype=dtype)

        if name not in group:
            group.create_dataset(
                name,
                data=value_array,
                maxshape=(None,),
                compression=self.compression,
                compression_opts=self.compression_opts,
            )
        else:
            dset = group[name]
            new_size = dset.shape[0] + 1
            dset.resize((new_size,))
            dset[-1] = value_array[0]

    def _append_string_dataset(self, group, name, value):
        str_dtype = h5py.string_dtype(encoding="utf-8")

        if name not in group:
            group.create_dataset(
                name,
                data=np.array([value], dtype=str_dtype),
                maxshape=(None,),
                dtype=str_dtype,
            )
        else:
            dset = group[name]
            new_size = dset.shape[0] + 1
            dset.resize((new_size,))
            dset[-1] = value

    def _write_constant_array(self, group, name, array_value, dtype=None):
        array_value = np.asarray(array_value, dtype=dtype)

        if name not in group:
            group.create_dataset(name, data=array_value)
        else:
            saved = group[name][...]
            if not np.array_equal(saved, array_value):
                raise ValueError(f"Inconsistent constant metadata detected in '{name}'.")

    def append_sample(self, results):
        """
        Append one simulation result into the HDF5 file.
        """
        velocity_model = np.asarray(results["velocity_model"], dtype=self.float_dtype)
        surface_seismograms = np.asarray(results["surface_seismograms"], dtype=self.float_dtype)

        sample_metadata = results.get("sample_metadata", {})

        with h5py.File(self.h5_path, "a") as h5f:
            self._ensure_groups(h5f)

            g_inputs = h5f["inputs"]
            g_outputs = h5f["outputs"]
            g_metadata = h5f["metadata"]

            # Main datasets
            self._append_expandable_dataset(g_inputs, "velocity_model", velocity_model, dtype=self.float_dtype)
            self._append_expandable_dataset(g_outputs, "surface_seismograms", surface_seismograms, dtype=self.float_dtype)

            # Numerical metadata
            self._append_scalar_dataset(g_metadata, "dx", results["dx"], dtype=np.float64)
            self._append_scalar_dataset(g_metadata, "dz", results["dz"], dtype=np.float64)
            self._append_scalar_dataset(g_metadata, "dt", results["dt"], dtype=np.float64)
            self._append_scalar_dataset(g_metadata, "nx", results["nx"], dtype=np.int32)
            self._append_scalar_dataset(g_metadata, "nz", results["nz"], dtype=np.int32)
            self._append_scalar_dataset(g_metadata, "nt", results["nt"], dtype=np.int32)

            # Source metadata
            self._append_scalar_dataset(g_metadata, "source_x", results["source_x"], dtype=np.int32)
            self._append_scalar_dataset(g_metadata, "source_z", results["source_z"], dtype=np.int32)

            # Receiver metadata (constant across all samples)
            self._write_constant_array(g_metadata, "receiver_x", results["receiver_x"], dtype=np.int32)
            self._write_constant_array(g_metadata, "receiver_z", results["receiver_z"], dtype=np.int32)

            # Model label
            self._append_string_dataset(g_metadata, "model_type", results["model_type"])

            # Expected sample metadata
            for key in self.expected_sample_metadata:
                value = sample_metadata.get(key, np.nan)
                self._append_scalar_dataset(g_metadata, key, value, dtype=np.float64)

    def write_splits(self, train_ids, val_ids, test_ids):
        """
        Save train/validation/test split indices.
        """
        with h5py.File(self.h5_path, "a") as h5f:
            self._ensure_groups(h5f)
            g_splits = h5f["splits"]

            for key in ["train_ids", "val_ids", "test_ids"]:
                if key in g_splits:
                    del g_splits[key]

            g_splits.create_dataset("train_ids", data=np.asarray(train_ids, dtype=np.int32))
            g_splits.create_dataset("val_ids", data=np.asarray(val_ids, dtype=np.int32))
            g_splits.create_dataset("test_ids", data=np.asarray(test_ids, dtype=np.int32))

    def count_samples(self):
        """
        Return the number of samples currently stored.
        """
        if not self.h5_path.exists():
            return 0

        with h5py.File(self.h5_path, "r") as h5f:
            if "inputs" not in h5f or "velocity_model" not in h5f["inputs"]:
                return 0
            return h5f["inputs"]["velocity_model"].shape[0]



##############################################################################################################################################
##############################################################################################################################################
######################################################       Spectrograms       ##############################################################
##############################################################################################################################################
##############################################################################################################################################
class SurfaceSeismogramSpectrograms:
    """
    Inspection class for plotting the spectrogram of each surface seismogram.

    This class is intended only for:
    - data inspection,
    - validation,
    - and qualitative comparison of time-frequency content.

    It is NOT intended for training.

    Parameters
    ----------
    results : dict
        Dictionary returned by animation2D_FDM.animate(), expected to contain:
        - "surface_seismograms"
        - "dt"
        - optionally "receiver_x", "receiver_z", "model_type"

    nrows : int, optional
        Number of subplot rows. Default is 2.

    ncols : int, optional
        Number of subplot columns. Default is 5.

    NFFT : int, optional
        Window length for the spectrogram. Default is 128.

    noverlap : int, optional
        Number of overlapping points between windows. Default is 96.

    cmap : str or colormap, optional
        Colormap for the spectrogram. Default is "viridis".

    db_floor : float, optional
        Minimum dB value to avoid extreme negative values. Default is -120.0.
    """

    def __init__(
        self,
        sample,
        results,
        nrows=2,
        ncols=5,
        NFFT=128,
        noverlap=96,
        cmap="viridis",
        db_floor=-120.0,
        random_seed=None,
    ):
        self.sample = sample
        self.results = results
        self.nrows = nrows
        self.ncols = ncols
        self.NFFT = NFFT
        self.noverlap = noverlap
        self.cmap = cmap
        self.db_floor = db_floor
        self.random_seed = random_seed

        # Required data
        self.seismograms = np.asarray(results["surface_seismograms"], dtype=float)
        self.dt = float(results["dt"])

        # Optional metadata
        self.receiver_x = np.asarray(results.get("receiver_x", []), dtype=int)
        self.receiver_z = np.asarray(results.get("receiver_z", []), dtype=int)
        self.model_type = str(results.get("model_type", "unknown"))

        # Basic checks
        if self.seismograms.ndim != 2:
            raise ValueError(
                f"'surface_seismograms' must have shape [n_receivers, nt], got {self.seismograms.shape}"
            )

        self.nrec, self.nt = self.seismograms.shape
        self.fs = 1.0 / self.dt

        # Original behavior:
        # if self.nrec > self.nrows * self.ncols:
        #     raise ValueError(
        #         f"Number of receivers ({self.nrec}) exceeds available subplots "
        #         f"({self.nrows * self.ncols}). Increase nrows or ncols."
        #     )
        self.max_receivers_to_plot = int(self.nrows * self.ncols)

        if self.max_receivers_to_plot <= 0:
            raise ValueError(
                f"nrows * ncols must be positive, got {self.max_receivers_to_plot}."
            )

        if self.NFFT > self.nt:
            raise ValueError(
                f"NFFT ({self.NFFT}) cannot be greater than nt ({self.nt})."
            )

        if self.noverlap >= self.NFFT:
            raise ValueError(
                f"noverlap ({self.noverlap}) must be smaller than NFFT ({self.NFFT})."
            )

    def _select_receivers_to_plot(self):
        """
        Select receivers to plot.

        If the number of receivers fits in the subplot grid, all receivers are used.
        Otherwise, a random subset is selected.
        """
        if self.nrec <= self.max_receivers_to_plot:
            return list(range(self.nrec))

        rng = np.random.default_rng(self.random_seed)
        selected = rng.choice(
            self.nrec,
            size=self.max_receivers_to_plot,
            replace=False,
        )

        return sorted(selected.tolist())

    def compute_spectrograms(self, receiver_indices=None):
        """
        Compute the spectrogram for each receiver trace.

        Returns
        -------
        spectrogram_results : list of dict
            Each entry contains:
            - "receiver_id"
            - "receiver_x"
            - "receiver_z"
            - "Pxx"
            - "Pxx_dB"
            - "frequencies"
            - "times"
        """
        if receiver_indices is None:
            receiver_indices = list(range(self.nrec))

        spectrogram_results = []
        eps = 1e-20

        # Original behavior:
        # for i in range(self.nrec):
        for i in receiver_indices:
            trace = self.seismograms[i]

            Pxx, frequencies, times = mlab.specgram(
                x=trace,
                NFFT=self.NFFT,
                Fs=self.fs,
                noverlap=self.noverlap,
            )

            Pxx_dB = 10.0 * np.log10(Pxx + eps)
            Pxx_dB = np.maximum(Pxx_dB, self.db_floor)

            rec_x = int(self.receiver_x[i]) if len(self.receiver_x) == self.nrec else None
            rec_z = int(self.receiver_z[i]) if len(self.receiver_z) == self.nrec else None

            spectrogram_results.append(
                {
                    "receiver_id": i + 1,
                    "receiver_x": rec_x,
                    "receiver_z": rec_z,
                    "Pxx": Pxx,
                    "Pxx_dB": Pxx_dB,
                    "frequencies": frequencies,
                    "times": times,
                }
            )

        return spectrogram_results

    def plot_spectrograms(self, fmax=None, common_color_scale=True):
        """
        Plot the spectrograms in a grid of subplots.

        Parameters
        ----------
        fmax : float or None, optional
            Maximum frequency to display. If None, the full range is shown.

        common_color_scale : bool, optional
            If True, use a common color scale for all subplots. Default is True.

        Returns
        -------
        spectrogram_results : list of dict
            Computed spectrogram data for all receivers.
        """
        receiver_indices = self._select_receivers_to_plot()
        spectrogram_results = self.compute_spectrograms(receiver_indices=receiver_indices)

        # Determine common color scale if requested
        if common_color_scale:
            all_db = np.concatenate(
                [item["Pxx_dB"].ravel() for item in spectrogram_results]
            )
            vmin = np.percentile(all_db, 5)
            vmax = np.percentile(all_db, 95)
        else:
            vmin = None
            vmax = None

        fig, axes = plt.subplots(
            self.nrows,
            self.ncols,
            figsize=(30, 8),
            constrained_layout=True
        )

        axes = np.atleast_2d(axes)
        axes_flat = axes.ravel()

        fig.suptitle(
            f"Surface Seismogram Spectrograms | Model = {self.model_type}, ",
            fontsize=18,
            fontweight="bold",
            color=(0, 0, 1),
        )

        last_im = None

        for i, ax in enumerate(axes_flat):
            # Original behavior:
            # if i < self.nrec:
            if i < len(spectrogram_results):
                item = spectrogram_results[i]
                Pxx_dB = item["Pxx_dB"]
                frequencies = item["frequencies"]
                times = item["times"]

                if fmax is not None:
                    mask_f = frequencies <= fmax
                    frequencies_plot = frequencies[mask_f]
                    Pxx_plot = Pxx_dB[mask_f, :]
                else:
                    frequencies_plot = frequencies
                    Pxx_plot = Pxx_dB

                last_im = ax.imshow(
                    Pxx_plot,
                    origin="lower",
                    aspect="auto",
                    extent=[times[0], times[-1], frequencies_plot[0], frequencies_plot[-1]],
                    cmap=self.cmap,
                    vmin=vmin,
                    vmax=vmax,
                )

                rec_id = item["receiver_id"]
                rec_x = item["receiver_x"]
                rec_z = item["receiver_z"]

                if rec_x is not None and rec_z is not None:
                    ax.set_title(
                        f"ST{rec_id} | ix={rec_x}, iz={rec_z}",
                        fontsize=10,
                        fontweight="bold",
                        color=(0, 0, 1),
                    )
                else:
                    ax.set_title(
                        f"ST{rec_id}",
                        fontsize=10,
                        fontweight="bold",
                        color=(0, 0, 1),
                    )

                ax.set_xlabel("Time (s)", fontsize=9)
                ax.set_ylabel("Frequency (Hz)", fontsize=9)

            else:
                ax.axis("off")

        if last_im is not None:
            cbar = fig.colorbar(last_im, ax=axes_flat.tolist(), pad=0.01, fraction=0.02)
            cbar.set_label("Power Spectral Density (dB)")

        plt.show()

        return spectrogram_results



##############################################################################################################################################
##############################################################################################################################################
############################################       HDF5 Time Frequency Comparator       ######################################################
##############################################################################################################################################
##############################################################################################################################################
class HDF5TimeFrequencyComparator:
    """
    Inspection class to compare the time-frequency content of the same receiver
    across several samples stored in an HDF5 database.

    This class is intended only for:
    - data inspection,
    - validation,
    - qualitative comparison of traces and spectrograms.

    It is NOT intended for training.

    Parameters
    ----------
    h5_path : str or Path
        Path to the HDF5 database.

    NFFT : int, optional
        Window length for spectrogram computation. Default is 128.

    noverlap : int, optional
        Number of overlapping points between windows. Default is 96.

    cmap : str or colormap, optional
        Colormap for the spectrograms. Default is "viridis".

    db_floor : float, optional
        Minimum dB value to avoid extreme negative values. Default is -120.0.
    """

    def __init__(
        self,
        h5_path,
        NFFT=128,
        noverlap=96,
        cmap="viridis",
        db_floor=-120.0,
    ):
        self.h5_path = Path(h5_path)
        self.NFFT = NFFT
        self.noverlap = noverlap
        self.cmap = cmap
        self.db_floor = db_floor

        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

    # ==========================================================================
    # Internal helpers
    # ==========================================================================
    def _decode_string(self, value):
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def _compute_basic_metrics(self, trace, dt):
        trace = np.asarray(trace, dtype=float)

        peak_idx = int(np.argmax(np.abs(trace)))
        peak_arrival_time = peak_idx * dt
        peak_amplitude = float(np.max(np.abs(trace)))
        total_energy = float(np.sum(trace**2) * dt)

        fft_vals = np.fft.rfft(trace)
        freqs = np.fft.rfftfreq(len(trace), d=dt)

        if len(freqs) > 1:
            dominant_idx = np.argmax(np.abs(fft_vals[1:])) + 1
            dominant_frequency = float(freqs[dominant_idx])
        else:
            dominant_frequency = 0.0

        return {
            "peak_arrival_time_s": peak_arrival_time,
            "peak_amplitude": peak_amplitude,
            "dominant_frequency_Hz": dominant_frequency,
            "total_energy": total_energy,
        }

    def _compute_spectrogram(self, trace, dt):
        eps = 1e-20
        fs = 1.0 / dt

        Pxx, frequencies, times = mlab.specgram(
            x=trace,
            NFFT=self.NFFT,
            Fs=fs,
            noverlap=self.noverlap,
        )

        Pxx_dB = 10.0 * np.log10(Pxx + eps)
        Pxx_dB = np.maximum(Pxx_dB, self.db_floor)

        return Pxx, Pxx_dB, frequencies, times

    def _read_sample_receiver(self, sample_id, receiver_id):
        with h5py.File(self.h5_path, "r") as h5f:
            n_samples = h5f["inputs/velocity_model"].shape[0]
            n_receivers = h5f["outputs/surface_seismograms"].shape[1]

            if sample_id < 0 or sample_id >= n_samples:
                raise IndexError(f"sample_id={sample_id} is out of range [0, {n_samples-1}]")

            if receiver_id < 0 or receiver_id >= n_receivers:
                raise IndexError(f"receiver_id={receiver_id} is out of range [0, {n_receivers-1}]")

            velocity_model = h5f["inputs/velocity_model"][sample_id].astype(float)
            trace = h5f["outputs/surface_seismograms"][sample_id, receiver_id].astype(float)

            dt = float(h5f["metadata/dt"][sample_id])
            dx = float(h5f["metadata/dx"][sample_id])
            dz = float(h5f["metadata/dz"][sample_id])

            model_type = self._decode_string(h5f["metadata/model_type"][sample_id])

            receiver_x = int(h5f["metadata/receiver_x"][receiver_id])
            receiver_z = int(h5f["metadata/receiver_z"][receiver_id])

            source_x = int(h5f["metadata/source_x"][sample_id])
            source_z = int(h5f["metadata/source_z"][sample_id])

            background_velocity = float(h5f["metadata/background_velocity"][sample_id])
            anomaly_center_x = float(h5f["metadata/anomaly_center_x"][sample_id])
            anomaly_center_z = float(h5f["metadata/anomaly_center_z"][sample_id])
            anomaly_radius_x = float(h5f["metadata/anomaly_radius_x"][sample_id])
            anomaly_radius_z = float(h5f["metadata/anomaly_radius_z"][sample_id])
            anomaly_velocity_contrast = float(h5f["metadata/anomaly_velocity_contrast"][sample_id])

        return {
            "sample_id": sample_id,
            "receiver_id": receiver_id,
            "velocity_model": velocity_model,
            "trace": trace,
            "dt": dt,
            "dx": dx,
            "dz": dz,
            "model_type": model_type,
            "receiver_x": receiver_x,
            "receiver_z": receiver_z,
            "source_x": source_x,
            "source_z": source_z,
            "background_velocity": background_velocity,
            "anomaly_center_x": anomaly_center_x,
            "anomaly_center_z": anomaly_center_z,
            "anomaly_radius_x": anomaly_radius_x,
            "anomaly_radius_z": anomaly_radius_z,
            "anomaly_velocity_contrast": anomaly_velocity_contrast,
        }

    # ==========================================================================
    # Public method
    # ==========================================================================
    def compare_receiver(
        self,
        sample_ids,
        receiver_id,
        fmax=None,
        common_color_scale=True,
        show_velocity_models=True,
    ):
        """
        Compare the same receiver across several samples.

        Parameters
        ----------
        sample_ids : list[int]
            Sample indices to compare.

        receiver_id : int
            Receiver index to compare (0-based index).

        fmax : float or None, optional
            Maximum frequency to display in the spectrograms.

        common_color_scale : bool, optional
            If True, all spectrograms use the same color scale.

        show_velocity_models : bool, optional
            If True, show the velocity models corresponding to the selected samples.

        Returns
        -------
        metrics_df : pandas.DataFrame
            Table containing simple metrics for each selected sample.
        """
        sample_ids = list(sample_ids)
        loaded = [self._read_sample_receiver(sid, receiver_id) for sid in sample_ids]

        # ----------------------------------------------------------------------
        # 1. Velocity models
        # ----------------------------------------------------------------------
        if show_velocity_models:
            fig, axes = plt.subplots(
                1,
                len(loaded),
                figsize=(6 * len(loaded), 5),
                constrained_layout=True
            )

            if len(loaded) == 1:
                axes = [axes]

            for ax, item in zip(axes, loaded):
                im = ax.imshow(item["velocity_model"], cmap="Spectral", aspect="auto")
                ax.set_title(
                    f"Sample {item['sample_id']}\nModel = {item['model_type']}",
                    fontsize=11,
                    fontweight="bold",
                    color=(0, 0, 1),
                )
                ax.set_xlabel("ix")
                ax.set_ylabel("iz")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            plt.show()

        # ----------------------------------------------------------------------
        # 2. Overlay traces
        # ----------------------------------------------------------------------
        plt.figure(figsize=(14, 6))

        metrics_rows = []

        for item in loaded:
            trace = item["trace"]
            dt = item["dt"]
            t = np.arange(len(trace)) * dt

            metrics = self._compute_basic_metrics(trace, dt)

            plt.plot(
                t,
                trace,
                linewidth=1.5,
                label=(
                    f"sample {item['sample_id']} | "
                    f"ix={item['receiver_x']}, iz={item['receiver_z']}"
                ),
            )

            metrics_rows.append(
                {
                    "sample_id": item["sample_id"],
                    "receiver_id": item["receiver_id"] + 1,
                    "receiver_x": item["receiver_x"],
                    "receiver_z": item["receiver_z"],
                    "model_type": item["model_type"],
                    "peak_arrival_time_s": metrics["peak_arrival_time_s"],
                    "peak_amplitude": metrics["peak_amplitude"],
                    "dominant_frequency_Hz": metrics["dominant_frequency_Hz"],
                    "total_energy": metrics["total_energy"],
                    "background_velocity": item["background_velocity"],
                    "anomaly_center_x": item["anomaly_center_x"],
                    "anomaly_center_z": item["anomaly_center_z"],
                    "anomaly_radius_x": item["anomaly_radius_x"],
                    "anomaly_radius_z": item["anomaly_radius_z"],
                    "anomaly_velocity_contrast": item["anomaly_velocity_contrast"],
                }
            )

        plt.title(
            f"Trace Comparison | Receiver ST{receiver_id + 1}",
            fontsize=14,
            fontweight="bold",
            color=(0, 0, 1),
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # ----------------------------------------------------------------------
        # 3. Spectrograms - side by side
        # ----------------------------------------------------------------------
        spec_data = []
        for item in loaded:
            _, Pxx_dB, frequencies, times = self._compute_spectrogram(item["trace"], item["dt"])

            if fmax is not None:
                mask_f = frequencies <= fmax
                frequencies_plot = frequencies[mask_f]
                Pxx_plot = Pxx_dB[mask_f, :]
            else:
                frequencies_plot = frequencies
                Pxx_plot = Pxx_dB

            spec_data.append(
                {
                    "sample_id": item["sample_id"],
                    "Pxx_dB": Pxx_plot,
                    "frequencies": frequencies_plot,
                    "times": times,
                }
            )

        if common_color_scale:
            all_db = np.concatenate([d["Pxx_dB"].ravel() for d in spec_data])
            vmin = np.percentile(all_db, 5)
            vmax = np.percentile(all_db, 95)
        else:
            vmin = None
            vmax = None

        fig, axes = plt.subplots(
            1,
            len(spec_data),
            figsize=(6 * len(spec_data), 4.8),
            constrained_layout=True
        )

        if len(spec_data) == 1:
            axes = [axes]

        last_im = None

        for ax, item_spec, item_loaded in zip(axes, spec_data, loaded):
            last_im = ax.imshow(
                item_spec["Pxx_dB"],
                origin="lower",
                aspect="auto",
                extent=[
                    item_spec["times"][0],
                    item_spec["times"][-1],
                    item_spec["frequencies"][0],
                    item_spec["frequencies"][-1],
                ],
                cmap=self.cmap,
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_title(
                f"Spectrogram | sample {item_loaded['sample_id']} | ST{receiver_id + 1}",
                fontsize=11,
                fontweight="bold",
                color=(0, 0, 1),
            )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")

        if last_im is not None:
            cbar = fig.colorbar(last_im, ax=axes, pad=0.01, fraction=0.02)
            cbar.set_label("Power Spectral Density (dB)")

        plt.show()

        # ----------------------------------------------------------------------
        # 4. Metrics table
        # ----------------------------------------------------------------------
        metrics_df = pd.DataFrame(metrics_rows)

        print("\n==============================================================")
        print("Time-frequency comparison metrics")
        print("==============================================================\n")
        print(metrics_df)

        return metrics_df


##############################################################################################################################################
##############################################################################################################################################
############################################       Torch HDF5 Dataset       ##################################################################
##############################################################################################################################################
##############################################################################################################################################
class HDF5SurfaceSeismogramTorchDataset(Dataset):
    """
    PyTorch Dataset for training on the HDF5 database.

    Input
    -----
    X = velocity_model

    Target
    ------
    Y = surface_seismograms

    Available splits
    ----------------
    - train
    - val
    - test
    """

    def __init__(
        self,
        h5_path,
        split="train",
        normalize_x=False,
        normalize_y=False,
        return_metadata=False,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                f"PyTorch is not available in this environment. Original error: {TORCH_IMPORT_ERROR}"
            )
        
        self.h5_path = Path(h5_path)
        self.split = split
        self.normalize_x = normalize_x
        self.normalize_y = normalize_y
        self.return_metadata = return_metadata

        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        if self.split not in ["train", "val", "test"]:
            raise ValueError("split must be one of: 'train', 'val', or 'test'.")

        with h5py.File(self.h5_path, "r") as h5f:
            split_key = f"splits/{self.split}_ids"
            self.sample_ids = h5f[split_key][:].astype(np.int64)

            self.n_total = h5f["inputs/velocity_model"].shape[0]
            self.x_shape = h5f["inputs/velocity_model"].shape[1:]        # (nz, nx)
            self.y_shape = h5f["outputs/surface_seismograms"].shape[1:]  # (nrec, nt)

            if self.normalize_x:
                x_all = h5f["inputs/velocity_model"][:].astype(np.float32)
                self.x_mean = float(x_all.mean())
                self.x_std = float(x_all.std() + 1e-8)
            else:
                self.x_mean = None
                self.x_std = None

            if self.normalize_y:
                y_all = h5f["outputs/surface_seismograms"][:].astype(np.float32)
                self.y_mean = float(y_all.mean())
                self.y_std = float(y_all.std() + 1e-8)
            else:
                self.y_mean = None
                self.y_std = None

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = int(self.sample_ids[idx])

        with h5py.File(self.h5_path, "r") as h5f:
            x = h5f["inputs/velocity_model"][sample_id].astype(np.float32)        # (nz, nx)
            y = h5f["outputs/surface_seismograms"][sample_id].astype(np.float32)  # (nrec, nt)

            # Add channel dimension for CNN input
            x = np.expand_dims(x, axis=0)  # (1, nz, nx)

            if self.normalize_x:
                x = (x - self.x_mean) / self.x_std

            if self.normalize_y:
                y = (y - self.y_mean) / self.y_std

            x_tensor = torch.from_numpy(x)  # (1, nz, nx)
            y_tensor = torch.from_numpy(y)  # (nrec, nt)

            if not self.return_metadata:
                return x_tensor, y_tensor

            metadata = {
                "sample_id": sample_id,
                "dt": float(h5f["metadata/dt"][sample_id]),
                "dx": float(h5f["metadata/dx"][sample_id]),
                "dz": float(h5f["metadata/dz"][sample_id]),
                "source_x": int(h5f["metadata/source_x"][sample_id]),
                "source_z": int(h5f["metadata/source_z"][sample_id]),
                "receiver_x": h5f["metadata/receiver_x"][:].astype(np.int32),
                "receiver_z": h5f["metadata/receiver_z"][:].astype(np.int32),
                "model_type": h5f["metadata/model_type"][sample_id],
            }

            return x_tensor, y_tensor, metadata



##############################################################################################################################################
##############################################################################################################################################
############################################       Torch HDF5 Dataset "Query"       ##########################################################
##############################################################################################################################################
##############################################################################################################################################
class HDF5ReceiverQueryTorchDataset(Dataset):                                                # Dataset for receiver-conditioned learning
    """
    PyTorch Dataset for receiver-query learning on the HDF5 database.

    Input
    -----
    X1 = velocity_model
    X2 = receiver coordinates

    Target
    ------
    Y = one receiver trace

    Philosophy
    ----------
    This dataset reformulates each simulation sample with many receivers
    into many query pairs:

        (sample_id, receiver_id) -> one trace

    Where:
    - the source remains fixed,
    - the source signature remains fixed,
    - the receiver bank is dense and fixed in the HDF5,
    - and the learning problem becomes receiver-conditioned.

    Available splits
    ----------------
    - train
    - val
    - test
    """

    def __init__(
        self,
        h5_path,
        split="train",
        receiver_ids=None,
        normalize_x=False,
        normalize_y=False,
        normalize_receiver_coords=True,
        return_metadata=False,
    ):
        if not TORCH_AVAILABLE:                                                               # Stop immediately if PyTorch is unavailable
            raise ImportError(
                f"PyTorch is not available in this environment. Original error: {TORCH_IMPORT_ERROR}"
            )

        self.h5_path = Path(h5_path)                                                          # Store HDF5 file path
        self.split = split                                                                    # Store dataset split name
        self.receiver_ids = receiver_ids                                                      # Store optional receiver subset
        self.normalize_x = normalize_x                                                        # Store velocity-model normalization flag
        self.normalize_y = normalize_y                                                        # Store trace normalization flag
        self.normalize_receiver_coords = normalize_receiver_coords                            # Store receiver-coordinate normalization flag
        self.return_metadata = return_metadata                                                # Store metadata return flag

        if not self.h5_path.exists():                                                         # Check that the HDF5 file exists
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        if self.split not in ["train", "val", "test"]:                                        # Validate split name
            raise ValueError("split must be one of: 'train', 'val', or 'test'.")

        with h5py.File(self.h5_path, "r") as h5f:                                             # Open HDF5 file for dataset initialization
            split_key = f"splits/{self.split}_ids"                                            # Build split key inside HDF5
            self.sample_ids = h5f[split_key][:].astype(np.int64)                              # Read sample ids for the selected split

            self.n_total = int(h5f["inputs/velocity_model"].shape[0])                         # Total number of simulation samples in HDF5
            self.x_shape = h5f["inputs/velocity_model"].shape[1:]                             # Velocity-model shape = (nz, nx)
            self.y_shape = h5f["outputs/surface_seismograms"].shape[1:]                       # Full seismogram shape = (nrec, nt)

            self.nz = int(self.x_shape[0])                                                    # Number of grid points in z direction
            self.nx = int(self.x_shape[1])                                                    # Number of grid points in x direction
            self.n_receivers_total = int(self.y_shape[0])                                     # Total number of receivers stored in HDF5
            self.n_time = int(self.y_shape[1])                                                # Number of time samples per trace

            self.receiver_x_all = h5f["metadata/receiver_x"][:].astype(np.int32)              # Read all receiver x coordinates
            self.receiver_z_all = h5f["metadata/receiver_z"][:].astype(np.int32)              # Read all receiver z coordinates

            if receiver_ids is None:                                                          # Use all receivers if no subset was provided
                self.receiver_ids = np.arange(self.n_receivers_total, dtype=np.int64)         # Build full receiver index array
            else:
                self.receiver_ids = np.asarray(receiver_ids, dtype=np.int64)                  # Convert user-defined receiver subset to array

                if self.receiver_ids.ndim != 1:                                               # Ensure receiver subset is one-dimensional
                    raise ValueError("receiver_ids must be a one-dimensional list or array.")

                if len(self.receiver_ids) == 0:                                               # Prevent empty receiver subset
                    raise ValueError("receiver_ids cannot be empty.")

                if np.any(self.receiver_ids < 0):                                             # Prevent negative receiver indices
                    raise ValueError("receiver_ids cannot contain negative indices.")

                if np.any(self.receiver_ids >= self.n_receivers_total):                       # Prevent out-of-range receiver indices
                    raise ValueError(
                        f"receiver_ids must be in the range [0, {self.n_receivers_total - 1}]."
                    )

            if self.normalize_x:                                                              # Compute global velocity-model statistics if requested
                x_all = h5f["inputs/velocity_model"][:].astype(np.float32)                    # Load all velocity models
                self.x_mean = float(x_all.mean())                                             # Store global mean of velocity models
                self.x_std = float(x_all.std() + 1e-8)                                        # Store global (standard deviation)std of velocity models
            else:
                self.x_mean = None                                                            # No x normalization mean
                self.x_std = None                                                             # No x normalization standard deviation (std)

            if self.normalize_y:                                                              # Compute global trace statistics if requested
                y_all = h5f["outputs/surface_seismograms"][:].astype(np.float32)             # Load all seismograms
                self.y_mean = float(y_all.mean())                                             # Store global mean of traces
                self.y_std = float(y_all.std() + 1e-8)                                        # Store global std of traces
            else:
                self.y_mean = None                                                            # No y normalization mean
                self.y_std = None                                                             # No y normalization std

        self.query_pairs = [                                                                  # Build all (sample_id, receiver_id) query pairs
            (int(sample_id), int(receiver_id))                                                # One query corresponds to one sample and one receiver
            for sample_id in self.sample_ids                                                  # Loop over all split samples
            for receiver_id in self.receiver_ids                                              # Loop over all selected receivers
        ]

        self.n_queries = len(self.query_pairs)                                                # Store total number of receiver queries
        self.n_selected_receivers = int(len(self.receiver_ids))                               # Store number of selected receivers

    def __len__(self):
        return self.n_queries                                                                 # Return total number of receiver queries

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.n_queries:                                                  # Validate query index
            raise IndexError(f"idx={idx} is out of range [0, {self.n_queries - 1}]")

        sample_id, receiver_id = self.query_pairs[idx]                                        # Recover sample and receiver ids for this query

        with h5py.File(self.h5_path, "r") as h5f:                                             # Open HDF5 file to read one query
            x = h5f["inputs/velocity_model"][sample_id].astype(np.float32)                    # Read one velocity model with shape (nz, nx)
            y_trace = h5f["outputs/surface_seismograms"][sample_id, receiver_id].astype(np.float32)  # Read one trace with shape (nt,)

            dt = float(h5f["metadata/dt"][sample_id])                                         # Read dt for this simulation sample
            dx = float(h5f["metadata/dx"][sample_id])                                         # Read dx for this simulation sample
            dz = float(h5f["metadata/dz"][sample_id])                                         # Read dz for this simulation sample
            source_x = int(h5f["metadata/source_x"][sample_id])                               # Read source x index
            source_z = int(h5f["metadata/source_z"][sample_id])                               # Read source z index
            model_type = h5f["metadata/model_type"][sample_id]                                # Read model label

        receiver_x = int(self.receiver_x_all[receiver_id])                                    # Recover receiver x coordinate from constant receiver bank
        receiver_z = int(self.receiver_z_all[receiver_id])                                    # Recover receiver z coordinate from constant receiver bank

        receiver_coord_raw = np.array([receiver_x, receiver_z], dtype=np.float32)             # Build raw receiver coordinate vector

        if self.normalize_receiver_coords:                                                    # Normalize receiver coordinates if requested
            receiver_coord = np.array(
                [
                    receiver_x / max(1, self.nx - 1),                                         # Normalize x coordinate to approximately [0, 1]
                    receiver_z / max(1, self.nz - 1),                                         # Normalize z coordinate to approximately [0, 1]
                ],
                dtype=np.float32,
            )
        else:
            receiver_coord = receiver_coord_raw.copy()                                        # Keep raw receiver coordinates without normalization

        x = np.expand_dims(x, axis=0)                                                         # Add channel dimension to velocity model -> (1, nz, nx)

        if self.normalize_x:                                                                  # Normalize velocity model if requested
            x = (x - self.x_mean) / self.x_std                                                # Apply global x normalization

        if self.normalize_y:                                                                  # Normalize output trace if requested
            y_trace = (y_trace - self.y_mean) / self.y_std                                    # Apply global y normalization

        x_tensor = torch.from_numpy(x)                                                        # Convert velocity model to torch tensor
        receiver_coord_tensor = torch.from_numpy(receiver_coord)                              # Convert receiver coordinates to torch tensor
        y_trace_tensor = torch.from_numpy(y_trace)                                            # Convert target trace to torch tensor

        if not self.return_metadata:                                                          # Return only tensors if metadata is not requested
            return x_tensor, receiver_coord_tensor, y_trace_tensor

        metadata = {
            "query_index": int(idx),                                                          # Store absolute query index inside this dataset
            "sample_id": int(sample_id),                                                      # Store mapped sample id from HDF5
            "receiver_id": int(receiver_id),                                                  # Store receiver id using 0-based indexing
            "receiver_id_1_based": int(receiver_id + 1),                                      # Store receiver id using 1-based indexing for plots
            "receiver_x": int(receiver_x),                                                    # Store receiver x coordinate
            "receiver_z": int(receiver_z),                                                    # Store receiver z coordinate
            "receiver_coord_raw": receiver_coord_raw.copy(),                                  # Store raw receiver coordinate vector
            "receiver_coord_used": receiver_coord.copy(),                                     # Store coordinate vector actually used by the model
            "dt": float(dt),                                                                  # Store dt
            "dx": float(dx),                                                                  # Store dx
            "dz": float(dz),                                                                  # Store dz
            "source_x": int(source_x),                                                        # Store source x coordinate
            "source_z": int(source_z),                                                        # Store source z coordinate
            "model_type": model_type,                                                         # Store model label
            "nx": int(self.nx),                                                               # Store nx
            "nz": int(self.nz),                                                               # Store nz
            "nt": int(self.n_time),                                                           # Store nt
        }

        return x_tensor, receiver_coord_tensor, y_trace_tensor, metadata                      # Return tensors plus metadata



##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
#############################################       Training with Fourier NET Encoding       #################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
if TORCH_AVAILABLE:                                                                              # Define the model classes only when PyTorch is available

    class FourierFeatureReceiverEncoding(nn.Module):                                             # Encode receiver coordinates using Fourier features
        """
        Fourier feature encoder for receiver coordinates.
        
        What this class does
        --------------------
        This class transforms the receiver coordinates into a richer representation before they are used by the main network.
        In this workflow, the learning problem is:
        
            velocity_model + receiver_coordinates -> one trace
            
        Because of that, the receiver position becomes an important input. If we use only the raw coordinates [x_r, z_r], the model may have limited capacity 
        to learn fine spatial variations of the response.

        To improve this, this class maps the receiver coordinates into Fourier features using sine and cosine functions with multiple frequency bands.

        Why this is useful
        ------------------
        - It gives the model a more expressive coordinate representation.
        - It helps the network capture fine spatial dependence.
        - It is a common and effective strategy in coordinate-conditioned models.

        Input
        -----
        receiver_coords : tensor with shape (batch, 2)

        Output
        ------
        encoded_coords : tensor with shape (batch, out_dim)
        """

        def __init__(self, in_dim=2, num_bands=16, max_frequency=16.0, include_input=True):      # Initialize coordinate encoder hyperparameters
            super().__init__()                                                                   # Initialize nn.Module parent class

            self.in_dim = int(in_dim)                                                            # Store input coordinate dimension
            self.num_bands = int(num_bands)                                                      # Store number of Fourier frequency bands
            self.max_frequency = float(max_frequency)                                            # Store maximum frequency used in Fourier encoding
            self.include_input = bool(include_input)                                             # Store whether raw coordinates are also concatenated

            if self.in_dim <= 0:                                                                 # Validate coordinate dimension
                raise ValueError("in_dim must be positive.")                                     # Raise an error if the coordinate dimension is invalid

            if self.num_bands <= 0:                                                              # Validate number of Fourier bands
                raise ValueError("num_bands must be positive.")                                  # Raise an error if the number of Fourier bands is invalid

            if self.max_frequency <= 0.0:                                                        # Validate maximum Fourier frequency
                raise ValueError("max_frequency must be positive.")                              # Raise an error if the maximum frequency is invalid

            frequencies = torch.logspace(                                                        # Build logarithmically spaced Fourier frequencies
                start=0.0,                                                                       # Start frequency exponent = 10^0
                end=np.log10(self.max_frequency),                                                # End frequency exponent = log10(max_frequency)
                steps=self.num_bands,                                                            # Number of frequency bands
                dtype=torch.float32,                                                             # Use float32 for compatibility and efficiency
            )

            self.register_buffer("frequencies", frequencies, persistent=True)                    # Register frequencies as a persistent non-trainable tensor

            base_dim = self.in_dim if self.include_input else 0                                  # Add raw coordinates only if requested
            trig_dim = 2 * self.in_dim * self.num_bands                                          # Two trigonometric branches: sin and cos
            self.out_dim = int(base_dim + trig_dim)                                              # Store total encoded coordinate dimension

        def forward(self, receiver_coords):                                                      # Encode normalized receiver coordinates
            if receiver_coords.ndim != 2:                                                        # Ensure input has shape (batch, coord_dim)
                raise ValueError(                                                                # Raise an error if tensor rank is incorrect
                    f"receiver_coords must have shape (batch, {self.in_dim}), got {receiver_coords.shape}"
                )

            if receiver_coords.shape[1] != self.in_dim:                                          # Ensure coordinate dimension matches expected value
                raise ValueError(                                                                # Raise an error if coordinate width is incorrect
                    f"receiver_coords must have shape (batch, {self.in_dim}), got {receiver_coords.shape}"
                )

            pieces = []                                                                          # Create a list to collect encoded feature blocks

            if self.include_input:                                                               # Append raw coordinates if requested
                pieces.append(receiver_coords)                                                   # Keep direct coordinate information for the network

            coords_expanded = receiver_coords.unsqueeze(-1)                                      # Expand coordinates to shape (batch, in_dim, 1)
            freq_view = self.frequencies.view(1, 1, -1)                                          # Reshape frequency buffer for broadcasting
            angles = 2.0 * np.pi * coords_expanded * freq_view                                   # Compute Fourier angles for all coordinates and bands

            sin_features = torch.sin(angles).reshape(receiver_coords.shape[0], -1)               # Flatten sine features to shape (batch, in_dim * num_bands)
            cos_features = torch.cos(angles).reshape(receiver_coords.shape[0], -1)               # Flatten cosine features to shape (batch, in_dim * num_bands)

            pieces.append(sin_features)                                                          # Append sine branch features
            pieces.append(cos_features)                                                          # Append cosine branch features

            encoded_coords = torch.cat(pieces, dim=1)                                            # Concatenate all coordinate feature blocks
            return encoded_coords                                                                # Return final encoded coordinates


    class ReceiverConditionedMLPBlock(nn.Module):                                                # Residual Multi-Layer Perceptron block (MLP) used after model-coordinate fusion
        """
        Residual Multi-Layer Perceptron block (MLP) for fused latent features.

        What this class does
        --------------------
        This class refines the fused latent representation after the network has already combined:
        - the information coming from the velocity model,
        - and the information coming from the receiver coordinates.

        In other words, this block does not encode the physics directly, and it is not the full seismic model by itself.

        Its job is to improve the quality of the latent representation after the two branches have already been merged.

        Why this is useful
        ------------------
        - It makes the fused latent space more flexible.
        - It helps stabilize training through a residual connection.
        - It improves nonlinear feature interaction without changing the overall architecture too much.
        """

        def __init__(self, width, dropout=0.05):                                                 # Initialize residual block width and dropout
            super().__init__()                                                                   # Initialize nn.Module parent class

            self.width = int(width)                                                              # Store latent width
            self.dropout = float(dropout)                                                        # Store dropout probability

            if self.width <= 0:                                                                  # Validate width
                raise ValueError("width must be positive.")                                      # Raise an error if width is invalid

            if self.dropout < 0.0 or self.dropout >= 1.0:                                        # Validate dropout range
                raise ValueError("dropout must be in the range [0, 1).")                         # Raise an error if dropout is invalid

            self.norm = nn.LayerNorm(self.width)                                                 # Normalize fused features before the residual transformation
            self.fc1 = nn.Linear(self.width, self.width)                                         # First linear layer inside the residual block
            self.act = nn.GELU()                                                                 # Nonlinear activation function
            self.drop1 = nn.Dropout(self.dropout)                                                # Dropout after first activation
            self.fc2 = nn.Linear(self.width, self.width)                                         # Second linear layer inside the residual block
            self.drop2 = nn.Dropout(self.dropout)                                                # Dropout after second linear transformation

        def forward(self, x):                                                                    # Apply residual MLP transformation
            residual = x                                                                         # Store input for the skip connection
            x = self.norm(x)                                                                     # Normalize features before transformation
            x = self.fc1(x)                                                                      # Apply first linear transformation
            x = self.act(x)                                                                      # Apply nonlinear activation
            x = self.drop1(x)                                                                    # Apply dropout for regularization
            x = self.fc2(x)                                                                      # Apply second linear transformation
            x = self.drop2(x)                                                                    # Apply second dropout
            x = residual + x                                                                     # Add skip connection to stabilize optimization
            return x                                                                             # Return residual block output


    class BaselineReceiverConditionedSeismogramNet(nn.Module):                                   # Receiver-conditioned model for Phase 1
        """
        Receiver-conditioned neural network for one-trace seismic emulation.

        What this class does
        --------------------
        This is the main model of Training.

        It solves the new learning problem introduced in this stage:

            velocity_model + receiver_coordinates -> one trace

        Instead of predicting all receiver traces at once, this model predicts the seismogram corresponding to one queried receiver.

        This is the correct direction for now, because the scientific goalis no longer just to emulate a fixed acquisition geometry, 
        but to learn how the waveform changes as the receiver location changes.

        Internal structure
        ------------------
        This model has three main parts:

        1. A CNN branch for the velocity model
           - extracts spatial information from the 2D medium

        2. A receiver-coordinate branch
           - encodes the queried receiver location using Fourier features
           - transforms it into a latent vector

        3. A fusion and decoding stage
           - modulates the model latent using receiver information
           - fuses both branches
           - decodes the final latent representation into one trace

        Why this architecture makes sense
        ---------------------------------
        - The velocity model is spatial, so CNNs are appropriate.
        - The receiver is a coordinate query, so coordinate encoding is needed.
        - The output is a temporal signal, so the final decoder produces a full 1D trace.

        Input
        -----
        x_model : tensor with shape (batch, 1, nz, nx)
            Velocity model.

        receiver_coords : tensor with shape (batch, 2)
            Receiver coordinates, preferably normalized to [0, 1].

        Output
        ------
        y_trace : tensor with shape (batch, n_time)
            Predicted seismogram for the queried receiver.
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
        ):
            super().__init__()                                                                   # Initialize nn.Module parent class

            self.n_time = int(n_time)                                                            # Store the number of time samples in the output trace
            self.coord_dim = int(coord_dim)                                                      # Store the receiver coordinate dimension
            self.coord_num_bands = int(coord_num_bands)                                          # Store the number of Fourier bands for coordinates
            self.coord_max_frequency = float(coord_max_frequency)                                # Store the maximum Fourier frequency for coordinates
            self.model_latent_dim = int(model_latent_dim)                                        # Store latent dimension of the model encoder branch
            self.coord_latent_dim = int(coord_latent_dim)                                        # Store latent dimension of the receiver branch
            self.fusion_dim = int(fusion_dim)                                                    # Store latent width after fusion
            self.decoder_hidden_dim = int(decoder_hidden_dim)                                    # Store hidden width of the temporal decoder
            self.dropout = float(dropout)                                                        # Store dropout probability used across the network

            if self.n_time <= 0:                                                                 # Validate output trace length
                raise ValueError("n_time must be positive.")                                     # Raise an error if trace length is invalid

            if self.coord_dim <= 0:                                                              # Validate coordinate dimension
                raise ValueError("coord_dim must be positive.")                                  # Raise an error if coordinate dimension is invalid

            if self.model_latent_dim <= 0:                                                       # Validate model latent width
                raise ValueError("model_latent_dim must be positive.")                           # Raise an error if model latent width is invalid

            if self.coord_latent_dim <= 0:                                                       # Validate receiver latent width
                raise ValueError("coord_latent_dim must be positive.")                           # Raise an error if receiver latent width is invalid

            if self.fusion_dim <= 0:                                                             # Validate fusion latent width
                raise ValueError("fusion_dim must be positive.")                                 # Raise an error if fusion width is invalid

            if self.decoder_hidden_dim <= 0:                                                     # Validate decoder hidden width
                raise ValueError("decoder_hidden_dim must be positive.")                         # Raise an error if decoder width is invalid

            if self.dropout < 0.0 or self.dropout >= 1.0:                                        # Validate dropout range
                raise ValueError("dropout must be in the range [0, 1).")                         # Raise an error if dropout is invalid

            self.receiver_fourier_encoder = FourierFeatureReceiverEncoding(                      # Create Fourier encoder for receiver coordinates
                in_dim=self.coord_dim,                                                           # Coordinate dimension is 2 = (x, z)
                num_bands=self.coord_num_bands,                                                  # Number of Fourier frequency bands
                max_frequency=self.coord_max_frequency,                                          # Maximum frequency used in Fourier encoding
                include_input=True,                                                              # Keep raw coordinates together with Fourier features
            )

            self.receiver_mlp = nn.Sequential(                                                   # Create receiver branch that maps encoded coordinates to a latent vector
                nn.Linear(self.receiver_fourier_encoder.out_dim, 128),                           # First projection from Fourier features to a compact latent space
                nn.GELU(),                                                                       # Apply nonlinear activation
                nn.LayerNorm(128),                                                               # Normalize receiver latent activations
                nn.Dropout(self.dropout),                                                        # Apply dropout for regularization
                nn.Linear(128, self.coord_latent_dim),                                           # Project to final receiver latent dimension
                nn.GELU(),                                                                       # Apply nonlinear activation
                nn.LayerNorm(self.coord_latent_dim),                                             # Normalize final receiver latent vector
            )

            self.model_encoder = nn.Sequential(                                                  # Create CNN branch for the 2D velocity model
                nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),                            # Downsample input model while extracting low-level spatial features
                nn.GELU(),                                                                       # Apply nonlinear activation
                nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),                           # Continue spatial feature extraction with stronger channel capacity
                nn.GELU(),                                                                       # Apply nonlinear activation
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),                          # Increase channels and reduce spatial size
                nn.GELU(),                                                                       # Apply nonlinear activation
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),                         # Build deeper spatial representation
                nn.GELU(),                                                                       # Apply nonlinear activation
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),                         # Keep strong feature width while compressing spatial size
                nn.GELU(),                                                                       # Apply nonlinear activation
                nn.AdaptiveAvgPool2d((4, 4)),                                                    # Force a fixed output grid independent of exact input size
            )

            self.model_head = nn.Sequential(                                                     # Convert CNN feature map into a compact model latent vector
                nn.Flatten(),                                                                    # Flatten pooled feature map to a vector
                nn.Linear(256 * 4 * 4, 1024),                                                    # Project flattened CNN features to a high-capacity latent representation
                nn.GELU(),                                                                       # Apply nonlinear activation
                nn.LayerNorm(1024),                                                              # Normalize model latent activations
                nn.Dropout(self.dropout),                                                        # Apply dropout for regularization
                nn.Linear(1024, self.model_latent_dim),                                          # Project to final model latent dimension
                nn.GELU(),                                                                       # Apply nonlinear activation
                nn.LayerNorm(self.model_latent_dim),                                             # Normalize final model latent vector
            )

            self.film_gamma = nn.Linear(self.coord_latent_dim, self.model_latent_dim)            # Learn multiplicative FiLM coefficients from receiver latent features
            self.film_beta = nn.Linear(self.coord_latent_dim, self.model_latent_dim)             # Learn additive FiLM coefficients from receiver latent features

            self.pre_fusion = nn.Sequential(                                                     # Fuse receiver-conditioned model latent with receiver latent
                nn.Linear(self.model_latent_dim + self.coord_latent_dim, self.fusion_dim),       # Concatenate both branches and project to fusion space
                nn.GELU(),                                                                       # Apply nonlinear activation
                nn.LayerNorm(self.fusion_dim),                                                   # Normalize fusion features
                nn.Dropout(self.dropout),                                                        # Apply dropout for regularization
            )

            self.fusion_block_1 = ReceiverConditionedMLPBlock(                                   # First residual fusion block
                width=self.fusion_dim,                                                           # Use fusion latent width
                dropout=self.dropout,                                                            # Use global dropout value
            )

            self.fusion_block_2 = ReceiverConditionedMLPBlock(                                   # Second residual fusion block
                width=self.fusion_dim,                                                           # Use fusion latent width
                dropout=self.dropout,                                                            # Use global dropout value
            )

            self.trace_decoder = nn.Sequential(                                                  # Decode fused latent vector into a complete output trace
                nn.Linear(self.fusion_dim, self.decoder_hidden_dim),                             # First decoder projection
                nn.GELU(),                                                                       # Apply nonlinear activation
                nn.LayerNorm(self.decoder_hidden_dim),                                           # Normalize decoder hidden activations
                nn.Dropout(self.dropout),                                                        # Apply dropout for regularization
                nn.Linear(self.decoder_hidden_dim, self.decoder_hidden_dim),                     # Second decoder projection at constant high capacity
                nn.GELU(),                                                                       # Apply nonlinear activation
                nn.LayerNorm(self.decoder_hidden_dim),                                           # Normalize decoder hidden activations
                nn.Dropout(self.dropout),                                                        # Apply dropout for regularization
                nn.Linear(self.decoder_hidden_dim, self.n_time),                                 # Final linear projection to the full temporal trace
            )

        def forward(self, x_model, receiver_coords):                                             # Predict one seismogram from model and receiver coordinates
            if x_model.ndim != 4:                                                                # Ensure x_model has shape (batch, channels, nz, nx)
                raise ValueError(                                                                # Raise an error if x_model rank is incorrect
                    f"x_model must have shape (batch, 1, nz, nx), got {x_model.shape}"
                )

            if x_model.shape[1] != 1:                                                            # Ensure the model branch receives exactly one input channel
                raise ValueError(                                                                # Raise an error if input channel count is incorrect
                    f"x_model must have shape (batch, 1, nz, nx), got {x_model.shape}"
                )

            if receiver_coords.ndim != 2:                                                        # Ensure receiver coordinates have shape (batch, coord_dim)
                raise ValueError(                                                                # Raise an error if receiver_coords rank is incorrect
                    f"receiver_coords must have shape (batch, {self.coord_dim}), got {receiver_coords.shape}"
                )

            if receiver_coords.shape[1] != self.coord_dim:                                       # Ensure receiver coordinate width matches the expected dimension
                raise ValueError(                                                                # Raise an error if receiver coordinate width is incorrect
                    f"receiver_coords must have shape (batch, {self.coord_dim}), got {receiver_coords.shape}"
                )

            model_features = self.model_encoder(x_model)                                         # Extract deep spatial features from the velocity model
            model_latent = self.model_head(model_features)                                       # Convert spatial features into a compact model latent vector

            receiver_features = self.receiver_fourier_encoder(receiver_coords)                   # Encode receiver coordinates with Fourier features
            receiver_latent = self.receiver_mlp(receiver_features)                               # Project encoded coordinates to the receiver latent space

            gamma = self.film_gamma(receiver_latent)                                             # Compute multiplicative FiLM coefficients from receiver latent vector
            beta = self.film_beta(receiver_latent)                                               # Compute additive FiLM coefficients from receiver latent vector

            modulated_model_latent = model_latent * (1.0 + gamma) + beta                         # Modulate model latent features according to receiver position

            fused_latent = torch.cat(                                                            # Concatenate receiver-conditioned model latent and receiver latent
                [modulated_model_latent, receiver_latent],                                       # Merge both branches along feature dimension
                dim=1,                                                                           # Concatenate along latent-feature axis
            )

            fused_latent = self.pre_fusion(fused_latent)                                         # Project concatenated latent vector into the fusion space
            fused_latent = self.fusion_block_1(fused_latent)                                     # Refine fused latent features with first residual block
            fused_latent = self.fusion_block_2(fused_latent)                                     # Refine fused latent features with second residual block

            y_trace = self.trace_decoder(fused_latent)                                           # Decode fused latent representation into the final trace
            return y_trace                                                                       # Return predicted seismogram with shape (batch, n_time)

else:

    class BaselineReceiverConditionedSeismogramNet:                                              # Fallback class when PyTorch is unavailable
        def __init__(self, *args, **kwargs):                                                     # Initialize fallback class
            raise ImportError(                                                                   # Raise informative import error
                f"PyTorch is not available in this environment. Original error: {TORCH_IMPORT_ERROR}"
            )



##############################################################################################################################################
##############################################################################################################################################
############################################       Baseline CNN Encoder Decoder       ########################################################
##############################################################################################################################################
##############################################################################################################################################
if TORCH_AVAILABLE:
    class BaselineCNNEncoderDecoder(nn.Module):
        """
        Baseline neural network for AI surface seismogram emulation.

        Input
        -----
        X = velocity_model
            shape: (batch, 1, nz, nx)

        Output
        ------
        Y = surface_seismograms
            shape: (batch, n_receivers, n_time)

        Architecture
        ------------
        - CNN encoder for spatial feature extraction
        - adaptive average pooling for latent compression
        - fully connected decoder for final seismogram prediction
        """

        def __init__(self, n_receivers=15, n_time=1500):
            super().__init__()

            self.n_receivers = n_receivers
            self.n_time = n_time
            self.output_dim = n_receivers * n_time

            # ---------------- Encoder ----------------
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),

                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),

                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),

                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),

                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),

                nn.AdaptiveAvgPool2d((4, 4))
            )

            # ---------------- Decoder ----------------
            self.decoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256 * 4 * 4, 512),
                nn.ReLU(),
                nn.Linear(512, self.output_dim)
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            x = x.view(-1, self.n_receivers, self.n_time)
            return x

else:

    class BaselineCNNEncoderDecoder:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"PyTorch is not available in this environment. Original error: {TORCH_IMPORT_ERROR}"
            )



##############################################################################################################################################
##############################################################################################################################################
############################################       Baseline CNN Hybrid Spectral Loss       ###################################################
##############################################################################################################################################
##############################################################################################################################################
if TORCH_AVAILABLE:
    class BaselineCNNHybridSpectralLoss(nn.Module):
        """
        Hybrid loss for baseline CNN training.

        This loss combines:
        - time-domain MSE,
        - time-domain MAE,
        - frequency-domain magnitude loss,
        - optional penalty for predicted energy outside a target frequency band.

        Parameters
        ----------
        alpha_time_mse : float, optional
            Weight for time-domain MSE.

        beta_time_mae : float, optional
            Weight for time-domain MAE.

        gamma_spectral : float, optional
            Weight for spectral magnitude loss in log scale.

        delta_out_of_band : float, optional
            Weight for the out-of-band energy penalty.

        fmin : float or None, optional
            Lower target frequency in Hz.

        fmax : float or None, optional
            Upper target frequency in Hz.

        eps : float, optional
            Small constant for numerical stability.
        """

        def __init__(
            self,
            alpha_time_mse=1.0,
            beta_time_mae=0.25,
            gamma_spectral=0.50,
            delta_out_of_band=0.10,
            fmin=None,
            fmax=None,
            eps=1e-8,
        ):
            super().__init__()

            self.alpha_time_mse = float(alpha_time_mse)
            self.beta_time_mae = float(beta_time_mae)
            self.gamma_spectral = float(gamma_spectral)
            self.delta_out_of_band = float(delta_out_of_band)
            self.fmin = None if fmin is None else float(fmin)
            self.fmax = None if fmax is None else float(fmax)
            self.eps = float(eps)

            if self.fmin is not None and self.fmin < 0.0:
                raise ValueError("fmin must be non-negative.")

            if self.fmax is not None and self.fmax <= 0.0:
                raise ValueError("fmax must be positive.")

            if self.fmin is not None and self.fmax is not None and self.fmax <= self.fmin:
                raise ValueError("fmax must be greater than fmin.")

        def _spectral_log_magnitude(self, signal):
            spectrum = torch.fft.rfft(signal, dim=-1)
            magnitude = torch.abs(spectrum)
            return torch.log1p(magnitude)

        def _out_of_band_penalty(self, y_pred, dt_value):
            if self.delta_out_of_band <= 0.0:
                return y_pred.new_tensor(0.0)

            dt_value = float(dt_value)
            if dt_value <= 0.0:
                raise ValueError("Each dt value must be positive.")

            n_time = y_pred.shape[-1]
            freqs = torch.fft.rfftfreq(n_time, d=dt_value, device=y_pred.device)
            spectrum = torch.fft.rfft(y_pred, dim=-1)
            power = torch.abs(spectrum) ** 2

            in_band_mask = torch.ones_like(freqs, dtype=torch.bool)

            if self.fmin is not None:
                in_band_mask = in_band_mask & (freqs >= self.fmin)

            if self.fmax is not None:
                in_band_mask = in_band_mask & (freqs <= self.fmax)

            out_of_band_mask = ~in_band_mask

            if not torch.any(out_of_band_mask):
                return y_pred.new_tensor(0.0)

            return power[..., out_of_band_mask].mean()

        def forward(self, y_pred, y_true, dt_batch):
            mse_time = torch.mean((y_pred - y_true) ** 2)
            mae_time = torch.mean(torch.abs(y_pred - y_true))

            if torch.is_tensor(dt_batch):
                dt_values = dt_batch.detach().cpu().reshape(-1).tolist()
            else:
                dt_values = np.asarray(dt_batch, dtype=float).reshape(-1).tolist()

            if len(dt_values) != y_pred.shape[0]:
                raise ValueError("dt_batch must contain one dt value per batch sample.")

            spectral_loss_accum = y_pred.new_tensor(0.0)
            out_of_band_loss_accum = y_pred.new_tensor(0.0)

            for i, dt_value in enumerate(dt_values):
                pred_log_mag = self._spectral_log_magnitude(y_pred[i:i+1])
                true_log_mag = self._spectral_log_magnitude(y_true[i:i+1])
                spectral_loss_accum = spectral_loss_accum + torch.mean((pred_log_mag - true_log_mag) ** 2)
                out_of_band_loss_accum = out_of_band_loss_accum + self._out_of_band_penalty(y_pred[i:i+1], dt_value)

            spectral_loss = spectral_loss_accum / len(dt_values)
            out_of_band_loss = out_of_band_loss_accum / len(dt_values)

            total_loss = (
                self.alpha_time_mse * mse_time
                + self.beta_time_mae * mae_time
                + self.gamma_spectral * spectral_loss
                + self.delta_out_of_band * out_of_band_loss
            )

            return total_loss

else:

    class BaselineCNNHybridSpectralLoss:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"PyTorch is not available in this environment. Original error: {TORCH_IMPORT_ERROR}"
            )


##############################################################################################################################################
##############################################################################################################################################
###########################################       Baseline CNN Performance Evaluator       ###################################################
##############################################################################################################################################
##############################################################################################################################################
if TORCH_AVAILABLE:
    class BaselineCNNPerformanceEvaluator:
        """
        Quantitative evaluator for the baseline CNN surface seismogram emulator.

        This class is intended for:
        - quantitative validation on train/val/test splits,
        - per-sample and per-receiver error analysis,
        - comparison of trace-level physical indicators,
        - optional plotting of prediction vs. target traces,
        - optional export of evaluation tables to the docs directory.

        Metrics currently included
        --------------------------
        - MSE
        - MAE
        - RMSE
        - relative L2 error
        - Pearson correlation
        - peak arrival time
        - peak arrival time absolute error
        - peak amplitude
        - peak amplitude absolute error
        - peak amplitude relative error
        - total energy
        - total energy relative error
        - dominant frequency
        - dominant frequency absolute error
        """

        def __init__(
            self,
            h5_path,
            best_model_path,
            split="val",
            normalize_x=False,
            normalize_y=False,
            n_receivers=15,
            n_time=1500,
            device=None,
            docs_dir=None,
        ):
            if split not in ["train", "val", "test"]:
                raise ValueError("split must be one of: 'train', 'val', or 'test'.")

            self.h5_path = Path(h5_path)
            self.best_model_path = Path(best_model_path)
            self.split = split
            self.normalize_x = normalize_x
            self.normalize_y = normalize_y
            self.n_receivers = int(n_receivers)
            self.n_time = int(n_time)

            if not self.h5_path.exists():
                raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

            if not self.best_model_path.exists():
                raise FileNotFoundError(f"Best model file not found: {self.best_model_path}")

            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)

            if docs_dir is None:
                self.docs_dir = self.h5_path.parent.parent.parent / "docs"
            else:
                self.docs_dir = Path(docs_dir)

            self.docs_dir.mkdir(parents=True, exist_ok=True)

            self.dataset = HDF5SurfaceSeismogramTorchDataset(
                h5_path=self.h5_path,
                split=self.split,
                normalize_x=self.normalize_x,
                normalize_y=self.normalize_y,
                return_metadata=True,
            )

            self.model = BaselineCNNEncoderDecoder(
                n_receivers=self.n_receivers,
                n_time=self.n_time,
            ).to(self.device)

            try:
                checkpoint = torch.load(
                    self.best_model_path,
                    map_location=self.device,
                    weights_only=False,
                )
            except TypeError:
                checkpoint = torch.load(
                    self.best_model_path,
                    map_location=self.device,
                )

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.checkpoint_epoch = checkpoint.get("epoch", None)
                self.checkpoint_train_loss = checkpoint.get("train_loss", None)
                self.checkpoint_val_loss = checkpoint.get("val_loss", None)
            else:
                self.model.load_state_dict(checkpoint)
                self.checkpoint_epoch = None
                self.checkpoint_train_loss = None
                self.checkpoint_val_loss = None

            self.model.eval()

            self.last_metrics_df = None
            self.last_receiver_summary_df = None
            self.last_overall_summary_df = None
            self.last_export_paths = None

        def _decode_string(self, value):
            if isinstance(value, bytes):
                return value.decode("utf-8")
            return str(value)

        def _compute_trace_metrics(self, y_true, y_pred, dt):
            y_true = np.asarray(y_true, dtype=float).ravel()
            y_pred = np.asarray(y_pred, dtype=float).ravel()

            eps = 1e-12
            error = y_pred - y_true

            mse = float(np.mean(error ** 2))
            mae = float(np.mean(np.abs(error)))
            rmse = float(np.sqrt(mse))

            true_norm = float(np.linalg.norm(y_true))
            error_norm = float(np.linalg.norm(error))
            relative_l2_error = float(error_norm / (true_norm + eps))

            std_true = float(np.std(y_true))
            std_pred = float(np.std(y_pred))
            if std_true < eps and std_pred < eps:
                correlation = 1.0 if np.allclose(y_true, y_pred) else 0.0
            elif std_true < eps or std_pred < eps:
                correlation = 0.0
            else:
                correlation = float(np.corrcoef(y_true, y_pred)[0, 1])

            peak_idx_true = int(np.argmax(np.abs(y_true)))
            peak_idx_pred = int(np.argmax(np.abs(y_pred)))

            peak_arrival_time_true_s = float(peak_idx_true * dt)
            peak_arrival_time_pred_s = float(peak_idx_pred * dt)
            peak_arrival_time_abs_error_s = float(
                abs(peak_arrival_time_pred_s - peak_arrival_time_true_s)
            )

            peak_amplitude_true = float(np.max(np.abs(y_true)))
            peak_amplitude_pred = float(np.max(np.abs(y_pred)))
            peak_amplitude_abs_error = float(abs(peak_amplitude_pred - peak_amplitude_true))
            peak_amplitude_relative_error = float(
                peak_amplitude_abs_error / (peak_amplitude_true + eps)
            )

            total_energy_true = float(np.sum(y_true ** 2) * dt)
            total_energy_pred = float(np.sum(y_pred ** 2) * dt)
            total_energy_relative_error = float(
                abs(total_energy_pred - total_energy_true) / (total_energy_true + eps)
            )

            fft_true = np.fft.rfft(y_true)
            fft_pred = np.fft.rfft(y_pred)
            freqs = np.fft.rfftfreq(len(y_true), d=dt)

            if len(freqs) > 1:
                dominant_idx_true = int(np.argmax(np.abs(fft_true[1:])) + 1)
                dominant_idx_pred = int(np.argmax(np.abs(fft_pred[1:])) + 1)
                dominant_frequency_true_Hz = float(freqs[dominant_idx_true])
                dominant_frequency_pred_Hz = float(freqs[dominant_idx_pred])
            else:
                dominant_frequency_true_Hz = 0.0
                dominant_frequency_pred_Hz = 0.0

            dominant_frequency_abs_error_Hz = float(
                abs(dominant_frequency_pred_Hz - dominant_frequency_true_Hz)
            )

            return {
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "relative_l2_error": relative_l2_error,
                "correlation": correlation,
                "peak_arrival_time_true_s": peak_arrival_time_true_s,
                "peak_arrival_time_pred_s": peak_arrival_time_pred_s,
                "peak_arrival_time_abs_error_s": peak_arrival_time_abs_error_s,
                "peak_amplitude_true": peak_amplitude_true,
                "peak_amplitude_pred": peak_amplitude_pred,
                "peak_amplitude_abs_error": peak_amplitude_abs_error,
                "peak_amplitude_relative_error": peak_amplitude_relative_error,
                "total_energy_true": total_energy_true,
                "total_energy_pred": total_energy_pred,
                "total_energy_relative_error": total_energy_relative_error,
                "dominant_frequency_true_Hz": dominant_frequency_true_Hz,
                "dominant_frequency_pred_Hz": dominant_frequency_pred_Hz,
                "dominant_frequency_abs_error_Hz": dominant_frequency_abs_error_Hz,
            }

        def _export_metrics_to_docs(self, metrics_df, receiver_summary_df, overall_summary_df, file_prefix):
            """
            Save evaluation tables to the docs directory.

            Files created
            -------------
            - <file_prefix>_detailed.xlsx
            - <file_prefix>_receiver_summary.xlsx
            - <file_prefix>_overall_summary.xlsx
            - <file_prefix>_summary.txt
            """
            # timestamp = time.strftime("%Y%m%d_%H%M%S")
            # base_name = f"{file_prefix}_{self.split}_{timestamp}"
            base_name = f"{file_prefix}_{self.split}"

            detailed_csv_path = self.docs_dir / f"{base_name}_detailed.xlsx"
            receiver_csv_path = self.docs_dir / f"{base_name}_receiver_summary.xlsx"
            overall_csv_path = self.docs_dir / f"{base_name}_overall_summary.xlsx"
            summary_txt_path = self.docs_dir / f"{base_name}_summary.txt"

            metrics_df.to_excel(detailed_csv_path, index=False)
            receiver_summary_df.to_excel(receiver_csv_path, index=False)
            overall_summary_df.to_excel(overall_csv_path, index=False)

            with open(summary_txt_path, "w", encoding="utf-8") as f:
                f.write("=" * 120 + "\n")
                f.write("Baseline CNN quantitative evaluation summary\n")
                f.write("=" * 120 + "\n\n")
                f.write(f"Split: {self.split}\n")
                f.write(f"Checkpoint epoch: {self.checkpoint_epoch}\n")
                f.write(f"Checkpoint train loss: {self.checkpoint_train_loss}\n")
                f.write(f"Checkpoint val loss: {self.checkpoint_val_loss}\n")
                f.write(f"HDF5 path: {self.h5_path}\n")
                f.write(f"Best model path: {self.best_model_path}\n")
                f.write(f"Docs directory: {self.docs_dir}\n\n")

                f.write("=" * 120 + "\n")
                f.write("Overall summary\n")
                f.write("=" * 120 + "\n")
                f.write(overall_summary_df.to_string(index=False))
                f.write("\n\n")

                f.write("=" * 120 + "\n")
                f.write("Receiver summary\n")
                f.write("=" * 120 + "\n")
                f.write(receiver_summary_df.to_string(index=False))
                f.write("\n")

            export_paths = {
                "detailed_csv": detailed_csv_path,
                "receiver_summary_csv": receiver_csv_path,
                "overall_summary_csv": overall_csv_path,
                "summary_txt": summary_txt_path,
            }

            self.last_export_paths = export_paths

            print("\n" + "=" * 120)
            print("Evaluation tables saved successfully in docs directory")
            print(f"Detailed metrics      : {detailed_csv_path}")
            print(f"Receiver summary      : {receiver_csv_path}")
            print(f"Overall summary       : {overall_csv_path}")
            print(f"Plain text summary    : {summary_txt_path}")
            print("=" * 120)
            print("\n")

            return export_paths

        def evaluate(
            self,
            max_samples=None,
            receiver_ids=None,
            verbose=True,
            save_to_docs=False,
            file_prefix="baseline_cnn_evaluation",
        ):
            """
            Run quantitative evaluation on the selected dataset split.

            Parameters
            ----------
            max_samples : int or None, optional
                Maximum number of samples from the split to evaluate.
                If None, all samples in the selected split are used.

            receiver_ids : list[int] or None, optional
                Receiver indices to evaluate using 0-based indexing.
                If None, all receivers are evaluated.

            verbose : bool, optional
                If True, print a compact summary.

            save_to_docs : bool, optional
                If True, export the generated tables to the docs directory.

            file_prefix : str, optional
                Prefix used for the exported files.

            Returns
            -------
            metrics_df : pandas.DataFrame
                One row per sample-receiver pair.

            receiver_summary_df : pandas.DataFrame
                Mean metrics grouped by receiver.

            overall_summary_df : pandas.DataFrame
                Mean metrics across all evaluated rows.
            """
            n_available_samples = len(self.dataset)

            if max_samples is None:
                n_eval_samples = n_available_samples
            else:
                n_eval_samples = min(int(max_samples), n_available_samples)

            if receiver_ids is None:
                receiver_ids = list(range(self.n_receivers))
            else:
                receiver_ids = [int(r) for r in receiver_ids]

            for receiver_id in receiver_ids:
                if receiver_id < 0 or receiver_id >= self.n_receivers:
                    raise IndexError(
                        f"receiver_id={receiver_id} is out of range [0, {self.n_receivers - 1}]"
                    )

            rows = []

            with torch.no_grad():
                for split_sample_index in range(n_eval_samples):
                    x_true, y_true, metadata = self.dataset[split_sample_index]

                    x_input = x_true.unsqueeze(0).to(self.device)
                    y_pred = self.model(x_input)

                    y_pred_np = y_pred.squeeze(0).cpu().numpy()
                    y_true_np = y_true.cpu().numpy()

                    dt = float(metadata["dt"])
                    dx = float(metadata["dx"])
                    dz = float(metadata["dz"])
                    sample_id = int(metadata["sample_id"])
                    source_x = int(metadata["source_x"])
                    source_z = int(metadata["source_z"])
                    model_type = self._decode_string(metadata["model_type"])

                    for receiver_id in receiver_ids:
                        trace_metrics = self._compute_trace_metrics(
                            y_true=y_true_np[receiver_id],
                            y_pred=y_pred_np[receiver_id],
                            dt=dt,
                        )

                        rows.append(
                            {
                                "split": self.split,
                                "split_sample_index": split_sample_index,
                                "sample_id": sample_id,
                                "receiver_id": receiver_id + 1,
                                "receiver_index_0_based": receiver_id,
                                "dt": dt,
                                "dx": dx,
                                "dz": dz,
                                "source_x": source_x,
                                "source_z": source_z,
                                "model_type": model_type,
                                **trace_metrics,
                            }
                        )

            metrics_df = pd.DataFrame(rows)

            metric_columns = [
                "mse",
                "mae",
                "rmse",
                "relative_l2_error",
                "correlation",
                "peak_arrival_time_abs_error_s",
                "peak_amplitude_abs_error",
                "peak_amplitude_relative_error",
                "total_energy_relative_error",
                "dominant_frequency_abs_error_Hz",
            ]

            receiver_summary_df = (
                metrics_df
                .groupby("receiver_id")[metric_columns]
                .mean()
                .reset_index()
            )

            overall_summary_df = pd.DataFrame(
                [
                    {
                        "split": self.split,
                        "n_evaluated_samples": n_eval_samples,
                        "n_receivers_evaluated": len(receiver_ids),
                        "n_rows": len(metrics_df),
                        "mse_mean": float(metrics_df["mse"].mean()),
                        "mae_mean": float(metrics_df["mae"].mean()),
                        "rmse_mean": float(metrics_df["rmse"].mean()),
                        "relative_l2_error_mean": float(metrics_df["relative_l2_error"].mean()),
                        "correlation_mean": float(metrics_df["correlation"].mean()),
                        "peak_arrival_time_abs_error_s_mean": float(metrics_df["peak_arrival_time_abs_error_s"].mean()),
                        "peak_amplitude_abs_error_mean": float(metrics_df["peak_amplitude_abs_error"].mean()),
                        "peak_amplitude_relative_error_mean": float(metrics_df["peak_amplitude_relative_error"].mean()),
                        "total_energy_relative_error_mean": float(metrics_df["total_energy_relative_error"].mean()),
                        "dominant_frequency_abs_error_Hz_mean": float(metrics_df["dominant_frequency_abs_error_Hz"].mean()),
                    }
                ]
            )

            self.last_metrics_df = metrics_df
            self.last_receiver_summary_df = receiver_summary_df
            self.last_overall_summary_df = overall_summary_df

            if verbose:
                print("\n" + "=" * 120)
                print("Baseline CNN quantitative evaluation")
                print(f"Split                 : {self.split}")
                print(f"Evaluated samples     : {n_eval_samples}")
                print(f"Evaluated receivers   : {len(receiver_ids)}")
                print(f"Total evaluated rows  : {len(metrics_df)}")
                print("=" * 120)
                print("\nOverall summary")
                print(overall_summary_df)
                print("\nReceiver summary")
                print(receiver_summary_df)

            if save_to_docs:
                self._export_metrics_to_docs(
                    metrics_df=metrics_df,
                    receiver_summary_df=receiver_summary_df,
                    overall_summary_df=overall_summary_df,
                    file_prefix=file_prefix,
                )

            return metrics_df, receiver_summary_df, overall_summary_df

        def plot_prediction_review(
            self,
            sample_index,
            receivers_to_plot=None,
            save=False,
            figure_path=None,
        ):
            """
            Plot one evaluated sample: velocity model + selected true/predicted traces.

            Parameters
            ----------
            sample_index : int
                Index inside the selected split dataset.

            receivers_to_plot : list[int] or None, optional
                Receiver indices to plot using 0-based indexing.
                If None, a default subset is selected.

            save : bool, optional
                If True, save the figure.

            figure_path : str or Path or None, optional
                Output path for the figure when save=True.

            Returns
            -------
            output : dict
                Dictionary containing arrays and metadata for the selected sample.
            """
            if sample_index < 0 or sample_index >= len(self.dataset):
                raise IndexError(
                    f"sample_index={sample_index} is out of range [0, {len(self.dataset) - 1}]"
                )

            if receivers_to_plot is None:
                if self.n_receivers >= 5:
                    receivers_to_plot = [0, 2, self.n_receivers // 2, self.n_receivers - 3, self.n_receivers - 1]
                else:
                    receivers_to_plot = list(range(self.n_receivers))

            receivers_to_plot = [int(r) for r in receivers_to_plot]

            x_true, y_true, metadata = self.dataset[sample_index]

            with torch.no_grad():
                x_input = x_true.unsqueeze(0).to(self.device)
                y_pred = self.model(x_input)

            x_true_np = x_true.squeeze(0).cpu().numpy()
            y_true_np = y_true.cpu().numpy()
            y_pred_np = y_pred.squeeze(0).cpu().numpy()

            isx = int(metadata["source_x"])
            isz = int(metadata["source_z"])
            irx = metadata["receiver_x"].astype(int)
            irz = metadata["receiver_z"].astype(int)
            mapped_sample_id = int(metadata["sample_id"])

            fig = plt.figure(figsize=(16, 12))
            fig.suptitle(
                (
                    f"Baseline CNN Prediction Review | Split = {self.split} | "
                    f"sample_index = {sample_index} | mapped_sample_id = {mapped_sample_id}"
                ),
                fontsize=12,
                fontweight="bold",
                color=(0, 0, 1),
            )

            ax0 = plt.subplot(3, 2, 1)
            im = ax0.imshow(x_true_np, cmap="Spectral", aspect="auto")
            ax0.set_title("Input Velocity Model", fontsize=10, color=(0, 0, 1))
            ax0.set_xlabel("ix")
            ax0.set_ylabel("iz")
            ax0.axis("off")

            cbar0 = fig.colorbar(im, ax=ax0, pad=0.01, fraction=0.03)
            cbar0.ax.tick_params(labelsize=8)
            cbar0.set_label("Velocity")

            ax0.scatter(irx, irz, marker="^", s=20, linewidths=0.5, color=(0, 0, 0))
            for k in range(len(irx)):
                ax0.text(
                    irx[k],
                    irz[k] * 1.0,
                    f"ST{k+1}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    color=(0, 0, 0),
                )

            ax0.scatter([isx], [isz], marker="*", s=100, color=(0, 0, 0))
            ax0.text(
                float(isx) * 1.05,
                float(isz),
                "Source",
                ha="left",
                va="center",
                fontweight="bold",
                color=(0, 0, 0),
            )

            max_panels = 5
            receivers_to_plot = receivers_to_plot[:max_panels]

            for i, rec in enumerate(receivers_to_plot, start=2):
                ax = plt.subplot(3, 2, i)
                ax.plot(
                    y_true_np[rec],
                    lw=1.0,
                    ls="-",
                    color=[0, 0, 0],
                    label=f"Simulated - ST{rec+1}",
                )
                ax.plot(
                    y_pred_np[rec],
                    lw=1.0,
                    ls="-",
                    color=[0, 0, 1],
                    label=f"Predicted - ST{rec+1}",
                )
                ax.set_title(f"Receiver ST{rec+1}", fontsize=10, color=(0, 0, 1))
                ax.set_xlabel("Time index", fontsize=8)
                ax.set_ylabel("Amplitude", fontsize=8)
                ax.tick_params(axis="x", labelsize=8)
                ax.tick_params(axis="y", labelsize=8)
                ax.grid(visible=True, axis="x")
                ax.set_xlim(0, y_true_np.shape[1])
                ax.legend(
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    borderaxespad=0,
                    fontsize=8,
                    frameon=False,
                )

            plt.tight_layout()

            if save:
                if figure_path is None:
                    figure_path = self.docs_dir / f"5. baseline_cnn_prediction_review_sample_{sample_index}.svg"
                else:
                    figure_path = Path(figure_path)

                plt.savefig(figure_path, dpi=200, bbox_inches="tight")
                print("\n" + "=" * 120)
                print("Prediction review figure saved successfully")
                print(figure_path)
                print("=" * 120)

            plt.show()

            return {
                "sample_index": sample_index,
                "sample_id": mapped_sample_id,
                "x_true": x_true_np,
                "y_true": y_true_np,
                "y_pred": y_pred_np,
                "metadata": metadata,
            }
        
        def plot_receiver_summary_metrics(
            self,
            receiver_summary_df=None,
            metrics_to_plot=None,
            save=False,
            figure_path=None,
        ):
            """
            Plot the most relevant mean metrics grouped by receiver.

            Parameters
            ----------
            receiver_summary_df : pandas.DataFrame or None, optional
                Summary dataframe returned by evaluate().
                If None, the method uses self.last_receiver_summary_df.

            metrics_to_plot : list[tuple] or None, optional
                List of tuples with:
                (column_name, y_label, criterion)

                criterion:
                - "min" means that smaller values are better
                - "max" means that larger values are better

            save : bool, optional
                If True, save the figure.

            figure_path : str or Path or None, optional
                Output path for the figure when save=True.

            Returns
            -------
            fig : matplotlib.figure.Figure
                Figure object.
            """
            if receiver_summary_df is None:
                receiver_summary_df = self.last_receiver_summary_df

            if receiver_summary_df is None:
                raise ValueError(
                    "receiver_summary_df is None. Run evaluate() first or pass a valid dataframe."
                )

            if metrics_to_plot is None:
                metrics_to_plot = [
                    ("correlation", "Correlation", "max"),
                    ("rmse", "RMSE", "min"),
                    ("relative_l2_error", "Relative L2 Error", "min"),
                    ("peak_arrival_time_abs_error_s", "Peak Arrival Time Abs. Error (s)", "min"),
                    ("dominant_frequency_abs_error_Hz", "Dominant Frequency Abs. Error (Hz)", "min"),
                ]

            required_columns = ["receiver_id"] + [item[0] for item in metrics_to_plot]
            missing_columns = [col for col in required_columns if col not in receiver_summary_df.columns]

            if len(missing_columns) > 0:
                raise ValueError(
                    f"The following required columns are missing in receiver_summary_df: {missing_columns}"
                )

            receiver_ids = receiver_summary_df["receiver_id"].to_numpy(dtype=int)

            fig, axes = plt.subplots(3, 2, figsize=(16, 12))
            axes = axes.ravel()

            fig.suptitle(
                f"Baseline CNN Mean Metrics by Receiver, Split = {self.split}",
                fontsize=12,
                fontweight="bold",
                color=(0, 0, 1),
            )

            for i, (metric_name, y_label, criterion) in enumerate(metrics_to_plot):
                ax = axes[i]
                metric_values = receiver_summary_df[metric_name].to_numpy(dtype=float)

                if criterion == "max":
                    best_idx = int(np.argmax(metric_values))
                else:
                    best_idx = int(np.argmin(metric_values))

                best_receiver = int(receiver_ids[best_idx])
                best_value = float(metric_values[best_idx])

                # ax.plot(receiver_ids, metric_values, lw=1.2, ls="-", color=[0, 0, 0], marker="o", markersize=5,markerfacecolor=[1, 1, 1],
                #     markeredgewidth=1.0, markeredgecolor=[0, 0, 0],alpha=0.9)
                ax.bar(receiver_ids, metric_values, color=[0.7, 0.7, 1.0], alpha=1.0, edgecolor=[0, 0, 0], linewidth=0.5, zorder=1,width=0.6)

                ax.plot(best_receiver, best_value, lw=0.0, marker="o", markersize=6,color=[1, 0, 0], markerfacecolor=[1, 0, 0], markeredgecolor=[0, 0, 0],alpha=1.0)
                
                ax.text(best_receiver,best_value,f"  Best ST{best_receiver}",fontsize=8,color=(1, 0, 0),fontweight="bold", ha="left", va="bottom")

                ax.set_title(f"{metric_name}",fontsize=10,fontweight="normal",color=(0, 0, 1))
                ax.set_xlabel("Receiver ID", fontsize=8)
                ax.set_ylabel(y_label, fontsize=8)
                ax.tick_params(axis="x", labelsize=8)
                ax.tick_params(axis="y", labelsize=8)
                ax.set_xticks(receiver_ids)
                ax.set_xlim(receiver_ids.min(), receiver_ids.max())
                ax.grid(visible=False, axis="x")
                ax.grid(visible=True, axis="y")
                ax.set_xlim(receiver_ids.min() - 1, receiver_ids.max() + 1)
                

            # Hide the last empty subplot
            if len(metrics_to_plot) < len(axes):
                for j in range(len(metrics_to_plot), len(axes)):
                    axes[j].axis("off")

            plt.tight_layout()

            if save:
                if figure_path is None:
                    # timestamp = time.strftime("%Y%m%d_%H%M%S")
                    # figure_path = self.docs_dir / f"baseline_cnn_receiver_summary_metrics_{self.split}_{timestamp}.png"
                    figure_path = self.docs_dir / f"7. baseline_cnn_receiver_summary_metrics_{self.split}.svg"
                else:
                    figure_path = Path(figure_path)

                plt.savefig(figure_path, dpi=200, bbox_inches="tight")

                print("\n" + "=" * 120)
                print("Receiver summary metrics figure saved successfully")
                print(figure_path)
                print("=" * 120)
                print("\n")

            plt.show()

            return fig
        
        
        
        def plot_detailed_metrics_histograms(
            self,
            metrics_df=None,
            metrics_to_plot=None,
            bins=20,
            save=False,
            figure_path=None,
        ):
            """
            Plot global histograms of the most relevant detailed metrics.

            Parameters
            ----------
            metrics_df : pandas.DataFrame or None, optional
                Detailed dataframe returned by evaluate().
                If None, the method uses self.last_metrics_df.

            metrics_to_plot : list[tuple] or None, optional
                List of tuples with:
                (column_name, x_label)

            bins : int, optional
                Number of histogram bins.

            save : bool, optional
                If True, save the figure.

            figure_path : str or Path or None, optional
                Output path for the figure when save=True.

            Returns
            -------
            fig : matplotlib.figure.Figure
                Figure object.
            """
            if metrics_df is None:
                metrics_df = self.last_metrics_df

            if metrics_df is None:
                raise ValueError(
                    "metrics_df is None. Run evaluate() first or pass a valid dataframe."
                )

            if metrics_to_plot is None:
                metrics_to_plot = [
                    ("correlation", "Correlation"),
                    ("rmse", "RMSE"),
                    ("relative_l2_error", "Relative L2 Error"),
                    ("peak_arrival_time_abs_error_s", "Peak Arrival Time Abs. Error (s)"),
                    ("dominant_frequency_abs_error_Hz", "Dominant Frequency Abs. Error (Hz)"),
                ]

            required_columns = [item[0] for item in metrics_to_plot]
            missing_columns = [col for col in required_columns if col not in metrics_df.columns]

            if len(missing_columns) > 0:
                raise ValueError(
                    f"The following required columns are missing in metrics_df: {missing_columns}"
                )

            fig, axes = plt.subplots(3, 2, figsize=(16, 12))
            axes = axes.ravel()

            fig.suptitle(
                f"Baseline CNN Detailed Metrics Distribution (Histogram), Split = {self.split}",
                fontsize=12,
                fontweight="bold",
                color=(0, 0, 1),
            )

            for i, (metric_name, x_label) in enumerate(metrics_to_plot):
                ax = axes[i]

                values = metrics_df[metric_name].to_numpy(dtype=float)
                values = values[np.isfinite(values)]

                if len(values) == 0:
                    ax.set_title(f"{metric_name}",fontsize=10,fontweight="normal",color=(0, 0, 1))
                    ax.text(0.5, 0.5,"No valid data",transform=ax.transAxes,ha="center",va="center",fontsize=10,
                        color=(1, 0, 0),fontweight="bold")
                    ax.set_axis_off()
                    continue

                mean_value = float(np.mean(values))
                median_value = float(np.median(values))

                ax.hist(values,bins=bins,color=[0.7, 1, 0.7],edgecolor=[0, 0, 0],linewidth=0.6,alpha=1.0,zorder=1)

                ax.axvline(mean_value,color=[0, 0, 0],linestyle="--",linewidth=1.2,zorder=2,label=f"Mean = {mean_value:.4f}")

                ax.axvline(median_value, color=[1, 0, 0],linestyle="-",linewidth=1.2,zorder=3,label=f"Median = {median_value:.4f}")

                ax.set_title(f"{metric_name}",fontsize=10,fontweight="normal",color=(0, 0, 1))
                ax.set_xlabel(x_label, fontsize=8)
                ax.set_ylabel("Count", fontsize=8)
                ax.tick_params(axis="x", labelsize=8)
                ax.tick_params(axis="y", labelsize=8)
                ax.grid(visible=False, axis="x")
                ax.grid(visible=True, axis="y", alpha=0.3)
                ax.legend(loc="center left",bbox_to_anchor=(1.02, 0.5),borderaxespad=0,fontsize=8,frameon=False)
            if len(metrics_to_plot) < len(axes):
                for j in range(len(metrics_to_plot), len(axes)):
                    axes[j].axis("off")
            plt.tight_layout()

            if save:
                if figure_path is None:
                    figure_path = self.docs_dir / f"1. baseline_cnn_detailed_metrics_histograms_{self.split}.svg"
                else:
                    figure_path = Path(figure_path)

                plt.savefig(figure_path, dpi=200, bbox_inches="tight")

                print("\n" + "=" * 120)
                print("Detailed metrics histograms figure saved successfully")
                print(figure_path)
                print("=" * 120)
                print("\n")
            plt.show()
            return fig
        
        
        
        def plot_detailed_metrics_boxplots_by_receiver(
            self,
            metrics_df=None,
            metrics_to_plot=None,
            save=False,
            figure_path=None,
            show_fliers=True,
        ):
            """
            Plot boxplots of detailed metrics grouped by receiver.

            Parameters
            ----------
            metrics_df : pandas.DataFrame or None, optional
                Detailed dataframe returned by evaluate().
                If None, the method uses self.last_metrics_df.

            metrics_to_plot : list[tuple] or None, optional
                List of tuples with:
                (column_name, y_label, criterion)

                criterion:
                - "min" means that smaller values are better
                - "max" means that larger values are better

            save : bool, optional
                If True, save the figure.

            figure_path : str or Path or None, optional
                Output path for the figure when save=True.

            show_fliers : bool, optional
                If True, show outliers in the boxplots.

            Returns
            -------
            fig : matplotlib.figure.Figure
                Figure object.
            """
            if metrics_df is None:
                metrics_df = self.last_metrics_df

            if metrics_df is None:
                raise ValueError(
                    "metrics_df is None. Run evaluate() first or pass a valid dataframe."
                )

            if metrics_to_plot is None:
                metrics_to_plot = [
                    ("rmse", "RMSE", "min"),
                    ("correlation", "Correlation", "max"),
                    ("peak_arrival_time_abs_error_s", "Peak Arrival Time Abs. Error (s)", "min"),
                ]

            required_columns = ["receiver_id"] + [item[0] for item in metrics_to_plot]
            missing_columns = [col for col in required_columns if col not in metrics_df.columns]

            if len(missing_columns) > 0:
                raise ValueError(
                    f"The following required columns are missing in metrics_df: {missing_columns}"
                )

            receiver_ids = np.sort(metrics_df["receiver_id"].unique().astype(int))

            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            axes = axes.ravel()

            fig.suptitle(
                f"Baseline CNN Detailed Metrics by Receiver (Boxplots), Split = {self.split}", fontsize=12, fontweight="bold",color=(0, 0, 1))

            for i, (metric_name, y_label, criterion) in enumerate(metrics_to_plot):
                ax = axes[i]

                data_by_receiver = []
                medians = []

                for receiver_id in receiver_ids:
                    values = metrics_df.loc[metrics_df["receiver_id"] == receiver_id,metric_name].to_numpy(dtype=float)

                    values = values[np.isfinite(values)]
                    data_by_receiver.append(values)

                    if len(values) > 0:
                        medians.append(float(np.median(values)))
                    else:
                        medians.append(np.nan)

                box = ax.boxplot(data_by_receiver,positions=receiver_ids,widths=0.6, patch_artist=True, showfliers=show_fliers, medianprops=dict(color=[1, 0, 0], linewidth=1.2),
                    boxprops=dict(facecolor=[0.7, 0.7, 1.0], edgecolor=[0, 0, 0], linewidth=0.8), whiskerprops=dict(color=[0, 0, 0], linewidth=0.8),capprops=dict(color=[0, 0, 0], linewidth=0.8),flierprops=dict(
                        marker="o", markersize=3,markerfacecolor=[1, 1, 1], markeredgecolor=[0, 0, 0],alpha=0.8,))

                medians_array = np.asarray(medians, dtype=float)
                valid_mask = np.isfinite(medians_array)

                if np.any(valid_mask):
                    valid_receiver_ids = receiver_ids[valid_mask]
                    valid_medians = medians_array[valid_mask]

                    ax.plot(valid_receiver_ids, valid_medians,lw=1.0,ls="--",color=[0, 0, 0],marker="o", markersize=4,markerfacecolor=[1, 1, 1], markeredgewidth=0.8,markeredgecolor=[0, 0, 0],alpha=0.9,zorder=3)

                    if criterion == "max":
                        best_idx = int(np.argmax(valid_medians))
                    else:
                        best_idx = int(np.argmin(valid_medians))

                    best_receiver = int(valid_receiver_ids[best_idx])
                    best_value = float(valid_medians[best_idx])

                    ax.plot(best_receiver,best_value,lw=0.0, marker="o", markersize=6,color=[1, 0, 0], markerfacecolor=[1, 0, 0], markeredgecolor=[0, 0, 0],alpha=1.0,zorder=4)

                    ax.text(best_receiver, best_value,f"  Best ST{best_receiver}", fontsize=8,color=(1, 0, 0),fontweight="bold",ha="left",va="bottom")

                ax.set_title(f"{metric_name}",fontsize=10,fontweight="normal",color=(0, 0, 1))
                ax.set_xlabel("Receiver ID", fontsize=8)
                ax.set_ylabel(y_label, fontsize=8)
                ax.tick_params(axis="x", labelsize=8)
                ax.tick_params(axis="y", labelsize=8)
                ax.set_xticks(receiver_ids)
                ax.set_xlim(receiver_ids.min() - 1, receiver_ids.max() + 1)
                ax.grid(visible=False, axis="x")
                ax.grid(visible=True, axis="y", alpha=0.3)

            if len(metrics_to_plot) < len(axes):
                for j in range(len(metrics_to_plot), len(axes)):
                    axes[j].axis("off")

            plt.tight_layout()

            if save:
                if figure_path is None:
                    figure_path = self.docs_dir / f"2. baseline_cnn_detailed_metrics_boxplots_by_receiver_{self.split}.svg"
                else:
                    figure_path = Path(figure_path)

                plt.savefig(figure_path, dpi=200, bbox_inches="tight")

                print("\n" + "=" * 120)
                print("Detailed metrics boxplots by receiver figure saved successfully")
                print(figure_path)
                print("=" * 120)
                print("\n")

            plt.show()
            return fig
        
        
        def plot_detailed_metric_heatmap(
            self,
            metrics_df=None,
            metric_name="rmse",
            y_axis="split_sample_index",
            cmap="viridis",
            save=False,
            figure_path=None,
            highlight_worst=True,
            show_colorbar=True,
        ):
            """
            Plot a heatmap of one detailed metric as a function of sample and receiver.

            Parameters
            ----------
            metrics_df : pandas.DataFrame or None, optional
                Detailed dataframe returned by evaluate().
                If None, the method uses self.last_metrics_df.

            metric_name : str, optional
                Metric column to plot.
                Example:
                - "rmse"
                - "correlation"
                - "relative_l2_error"
                - "peak_arrival_time_abs_error_s"
                - "dominant_frequency_abs_error_Hz"

            y_axis : str, optional
                Column used for the heatmap vertical axis.
                Recommended:
                - "split_sample_index"
                - "sample_id"

            cmap : str, optional
                Colormap used in the heatmap.

            save : bool, optional
                If True, save the figure.

            figure_path : str or Path or None, optional
                Output path for the figure when save=True.

            highlight_worst : bool, optional
                If True, highlight the worst case in the heatmap.

            show_colorbar : bool, optional
                If True, show the colorbar.

            Returns
            -------
            fig : matplotlib.figure.Figure
                Figure object.

            heatmap_df : pandas.DataFrame
                Pivoted dataframe used to create the heatmap.
            """
            if metrics_df is None:
                metrics_df = self.last_metrics_df

            if metrics_df is None:
                raise ValueError(
                    "metrics_df is None. Run evaluate() first or pass a valid dataframe."
                )

            required_columns = ["receiver_id", y_axis, metric_name]
            missing_columns = [col for col in required_columns if col not in metrics_df.columns]

            if len(missing_columns) > 0:
                raise ValueError(
                    f"The following required columns are missing in metrics_df: {missing_columns}"
                )

            heatmap_df = metrics_df.pivot_table(
                index=y_axis,
                columns="receiver_id",
                values=metric_name,
                aggfunc="mean",
            )

            heatmap_df = heatmap_df.sort_index(axis=0).sort_index(axis=1)

            heatmap_values = heatmap_df.to_numpy(dtype=float)
            y_values = heatmap_df.index.to_numpy()
            x_values = heatmap_df.columns.to_numpy(dtype=int)

            fig, ax = plt.subplots(figsize=(16, 8))

            im = ax.imshow(
                heatmap_values,
                aspect="auto",
                origin="lower",
                cmap=cmap,
            )

            fig.suptitle(
                f"Baseline CNN Heatmap, Metric = {metric_name}, Split = {self.split}",
                fontsize=12,
                fontweight="bold",
                color=(0, 0, 1),
            )

            ax.set_xlabel("Receiver ID", fontsize=8)

            if y_axis == "split_sample_index":
                ax.set_ylabel("Sample Index in Split", fontsize=8)
            elif y_axis == "sample_id":
                ax.set_ylabel("Mapped Sample ID", fontsize=8)
            else:
                ax.set_ylabel(y_axis, fontsize=8)

            ax.set_title(
                f"{metric_name}",
                fontsize=10,
                fontweight="normal",
                color=(0, 0, 1),
            )

            ax.tick_params(axis="x", labelsize=8)
            ax.tick_params(axis="y", labelsize=8)

            ax.set_xticks(np.arange(len(x_values)))
            ax.set_xticklabels(x_values)

            n_y = len(y_values)
            if n_y <= 15:
                y_tick_positions = np.arange(n_y)
            else:
                y_tick_positions = np.linspace(0, n_y - 1, 10, dtype=int)

            ax.set_yticks(y_tick_positions)
            ax.set_yticklabels([y_values[i] for i in y_tick_positions])

            if show_colorbar:
                cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
                cbar.ax.tick_params(labelsize=8)
                cbar.set_label(metric_name, fontsize=8)

            if highlight_worst:
                valid_mask = np.isfinite(heatmap_values)

                if np.any(valid_mask):
                    if metric_name == "correlation":
                        flat_index = np.nanargmin(heatmap_values)
                    else:
                        flat_index = np.nanargmax(heatmap_values)

                    iy, ix = np.unravel_index(flat_index, heatmap_values.shape)

                    worst_receiver = int(x_values[ix])
                    worst_sample = y_values[iy]
                    worst_value = float(heatmap_values[iy, ix])

                    ax.plot(
                        ix,
                        iy,
                        marker="s",
                        markersize=8,
                        markerfacecolor="none",
                        markeredgecolor=[1, 0, 0],
                        markeredgewidth=1.5,
                        zorder=3,
                    )

                    ax.text(
                        ix + 0.2,
                        iy,
                        f"Worst case\nST{worst_receiver}, sample={worst_sample}\n{worst_value:.4f}",
                        fontsize=8,
                        color=(1, 0, 0),
                        fontweight="bold",
                        ha="left",
                        va="center",
                    )

            plt.tight_layout()

            if save:
                if figure_path is None:
                    figure_path = self.docs_dir / f"3. baseline_cnn_heatmap_{metric_name}_{self.split}.svg"
                else:
                    figure_path = Path(figure_path)

                plt.savefig(figure_path, dpi=200, bbox_inches="tight")

                print("\n" + "=" * 120)
                print("Detailed metric heatmap figure saved successfully")
                print(figure_path)
                print("=" * 120)
                print("\n")
            plt.show()
            return fig, heatmap_df
        
        
        def plot_detailed_metric_3d_bars(
            self,
            metrics_df=None,
            metric_name="rmse",
            y_axis="split_sample_index",
            cmap="viridis",
            save=False,
            figure_path=None,
            highlight_worst=True,
            elev=28,
            azim=-55,
        ):
            """
            Plot one detailed metric in 3D bar format as a function of sample and receiver.

            Parameters
            ----------
            metrics_df : pandas.DataFrame or None, optional
                Detailed dataframe returned by evaluate().
                If None, the method uses self.last_metrics_df.

            metric_name : str, optional
                Metric column to plot.
                Example:
                - "rmse"
                - "correlation"
                - "relative_l2_error"
                - "peak_arrival_time_abs_error_s"
                - "dominant_frequency_abs_error_Hz"

            y_axis : str, optional
                Column used for the vertical indexing of samples.
                Recommended:
                - "split_sample_index"
                - "sample_id"

            cmap : str, optional
                Colormap used to color the bars according to metric value.

            save : bool, optional
                If True, save the figure.

            figure_path : str or Path or None, optional
                Output path for the figure when save=True.

            highlight_worst : bool, optional
                If True, highlight the worst case in red.

            elev : float, optional
                Elevation angle for the 3D view.

            azim : float, optional
                Azimuth angle for the 3D view.

            Returns
            -------
            fig : matplotlib.figure.Figure
                Figure object.

            heatmap_df : pandas.DataFrame
                Pivoted dataframe used to create the 3D bar plot.
            """
            if metrics_df is None:
                metrics_df = self.last_metrics_df

            if metrics_df is None:
                raise ValueError(
                    "metrics_df is None. Run evaluate() first or pass a valid dataframe."
                )

            required_columns = ["receiver_id", y_axis, metric_name]
            missing_columns = [col for col in required_columns if col not in metrics_df.columns]

            if len(missing_columns) > 0:
                raise ValueError(
                    f"The following required columns are missing in metrics_df: {missing_columns}"
                )

            heatmap_df = metrics_df.pivot_table(
                index=y_axis,
                columns="receiver_id",
                values=metric_name,
                aggfunc="mean",
            )

            heatmap_df = heatmap_df.sort_index(axis=0).sort_index(axis=1)

            z_values = heatmap_df.to_numpy(dtype=float)
            y_values = heatmap_df.index.to_numpy()
            x_values = heatmap_df.columns.to_numpy(dtype=int)

            n_rows, n_cols = z_values.shape

            fig = plt.figure(figsize=(16, 16))
            ax = fig.add_subplot(111, projection="3d")
            ax.set_box_aspect((n_cols, n_rows, 12))

            fig.suptitle(
                f"Baseline CNN 3D Bars, Metric = {metric_name}, Split = {self.split}",fontsize=12,fontweight="bold",color=(0, 0, 1))

            xpos = []
            ypos = []
            zpos = []
            dx = []
            dy = []
            dz = []
            colors = []

            cmap_obj = plt.get_cmap(cmap)
            valid_values = z_values[np.isfinite(z_values)]

            if len(valid_values) == 0:
                raise ValueError(f"No valid values found for metric '{metric_name}'.")

            norm = matplotlib.colors.Normalize(
                vmin=float(np.min(valid_values)),
                vmax=float(np.max(valid_values)),
            )

            for iy in range(n_rows):
                for ix in range(n_cols):
                    value = z_values[iy, ix]

                    if not np.isfinite(value):
                        continue

                    xpos.append(ix)
                    ypos.append(iy)
                    zpos.append(0.0)
                    dx.append(0.7)
                    dy.append(0.7)
                    dz.append(value)
                    colors.append(cmap_obj(norm(value)))

            ax.bar3d(xpos,ypos,zpos,dx,dy,dz,color=colors,edgecolor=[0, 0, 0],linewidth=0.4,shade=True,alpha=0.95,zsort="average")

            ax.set_title(f"{metric_name}",fontsize=10,fontweight="normal",color=(0, 0, 1), pad=12)

            ax.set_xlabel("Receiver ID", fontsize=8, labelpad=10)

            if y_axis == "split_sample_index":
                ax.set_ylabel("Sample Index in Split", fontsize=8, labelpad=10)
            elif y_axis == "sample_id":
                ax.set_ylabel("Mapped Sample ID", fontsize=8, labelpad=10)
            else:
                ax.set_ylabel(y_axis, fontsize=8, labelpad=10)

            ax.set_zlabel(metric_name, fontsize=8, labelpad=8)

            ax.tick_params(axis="x", labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax.tick_params(axis="z", labelsize=8)
            
            ax.xaxis._axinfo["grid"]["linewidth"] = 0
            ax.yaxis._axinfo["grid"]["linewidth"] = 0
            ax.zaxis._axinfo["grid"]["linewidth"] = 0.6
            ax.zaxis._axinfo["grid"]["linestyle"] = "-"
            ax.zaxis._axinfo["grid"]["color"] = (0.7, 0.7, 0.7, 1.0)

            ax.set_xticks(np.arange(n_cols) + 0.35)
            ax.set_xticklabels(x_values)

            ax.set_yticks(np.arange(n_rows) + 0.35)
            ax.set_yticklabels(y_values)

            ax.view_init(elev=elev, azim=azim)
            ax.set_box_aspect((n_cols, n_rows, 8))

            mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
            mappable.set_array([])
            # cbar = fig.colorbar(mappable, ax=ax, pad=0.08, fraction=0.03)
            cbar = fig.colorbar(mappable, ax=ax, pad=0.02, fraction=0.035, shrink=0.92)
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label(metric_name, fontsize=8)

            if highlight_worst:
                valid_mask = np.isfinite(z_values)

                if np.any(valid_mask):
                    if metric_name == "correlation":
                        flat_index = np.nanargmin(z_values)
                    else:
                        flat_index = np.nanargmax(z_values)

                    iy_worst, ix_worst = np.unravel_index(flat_index, z_values.shape)

                    worst_receiver = int(x_values[ix_worst])
                    worst_sample = y_values[iy_worst]
                    worst_value = float(z_values[iy_worst, ix_worst])

                    ax.bar3d(ix_worst,iy_worst, 0.0,0.7, 0.7, worst_value,color=[1, 0, 0], edgecolor=[0, 0, 0],linewidth=3.0, shade=True, alpha=1.0,zsort="average")

                    ax.text(ix_worst + 0.9, iy_worst + 0.1, worst_value,f"Worst case\nST{worst_receiver}, sample={worst_sample}\n{worst_value:.4f}",fontsize=8,
                        color=(1, 0, 0),fontweight="bold",ha="left",va="bottom")
            plt.tight_layout()
            if save:
                if figure_path is None:
                    figure_path = self.docs_dir / f"4. baseline_cnn_3d_bars_{metric_name}_{self.split}.svg"
                else:
                    figure_path = Path(figure_path)

                plt.savefig(figure_path, dpi=200, bbox_inches="tight")

                print("\n" + "=" * 120)
                print("Detailed metric 3D bars figure saved successfully")
                print(figure_path)
                print("=" * 120)
                print("\n")
            plt.show()
            return fig, heatmap_df
else:

    class BaselineCNNPerformanceEvaluator:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"PyTorch is not available in this environment. Original error: {TORCH_IMPORT_ERROR}"
            )


##############################################################################################################################################
##############################################################################################################################################
###########################################       Baseline CNN Spectrogram Comparator       ##################################################
##############################################################################################################################################
##############################################################################################################################################
if TORCH_AVAILABLE:
    class BaselineCNNSpectrogramComparator:
        """
        Spectrogram comparison class for the trained baseline CNN.

        This class is intended for:
        - visual comparison of simulated and predicted seismograms
        - spectrogram inspection for selected receivers
        - absolute difference analysis in dB

        For one selected sample, the class plots:
        - Simulated spectrogram
        - Predicted spectrogram
        - Absolute dB difference spectrogram

        Parameters
        ----------
        h5_path : str or Path
            Path to the HDF5 dataset.

        best_model_path : str or Path
            Path to the trained baseline CNN checkpoint.

        split : str, optional
            Dataset split: "train", "val", or "test".

        normalize_x : bool, optional
            Whether to normalize the velocity model input.

        normalize_y : bool, optional
            Whether to normalize the seismogram target.

        n_receivers : int, optional
            Number of output receivers.

        n_time : int, optional
            Number of time samples in the output.

        NFFT : int, optional
            Window length for spectrogram computation.

        noverlap : int, optional
            Number of overlapping points between windows.

        cmap_main : str, optional
            Colormap used for simulated and predicted spectrograms.

        cmap_diff : str, optional
            Colormap used for absolute dB difference spectrograms.

        db_floor : float, optional
            Minimum dB value to avoid extreme negative values.

        device : str or None, optional
            Torch device. If None, choose automatically.

        docs_dir : str or Path or None, optional
            Directory used to save figures.
        """

        def __init__(
            self,
            h5_path,
            best_model_path,
            split="val",
            normalize_x=False,
            normalize_y=False,
            n_receivers=15,
            n_time=1500,
            NFFT=128,
            noverlap=96,
            cmap_main="viridis",
            cmap_diff="magma",
            db_floor=-120.0,
            device=None,
            docs_dir=None,
        ):
            if split not in ["train", "val", "test"]:
                raise ValueError("split must be one of: 'train', 'val', or 'test'.")

            self.h5_path = Path(h5_path)
            self.best_model_path = Path(best_model_path)
            self.split = split
            self.normalize_x = normalize_x
            self.normalize_y = normalize_y
            self.n_receivers = int(n_receivers)
            self.n_time = int(n_time)
            self.NFFT = int(NFFT)
            self.noverlap = int(noverlap)
            self.cmap_main = cmap_main
            self.cmap_diff = cmap_diff
            self.db_floor = float(db_floor)

            if not self.h5_path.exists():
                raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

            if not self.best_model_path.exists():
                raise FileNotFoundError(f"Best model file not found: {self.best_model_path}")

            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)

            if docs_dir is None:
                self.docs_dir = self.h5_path.parent.parent.parent / "docs"
            else:
                self.docs_dir = Path(docs_dir)

            self.docs_dir.mkdir(parents=True, exist_ok=True)

            self.dataset = HDF5SurfaceSeismogramTorchDataset(
                h5_path=self.h5_path,
                split=self.split,
                normalize_x=self.normalize_x,
                normalize_y=self.normalize_y,
                return_metadata=True,
            )

            self.model = BaselineCNNEncoderDecoder(
                n_receivers=self.n_receivers,
                n_time=self.n_time,
            ).to(self.device)

            try:
                checkpoint = torch.load(
                    self.best_model_path,
                    map_location=self.device,
                    weights_only=False,
                )
            except TypeError:
                checkpoint = torch.load(
                    self.best_model_path,
                    map_location=self.device,
                )

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.eval()

        def _compute_spectrogram(self, trace, dt):
            """
            Compute spectrogram and return it in dB scale.
            """
            eps = 1e-20
            fs = 1.0 / dt

            Pxx, frequencies, times = mlab.specgram(
                x=np.asarray(trace, dtype=float),
                NFFT=self.NFFT,
                Fs=fs,
                noverlap=self.noverlap,
            )

            Pxx_dB = 10.0 * np.log10(Pxx + eps)
            Pxx_dB = np.maximum(Pxx_dB, self.db_floor)

            return Pxx, Pxx_dB, frequencies, times

        def _default_receivers_to_plot(self, n_receivers_to_plot=4):
            """
            Select a default subset of evenly distributed receivers.
            """
            n_receivers_to_plot = int(n_receivers_to_plot)

            if n_receivers_to_plot <= 0:
                raise ValueError("n_receivers_to_plot must be positive.")

            if self.n_receivers <= n_receivers_to_plot:
                return list(range(self.n_receivers))

            receivers = np.linspace(
                0,
                self.n_receivers - 1,
                n_receivers_to_plot,
                dtype=int,
            )

            return list(np.unique(receivers))

        def plot_spectrogram_comparison(
            self,
            sample_index,
            result_dict=None,
            receivers_to_plot=None,
            fmax=None,
            common_color_scale=True,
            common_diff_scale=True,
            save=False,
            figure_path=None,
        ):
            """
            Plot simulated, predicted, and absolute dB difference spectrograms
            for selected receivers of one sample.

            Parameters
            ----------
            sample_index : int
                Index inside the selected split dataset.
                This is used only when result_dict is None.

            result_dict : dict or None, optional
                Optional dictionary with precomputed signals.
                If provided, the method uses:
                - result_dict["y_true"] if available
                - result_dict["y_pred_filtered"] if available
                - otherwise result_dict["y_pred"]
                - result_dict["metadata"]

                In this mode, the model is not evaluated again.

            receivers_to_plot : list[int] or None, optional
                Receiver indices to plot using 0-based indexing.
                If None, 4 receivers are selected automatically.

            fmax : float or None, optional
                Maximum frequency to display.

            common_color_scale : bool, optional
                If True, use one common color scale for simulated and predicted panels.

            common_diff_scale : bool, optional
                If True, use one common color scale for difference panels.

            save : bool, optional
                If True, save the figure.

            figure_path : str or Path or None, optional
                Output path for the figure when save=True.

            Returns
            -------
            output : dict
                Dictionary containing sample arrays, metadata, and spectrogram results.
            """
            if result_dict is None:
                if sample_index < 0 or sample_index >= len(self.dataset):
                    raise IndexError(
                        f"sample_index={sample_index} is out of range [0, {len(self.dataset) - 1}]"
                    )

                if receivers_to_plot is None:
                    receivers_to_plot = self._default_receivers_to_plot(n_receivers_to_plot=4)

                receivers_to_plot = [int(r) for r in receivers_to_plot]

                for receiver_id in receivers_to_plot:
                    if receiver_id < 0 or receiver_id >= self.n_receivers:
                        raise IndexError(
                            f"receiver_id={receiver_id} is out of range [0, {self.n_receivers - 1}]"
                        )

                x_true, y_true, metadata = self.dataset[sample_index]

                with torch.no_grad():
                    x_input = x_true.unsqueeze(0).to(self.device)
                    y_pred = self.model(x_input)

                x_true_np = x_true.squeeze(0).cpu().numpy()
                y_true_np = y_true.cpu().numpy()
                y_pred_np = y_pred.squeeze(0).cpu().numpy()
                dt = float(metadata["dt"])
                mapped_sample_id = int(metadata["sample_id"])
                receiver_x = metadata["receiver_x"].astype(int)
                receiver_z = metadata["receiver_z"].astype(int)
            else:
                if not isinstance(result_dict, dict):
                    raise TypeError("result_dict must be a dictionary with precomputed baseline CNN results.")

                if "metadata" not in result_dict:
                    raise ValueError("result_dict must contain key 'metadata'.")

                if "y_true" not in result_dict:
                    raise ValueError("result_dict must contain key 'y_true' for spectrogram comparison.")

                if "y_pred_filtered" in result_dict:
                    y_pred_source = result_dict["y_pred_filtered"]
                elif "y_pred" in result_dict:
                    y_pred_source = result_dict["y_pred"]
                else:
                    raise ValueError("result_dict must contain key 'y_pred' or 'y_pred_filtered'.")

                metadata = result_dict["metadata"]
                y_true_np = np.asarray(result_dict["y_true"], dtype=float)
                y_pred_np = np.asarray(y_pred_source, dtype=float)
                x_true_np = None if result_dict.get("x_true", None) is None else np.asarray(result_dict["x_true"])
                dt = float(metadata["dt"])
                mapped_sample_id = int(result_dict.get("sample_id", metadata.get("sample_id", -1)))
                sample_index = int(result_dict.get("sample_index", sample_index))
                receiver_x = metadata["receiver_x"].astype(int)
                receiver_z = metadata["receiver_z"].astype(int)

                if receivers_to_plot is None:
                    if result_dict.get("receivers_to_plot", None) is not None:
                        receivers_to_plot = [int(r) for r in result_dict["receivers_to_plot"]]
                    else:
                        receivers_to_plot = self._default_receivers_to_plot(n_receivers_to_plot=4)
                else:
                    receivers_to_plot = [int(r) for r in receivers_to_plot]

                for receiver_id in receivers_to_plot:
                    if receiver_id < 0 or receiver_id >= y_pred_np.shape[0]:
                        raise IndexError(
                            f"receiver_id={receiver_id} is out of range [0, {y_pred_np.shape[0] - 1}]"
                        )

            spectrogram_results = []

            for rec in receivers_to_plot:
                _, Pxx_true_dB, frequencies, times = self._compute_spectrogram(y_true_np[rec], dt)
                _, Pxx_pred_dB, _, _ = self._compute_spectrogram(y_pred_np[rec], dt)

                abs_diff_dB = np.abs(Pxx_true_dB - Pxx_pred_dB)

                if fmax is not None:
                    mask_f = frequencies <= fmax
                    frequencies_plot = frequencies[mask_f]
                    Pxx_true_plot = Pxx_true_dB[mask_f, :]
                    Pxx_pred_plot = Pxx_pred_dB[mask_f, :]
                    abs_diff_plot = abs_diff_dB[mask_f, :]
                else:
                    frequencies_plot = frequencies
                    Pxx_true_plot = Pxx_true_dB
                    Pxx_pred_plot = Pxx_pred_dB
                    abs_diff_plot = abs_diff_dB

                spectrogram_results.append(
                    {
                        "receiver_id": rec + 1,
                        "receiver_index_0_based": rec,
                        "receiver_x": int(receiver_x[rec]),
                        "receiver_z": int(receiver_z[rec]),
                        "frequencies": frequencies_plot,
                        "times": times,
                        "Pxx_true_dB": Pxx_true_plot,
                        "Pxx_pred_dB": Pxx_pred_plot,
                        "Pxx_abs_diff_dB": abs_diff_plot,
                    }
                )

            if common_color_scale:
                all_main_db = np.concatenate(
                    [
                        np.concatenate(
                            [
                                item["Pxx_true_dB"].ravel(),
                                item["Pxx_pred_dB"].ravel(),
                            ]
                        )
                        for item in spectrogram_results
                    ]
                )
                vmin_main = np.percentile(all_main_db, 5)
                vmax_main = np.percentile(all_main_db, 95)
            else:
                vmin_main = None
                vmax_main = None

            if common_diff_scale:
                all_diff_db = np.concatenate(
                    [item["Pxx_abs_diff_dB"].ravel() for item in spectrogram_results]
                )
                vmin_diff = np.percentile(all_diff_db, 5)
                vmax_diff = np.percentile(all_diff_db, 95)
            else:
                vmin_diff = None
                vmax_diff = None

            nrows = len(receivers_to_plot)
            ncols = 3

            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(18, 3.8 * nrows),
                constrained_layout=True,
            )

            axes = np.atleast_2d(axes)

            fig.suptitle(f"Baseline CNN Spectrogram Comparison | Split = {self.split} | sample_index = {sample_index} | mapped_sample_id = {mapped_sample_id}",
                fontsize=12,fontweight="bold",color=(0, 0, 1))

            last_im_main = None
            last_im_diff = None

            for i, item in enumerate(spectrogram_results):
                rec_id = item["receiver_id"]
                rec_x = item["receiver_x"]
                rec_z = item["receiver_z"]
                frequencies_plot = item["frequencies"]
                times = item["times"]

                # ---------------- Simulated ----------------
                ax = axes[i, 0]
                last_im_main = ax.imshow(item["Pxx_true_dB"],origin="lower",aspect="auto", extent=[times[0], times[-1], frequencies_plot[0], frequencies_plot[-1]],cmap=self.cmap_main,vmin=vmin_main,vmax=vmax_main)
                ax.set_title(f"Simulated | ST{rec_id} | ix={rec_x}, iz={rec_z}",fontsize=10, fontweight="bold", color=(0, 0, 1))
                ax.set_xlabel("Time (s)", fontsize=8)
                ax.set_ylabel("Frequency (Hz)", fontsize=8)
                ax.tick_params(axis="x", labelsize=8)
                ax.tick_params(axis="y", labelsize=8)

                # ---------------- Predicted ----------------
                ax = axes[i, 1]
                ax.imshow(item["Pxx_pred_dB"],origin="lower",aspect="auto",extent=[times[0], times[-1], frequencies_plot[0], frequencies_plot[-1]],cmap=self.cmap_main,vmin=vmin_main,vmax=vmax_main)
                ax.set_title(f"Predicted | ST{rec_id}",fontsize=10,fontweight="bold",color=(0, 0, 1))
                ax.set_xlabel("Time (s)", fontsize=8)
                ax.set_ylabel("Frequency (Hz)", fontsize=8)
                ax.tick_params(axis="x", labelsize=8)
                ax.tick_params(axis="y", labelsize=8)

                # ---------------- Absolute dB difference ----------------
                ax = axes[i, 2]
                last_im_diff = ax.imshow(item["Pxx_abs_diff_dB"], origin="lower",aspect="auto",extent=[times[0], times[-1], frequencies_plot[0], frequencies_plot[-1]],cmap=self.cmap_diff,vmin=vmin_diff,vmax=vmax_diff)
                ax.set_title(f"Absolute dB Difference | ST{rec_id}", fontsize=10, fontweight="bold",color=(0, 0, 1))
                ax.set_xlabel("Time (s)", fontsize=8)
                ax.set_ylabel("Frequency (Hz)", fontsize=8)
                ax.tick_params(axis="x", labelsize=8)
                ax.tick_params(axis="y", labelsize=8)

            if last_im_main is not None:
                cbar_main = fig.colorbar(last_im_main,ax=axes[:, :2].ravel().tolist(),pad=0.01,fraction=0.02, shrink=0.92)
                cbar_main.set_label("Power Spectral Density (dB)")

            if last_im_diff is not None:
                cbar_diff = fig.colorbar(last_im_diff,ax=axes[:, 2].ravel().tolist(),pad=0.01,fraction=0.02, shrink=0.92)
                cbar_diff.set_label("Absolute Difference (dB)")

            if save:
                if figure_path is None:
                    figure_path = self.docs_dir / f"6. baseline_cnn_spectrogram_comparison_sample_{sample_index}_{self.split}.svg"
                else:
                    figure_path = Path(figure_path)

                plt.savefig(figure_path, dpi=200, bbox_inches="tight")

                print("\n" + "=" * 120)
                print("Baseline CNN spectrogram comparison figure saved successfully")
                print(figure_path)
                print("=" * 120)
                print("\n")

            plt.show()

            return {
                "sample_index": sample_index,
                "sample_id": mapped_sample_id,
                "receivers_to_plot": receivers_to_plot,
                "x_true": x_true_np,
                "y_true": y_true_np,
                "y_pred": y_pred_np,
                "metadata": metadata,
                "spectrogram_results": spectrogram_results,
            }

    ##############################################################################################################################################
    ##############################################################################################################################################
    ###########################################       Baseline CNN Bandpass Comparator       #####################################################
    ##############################################################################################################################################
    ##############################################################################################################################################
    class BaselineCNNBandpassComparator:
        """
        Bandpass post-processing class for baseline CNN results.

        This class is intended for:
        - filtering predicted traces already generated by previous baseline CNN classes,
        - returning a new result dictionary that can be reused by other plotting classes.

        Important note
        --------------
        This class does not compute predictions and does not create plots.
        It only applies a simple frequency-domain bandpass filter to existing signals.

        Parameters
        ----------
        result_dict : dict
            Dictionary returned by a previous baseline CNN method.
            It must contain at least:
            - "y_pred"
            - "metadata"

            Optional keys preserved in the output:
            - "y_true"
            - "sample_index"
            - "sample_id"
            - "receivers_to_plot"

        fmin : float, optional
            Lower cutoff frequency in Hz for the bandpass filter.

        fmax : float or None, optional
            Upper cutoff frequency in Hz for the bandpass filter.
            If None, only the lower cutoff is applied.
        """

        def __init__(
            self,
            result_dict,
            fmin=1.0,
            fmax=None,
        ):
            if not isinstance(result_dict, dict):
                raise TypeError("result_dict must be a dictionary returned by a previous baseline CNN method.")

            if "y_pred" not in result_dict:
                raise ValueError("result_dict must contain key 'y_pred'.")

            if "metadata" not in result_dict:
                raise ValueError("result_dict must contain key 'metadata'.")

            self.result_dict = result_dict
            self.y_pred = np.asarray(result_dict["y_pred"], dtype=float)
            self.metadata = result_dict["metadata"]
            self.n_receivers = int(self.y_pred.shape[0])
            self.fmin = float(fmin)
            self.fmax = None if fmax is None else float(fmax)

            if self.fmin < 0.0:
                raise ValueError("fmin must be non-negative.")

            if self.fmax is not None and self.fmax <= self.fmin:
                raise ValueError("fmax must be greater than fmin.")

            if self.y_pred.ndim != 2:
                raise ValueError("result_dict['y_pred'] must have shape (n_receivers, n_time).")

            if "dt" not in self.metadata:
                raise ValueError("result_dict['metadata'] must contain key 'dt'.")

            self.dt = float(self.metadata["dt"])

        def _apply_bandpass_filter(self, trace, dt):
            """
            Apply a simple FFT-based bandpass filter to one trace.
            """
            trace = np.asarray(trace, dtype=float).ravel()

            if trace.size == 0:
                return trace.copy()

            freqs = np.fft.rfftfreq(trace.size, d=dt)
            spectrum = np.fft.rfft(trace)

            mask = freqs >= self.fmin

            if self.fmax is not None:
                mask = mask & (freqs <= self.fmax)

            filtered_spectrum = spectrum * mask.astype(float)
            filtered_trace = np.fft.irfft(filtered_spectrum, n=trace.size)

            return filtered_trace

        def filter_signals(self, receiver_ids=None):
            """
            Filter predicted traces and return a new result dictionary.

            Parameters
            ----------
            receiver_ids : list[int] or None, optional
                Receiver indices to filter using 0-based indexing.
                If None, all receivers are filtered.

            Returns
            -------
            filtered_result_dict : dict
                Copy of the input result dictionary with:
                - "y_pred_filtered"
                - "bandpass_filter"
            """
            if receiver_ids is None:
                receiver_ids = list(range(self.n_receivers))
            else:
                receiver_ids = [int(r) for r in receiver_ids]

            for receiver_id in receiver_ids:
                if receiver_id < 0 or receiver_id >= self.n_receivers:
                    raise IndexError(
                        f"receiver_id={receiver_id} is out of range [0, {self.n_receivers - 1}]"
                    )

            y_pred_filtered = np.array(self.y_pred, copy=True)

            for rec in receiver_ids:
                y_pred_filtered[rec] = self._apply_bandpass_filter(self.y_pred[rec], self.dt)

            filtered_result_dict = dict(self.result_dict)
            filtered_result_dict["y_pred_filtered"] = y_pred_filtered
            filtered_result_dict["bandpass_filter"] = {
                "fmin": self.fmin,
                "fmax": self.fmax,
                "dt": self.dt,
                "receiver_ids": receiver_ids,
                "method": "fft_bandpass",
            }

            return filtered_result_dict

else:

    class BaselineCNNSpectrogramComparator:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"PyTorch is not available in this environment. Original error: {TORCH_IMPORT_ERROR}"
            )

    class BaselineCNNBandpassComparator:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"PyTorch is not available in this environment. Original error: {TORCH_IMPORT_ERROR}"
            )




##############################################################################################################################################
##############################################################################################################################################
############################################ Loss history Conditioned Recivers (FourierONet)  ################################################
##############################################################################################################################################
##############################################################################################################################################

class ReceiverConditionedLossHistoryReviewer:                                                    # Class to load, summarize, and plot receiver-conditioned CNN loss histories
    """
    Review class for receiver-conditioned CNN loss history.

    This class loads the NPZ file generated during training and produces:
    - a numerical summary,
    - one dashboard figure with multiple subplots,
    - and one individual figure for epoch train/validation losses.

    Current expected keys inside the NPZ file
    -----------------------------------------
    - train_losses
    - val_losses
    - train_batch_losses
    - train_batch_steps
    - epoch_times_sec
    - best_epoch
    - best_val_loss
    - num_epochs
    - batch_size_query
    """

    def __init__(self, loss_history_path, outputs_dir):                                          # Initialize review class with NPZ path and outputs directory
        self.loss_history_path = Path(loss_history_path)                                          # Store NPZ history path
        self.outputs_dir = Path(outputs_dir)                                                      # Store outputs directory
        self.outputs_dir.mkdir(parents=True, exist_ok=True)                                       # Create outputs directory if it does not exist

        if not self.loss_history_path.exists():                                                   # Check that the NPZ file exists
            raise FileNotFoundError(f"Loss history file not found: {self.loss_history_path}")     # Stop execution if NPZ file does not exist

        self.history = None                                                                       # Placeholder for NPZ history object

        self.train_losses = None                                                                  # Placeholder for epoch train losses
        self.val_losses = None                                                                    # Placeholder for epoch validation losses
        
        self.train_batch_losses = None                                                            # Placeholder for sampled train-batch losses
        self.train_batch_steps = None                                                             # Placeholder for sampled global train-step indices
        self.epoch_times_sec = None                                                               # Placeholder for epoch computation times
        
        self.train_batch_losses = None                                                            # Placeholder for sampled train-batch losses
        self.train_batch_steps = None                                                             # Placeholder for sampled global train-step indices
        self.train_batch_epochs = None                                                            # Placeholder for epoch index of each sampled train-batch loss
        self.train_batch_step_in_epoch = None                                                     # Placeholder for batch index inside epoch of each sampled train-batch loss
        self.epoch_times_sec = None                                                               # Placeholder for epoch computation times
        
        
        self.best_epoch = None                                                                    # Placeholder for best epoch
        self.best_val_loss = None                                                                 # Placeholder for best validation loss
        self.num_epochs = None                                                                    # Placeholder for number of epochs
        self.batch_size_query = None                                                              # Placeholder for batch size

        self.epochs = None                                                                        # Placeholder for epoch indices
        self.gap_losses = None                                                                    # Placeholder for validation-train gap
                                    

    def load_history(self):                                                                       # Load all available histories from the NPZ file
        self.history = np.load(self.loss_history_path)                                            # Load compressed NPZ file

        self.train_losses = self.history["train_losses"]                                          # Load train losses per epoch
        self.val_losses = self.history["val_losses"]                                              # Load validation losses per epoch

        self.train_batch_losses = self.history["train_batch_losses"] if "train_batch_losses" in self.history.files else None              # Load sampled train-batch losses if available
        self.train_batch_steps = self.history["train_batch_steps"] if "train_batch_steps" in self.history.files else None                 # Load sampled global train-step indices if available
        self.train_batch_epochs = self.history["train_batch_epochs"] if "train_batch_epochs" in self.history.files else None              # Load epoch index of each sampled train-batch loss if available
        self.train_batch_step_in_epoch = self.history["train_batch_step_in_epoch"] if "train_batch_step_in_epoch" in self.history.files else None  # Load batch index inside epoch if available
        self.epoch_times_sec = self.history["epoch_times_sec"] if "epoch_times_sec" in self.history.files else None                      # Load epoch computation times if available

        self.best_epoch = int(self.history["best_epoch"]) if "best_epoch" in self.history.files else int(np.argmin(self.val_losses) + 1)    # Load best epoch if available, otherwise infer it
        self.best_val_loss = float(self.history["best_val_loss"]) if "best_val_loss" in self.history.files else float(np.min(self.val_losses))  # Load best validation loss if available, otherwise infer it
        self.num_epochs = int(self.history["num_epochs"]) if "num_epochs" in self.history.files else int(len(self.train_losses))               # Load number of epochs if available, otherwise infer it
        self.batch_size_query = int(self.history["batch_size_query"]) if "batch_size_query" in self.history.files else None                    # Load batch size if available

        self.epochs = np.arange(1, len(self.train_losses) + 1)                                    # Build epoch index vector
        self.gap_losses = self.val_losses - self.train_losses                                     # Compute validation-train gap

    def print_summary(self):                                                                      # Print summary information of the loaded history
        print("\n" + "=" * 120)                                                                   # Print separator line
        print("Receiver-conditioned loss history loaded successfully")                            # Print success message
        print(f"Available keys             : {self.history.files}")                               # Print all keys stored in the NPZ file
        print(f"Number of epochs           : {self.num_epochs}")                                  # Print number of epochs
        print(f"Best epoch                 : {self.best_epoch}")                                  # Print best epoch
        print(f"Best validation loss       : {self.best_val_loss:.6f}")                           # Print best validation loss
        print(f"Initial train loss         : {float(self.train_losses[0]):.6f}")                  # Print initial training loss
        print(f"Final train loss           : {float(self.train_losses[-1]):.6f}")                 # Print final training loss
        print(f"Initial validation loss    : {float(self.val_losses[0]):.6f}")                    # Print initial validation loss
        print(f"Final validation loss      : {float(self.val_losses[-1]):.6f}")                   # Print final validation loss
        print(f"Final generalization gap   : {float(self.gap_losses[-1]):.6f}")                   # Print final gap between validation and training loss

        if self.train_batch_losses is not None and self.train_batch_steps is not None:            # Print sampled batch information if available
            print(f"Saved train-batch points    : {len(self.train_batch_losses)}")                # Print number of saved train-batch points

        if self.epoch_times_sec is not None:                                                      # Print epoch-time information if available
            print(f"Mean epoch time [sec]       : {float(np.mean(self.epoch_times_sec)):.3f}")    # Print mean epoch time in seconds
            print(f"Total epoch time [sec]      : {float(np.sum(self.epoch_times_sec)):.3f}")     # Print total accumulated epoch time in seconds

        if self.batch_size_query is not None:                                                     # Print batch size if available
            print(f"Batch size query            : {self.batch_size_query}")                       # Print query batch size

        print("=" * 120)                                                                          # Print separator line
        print("\n")                                                                               # Print blank line

    def plot_dashboard(self):                                                                     # Create one dashboard figure with multiple subplots
        fig, axes = plt.subplots(3, 2, figsize=(18, 13), constrained_layout=True)                 # Create 3x2 dashboard figure
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()                                             # Flatten subplot axes

        #---------------- Panel 1: epoch train vs validation loss ----------------
        ax1.plot(self.epochs, self.train_losses, lw=1.0, ls="-", color=[0, 0, 0], label="Train Loss")          # Plot train loss per epoch
        ax1.plot(self.epochs, self.val_losses, lw=1.0, ls="-", color=[0, 0, 1], label="Validation Loss")       # Plot validation loss per epoch
        ax1.axvline(self.best_epoch, ls="--", lw=2.0, color=[1, 0, 0], label=f"Best Epoch = {self.best_epoch}") # Plot best epoch
        ax1.set_title("Epoch Loss History", fontsize=10, fontweight="bold", color=(0, 0, 1))      # Set subplot title
        ax1.set_xlabel("Epoch")                                                                   # Label x axis
        ax1.set_ylabel("Loss")                                                                    # Label y axis
        ax1.grid(visible=True, axis="x")                                                          # Show x-axis grid
        ax1.set_xlim(1, self.num_epochs)                                                          # Set x-axis limits to match number of epochs
        ax1.set_ylim(0, max(float(np.max(self.train_losses)), float(np.max(self.val_losses))) * 1.1)  # Set y-axis limits with some margin
        ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0, fontsize=8, frameon=False)  # Place legend outside axis

        #---------------- Panel 2: generalization gap ----------------
        ax2.plot(self.epochs, self.gap_losses, lw=1.5, ls="-", color=[1, 0, 0], label="Validation - Train")    # Plot generalization gap
        ax2.axhline(0.0, ls="--", lw=1.0, color=[0, 0, 0], label="Zero Gap")                      # Plot zero-gap reference line
        ax2.set_title("Generalization Gap", fontsize=10, fontweight="bold", color=(0, 0, 1))      # Set subplot title
        ax2.set_xlabel("Epoch")                                                                   # Label x axis
        ax2.set_ylabel("Gap")                                                                     # Label y axis
        ax2.grid(visible=True, axis="x")                                                          # Show x-axis grid
        ax2.set_xlim(1, self.num_epochs)                                                          # Set x-axis limits to match number of epochs
        ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0, fontsize=8, frameon=False)  # Place legend outside axis

        #---------------- Panel 3: semilog epoch losses ----------------
        ax3.semilogy(self.epochs, self.train_losses, lw=1.0, ls="-", color=[0, 0, 0], label="Train Loss")      # Plot train loss in semilog scale
        ax3.semilogy(self.epochs, self.val_losses, lw=1.0, ls="-", color=[0, 0, 1], label="Validation Loss")   # Plot validation loss in semilog scale
        ax3.axvline(self.best_epoch, ls="--", lw=2.0, color=[1, 0, 0], label=f"Best Epoch = {self.best_epoch}") # Plot best epoch
        ax3.set_title("Epoch Loss History (Semilog)", fontsize=10, fontweight="bold", color=(0, 0, 1))         # Set subplot title
        ax3.set_xlabel("Epoch")                                                                   # Label x axis
        ax3.set_ylabel("Loss (log scale)")                                                        # Label y axis
        ax3.grid(which="major", axis="y", ls="-", lw=0.8)                                         # Show major grid lines for y axis
        ax3.grid(which="minor", axis="y", ls="-", lw=0.5)                                         # Show minor grid lines for y axis
        ax3.grid(which="major", axis="x", ls="-", lw=0.8)                                         # Show major grid lines for x axis
        ax3.set_ylim(bottom= 0, top=max(float(np.max(self.train_losses)), float(np.max(self.val_losses))) * 1.1)  # Set y-axis limits with some margin
        ax3.set_xlim(1, self.num_epochs)                                                          # Set x-axis limits to match number of epochs
        ax3.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0, fontsize=8, frameon=False)  # Place legend outside axis

        #---------------- Panel 4: sampled train-batch losses separated by epoch ----------------
        if (
            self.train_batch_losses is not None
            and self.train_batch_epochs is not None
            and self.train_batch_step_in_epoch is not None
        ):                                                                                        # Plot sampled train-batch losses separated by epoch if available

            unique_epochs = np.unique(self.train_batch_epochs)                                    # Extract unique epoch indices
            n_epochs_plot = len(unique_epochs)                                                    # Count how many epochs will be plotted

            gray_values = np.linspace(0.85, 0.15, n_epochs_plot)                                  # Create grayscale values from light gray to dark gray

            for i, epoch_id in enumerate(unique_epochs):                                          # Loop over unique epochs
                mask = self.train_batch_epochs == epoch_id                                        # Select sampled points belonging to the current epoch

                ax4.plot(
                    self.train_batch_step_in_epoch[mask],                                         # Plot batch index inside epoch on x axis
                    self.train_batch_losses[mask],                                                # Plot sampled train-batch loss on y axis
                    lw=0.8,
                    ls="-",
                    alpha = 0.7,
                    color=[gray_values[i], gray_values[i], gray_values[i]],                       # Use grayscale color for the current epoch
                    label=f"Epoch {int(epoch_id)}",
                )
            
            handles, labels = ax4.get_legend_handles_labels()                                     # Get all plotted lines and labels
            n_labels = min(20, len(handles))                                                      # Number of legend entries to show
            idx = np.sort(np.random.choice(len(handles), size=n_labels, replace=False))

            ax4.plot(self.train_batch_step_in_epoch[self.best_epoch == self.train_batch_epochs], 
                     self.train_batch_losses[self.best_epoch == self.train_batch_epochs], 
                     lw=1.5, ls="-", color=[0, 0, 1], label=f"Best Epoch = {self.best_epoch}")    # Highlight best epoch 
            
            ax4.set_title("Sampled Train-Batch Loss per Epoch", fontsize=10, fontweight="bold", color=(0, 0, 1))  # Set subplot title
            ax4.set_xlabel("Batch index inside epoch")                                            # Label x axis
            ax4.set_ylabel("Batch Loss")                                                          # Label y axis
            # ax4.grid(visible=True, axis="x")                                                    # Show x-axis grid
            # ax4.legend(handles=sample_labels, loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0, fontsize=8, frameon=False)  # Place legend outside axis
            ax4.legend( 
                        [handles[j] for j in idx],                                                        # Selected line handles
                        [labels[j] for j in idx],                                                         # Selected labels
                        loc='center left',
                        bbox_to_anchor=(1.02, 0.5),
                        borderaxespad=0,
                        fontsize=8,
                        frameon=False
                    )                                                                                     # Place legend outside axis with selected entries
            ax4.set_xlim(1, max(self.train_batch_step_in_epoch) + 1)                              # Set x-axis limits to match batch indiceshs
            ax4.set_ylim(0, max(float(np.max(self.train_batch_losses)), float(np.max(self.train_losses))) * 1.1)  # Set y-axis limits with some margin
        else:
            ax4.axis("off")                                                                       # Turn off panel if epoch-separated batch history is unavailable
            ax4.text(0.5, 0.5, "train_batch_epochs or train_batch_step_in_epoch not available", ha="center", va="center", fontsize=10, fontweight="bold", color=(0, 0, 1))  # Show placeholder text

        #---------------- Panel 5: epoch computation time ----------------
        if self.epoch_times_sec is not None:                                                      # Plot epoch times if available
            ax5.plot(self.epochs, self.epoch_times_sec, lw=1.0, ls="-", color=[0, 0.5, 0], label="Epoch Time [sec]")  # Plot epoch time in seconds
            ax5.set_title("Computation Time per Epoch", fontsize=10, fontweight="bold", color=(0, 0, 1))           # Set subplot title
            ax5.set_xlabel("Epoch")                                                               # Label x axis
            ax5.set_ylabel("Time [sec]")                                                          # Label y axis
            ax5.grid(visible=True, axis="x")                                                      # Show x-axis grid
            ax5.set_xlim(1, self.num_epochs)                                                      # Set x-axis limits to match number of epochs
            ax5.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0, fontsize=8, frameon=False)  # Place legend outside axis
        else:
            ax5.axis("off")                                                                       # Turn off panel if epoch-time history is unavailable
            ax5.text(0.5, 0.5, "epoch_times_sec not available", ha="center", va="center", fontsize=10, fontweight="bold", color=(0, 0, 1))  # Show placeholder text

        #---------------- Panel 6: text summary ----------------
        ax6.axis("off")                                                                           # Turn off axes for summary-text panel
        summary_text = (
            f"Receiver-Conditioned CNN Summary\n\n"
            f"Epochs = {self.num_epochs}\n"
            f"Best epoch = {self.best_epoch}\n"
            f"Best val loss = {self.best_val_loss:.6f}\n"
            f"Initial train loss = {float(self.train_losses[0]):.6f}\n"
            f"Final train loss = {float(self.train_losses[-1]):.6f}\n"
            f"Initial val loss = {float(self.val_losses[0]):.6f}\n"
            f"Final val loss = {float(self.val_losses[-1]):.6f}\n"
            f"Final gap = {float(self.gap_losses[-1]):.6f}\n"
            f"Batch size query = {self.batch_size_query}"
        )                                                                                         # Build summary text for dashboard
        ax6.text(0.03, 0.97, summary_text, ha="left", va="top", fontsize=10, fontweight="bold", color=(0, 0, 1), transform=ax6.transAxes)  # Draw summary text

        fig.suptitle("Receiver-Conditioned CNN Loss Dashboard (Using FourierOnet)", fontsize=12, fontweight="bold", color=(0, 0, 1))  # Set dashboard title

        figure_name = "1.receiver_conditioned_cnn_loss_dashboard_FourierOnet.svg"                 # Define dashboard output figure name
        figure_path = self.outputs_dir / figure_name                                              # Define dashboard output figure path
        plt.savefig(figure_path, dpi=200, bbox_inches="tight")                                    # Save dashboard figure
        plt.show()                                                                                # Display dashboard figure

        print("\n" + "=" * 120)                                                                   # Print separator line
        print("Receiver-conditioned loss dashboard saved successfully")                           # Print confirmation message
        print(figure_path)                                                                        # Print saved dashboard path
        print("=" * 120)                                                                          # Print separator line
        print("\n")                                                                               # Print blank line


    def run_all(self):                                                                            # Execute complete review workflow
        self.load_history()                                                                       # Load NPZ data
        self.print_summary()                                                                      # Print numerical summary
        self.plot_dashboard()                                                                     # Create dashboard figure
        


##############################################################################################################################################
##############################################################################################################################################
##########################       Reciver Conditioned Coordinates CNN (FourierOnet) Performance Evaluator       ###############################
##############################################################################################################################################
##############################################################################################################################################

if TORCH_AVAILABLE:
    class ReceiverConditionedShuffledCoordsReviewer:
        """
        Focused reviewer for the trained receiver-conditioned CNN (FourierOnet) using
        correct receiver coordinates versus shuffled receiver coordinates.

        What this class does
        --------------------
        This class is intentionally limited to:
        - loading one trained receiver-conditioned model,
        - loading one full sample from the selected split,
        - reconstructing the full receiver gather by querying one receiver at a time,
        - comparing:
            1. correct receiver coordinates
            2. shuffled receiver coordinates
        - computing per-receiver metrics,
        - plotting:
            a) metric summary
            b) trace review
        """

        def __init__(
            self,
            h5_path,
            best_model_path,
            split="val",
            normalize_x=False,
            normalize_y=False,
            device=None,
            outputs_dir=None,
        ):
            self.h5_path = Path(h5_path)                                                         # Store HDF5 dataset path
            self.best_model_path = Path(best_model_path)                                         # Store trained checkpoint path
            self.split = str(split)                                                              # Store dataset split
            self.normalize_x = bool(normalize_x)                                                 # Store input normalization flag
            self.normalize_y = bool(normalize_y)                                                 # Store output normalization flag

            if self.split not in ["train", "val", "test"]:                                       # Validate split name
                raise ValueError("split must be one of: 'train', 'val', or 'test'.")

            if not self.h5_path.exists():                                                        # Check that HDF5 file exists
                raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

            if not self.best_model_path.exists():                                                # Check that checkpoint file exists
                raise FileNotFoundError(f"Best model file not found: {self.best_model_path}")

            if device is None:                                                                   # Select device automatically if user did not provide one
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)

            if outputs_dir is None:                                                              # Define default outputs directory if none was provided
                self.outputs_dir = self.h5_path.parent.parent.parent / "outputs"
            else:
                self.outputs_dir = Path(outputs_dir)

            self.outputs_dir.mkdir(parents=True, exist_ok=True)                                  # Create outputs directory if needed

            self.dataset = HDF5SurfaceSeismogramTorchDataset(                                    # Reuse full-sample dataset already available in core
                h5_path=self.h5_path,
                split=self.split,
                normalize_x=self.normalize_x,
                normalize_y=self.normalize_y,
                return_metadata=True,
            )

            self.checkpoint = None                                                               # Placeholder for loaded checkpoint
            self.model = None                                                                    # Placeholder for rebuilt receiver-conditioned model

            self.last_result_dict = None                                                         # Placeholder for last reviewed sample
            self.last_metrics_df = None                                                          # Placeholder for last per-receiver metrics table
            self.last_overall_summary_df = None                                                  # Placeholder for last overall summary table

            self._load_checkpoint_and_model()                                                    # Load model immediately during initialization

        def _load_checkpoint_and_model(self):
            try:
                self.checkpoint = torch.load(                                                    # Load trained checkpoint with modern signature if available
                    self.best_model_path,
                    map_location=self.device,
                    weights_only=False,
                )
            except TypeError:
                self.checkpoint = torch.load(                                                    # Fallback for environments without weights_only support
                    self.best_model_path,
                    map_location=self.device,
                )

            if not isinstance(self.checkpoint, dict) or "model_state_dict" not in self.checkpoint:  # Validate checkpoint structure
                raise ValueError("The checkpoint does not contain a valid 'model_state_dict' dictionary.")

            self.model = BaselineReceiverConditionedSeismogramNet(                               # Rebuild trained receiver-conditioned model from saved hyperparameters
                n_time=int(self.checkpoint["n_time"]),
                coord_dim=int(self.checkpoint["coord_dim"]),
                coord_num_bands=int(self.checkpoint["coord_num_bands"]),
                coord_max_frequency=float(self.checkpoint["coord_max_frequency"]),
                model_latent_dim=int(self.checkpoint["model_latent_dim"]),
                coord_latent_dim=int(self.checkpoint["coord_latent_dim"]),
                fusion_dim=int(self.checkpoint["fusion_dim"]),
                decoder_hidden_dim=int(self.checkpoint["decoder_hidden_dim"]),
                dropout=float(self.checkpoint["dropout"]),
            ).to(self.device)

            self.model.load_state_dict(self.checkpoint["model_state_dict"])                      # Load learned weights into rebuilt model
            self.model.eval()                                                                    # Put model in evaluation mode

            print("\n" + "=" * 120)
            print("Receiver-conditioned best checkpoint loaded successfully")
            print(f"Checkpoint epoch            : {self.checkpoint.get('epoch', 'unknown')}")
            print(f"Checkpoint train loss       : {self.checkpoint.get('train_loss', 'unknown')}")
            print(f"Checkpoint validation loss  : {self.checkpoint.get('val_loss', 'unknown')}")
            print(f"Selected device             : {self.device}")
            print("=" * 120)
            print("\n")

        def _decode_string(self, value):
            if isinstance(value, bytes):                                                         # Decode byte strings returned by HDF5 when needed
                return value.decode("utf-8")
            return str(value)

        def _safe_correlation(self, y_true, y_pred):
            y_true = np.asarray(y_true, dtype=float).ravel()                                     # Convert true trace to one-dimensional float array
            y_pred = np.asarray(y_pred, dtype=float).ravel()                                     # Convert predicted trace to one-dimensional float array

            eps = 1e-12                                                                          # Small tolerance for nearly constant traces
            std_true = float(np.std(y_true))                                                     # Standard deviation of true trace
            std_pred = float(np.std(y_pred))                                                     # Standard deviation of predicted trace

            if std_true < eps and std_pred < eps:                                                # Handle both traces almost constant
                return 1.0 if np.allclose(y_true, y_pred) else 0.0

            if std_true < eps or std_pred < eps:                                                 # Handle one degenerate trace
                return 0.0

            return float(np.corrcoef(y_true, y_pred)[0, 1])                                      # Return Pearson correlation for standard case

        def _compute_trace_metrics(self, y_true, y_pred, dt):
            y_true = np.asarray(y_true, dtype=float).ravel()                                     # Convert true trace to one-dimensional float array
            y_pred = np.asarray(y_pred, dtype=float).ravel()                                     # Convert predicted trace to one-dimensional float array

            eps = 1e-12                                                                          # Small numerical tolerance
            error = y_pred - y_true                                                              # Compute prediction error vector

            mse = float(np.mean(error ** 2))                                                     # Mean squared error
            mae = float(np.mean(np.abs(error)))                                                  # Mean absolute error
            rmse = float(np.sqrt(mse))                                                           # Root mean squared error

            true_norm = float(np.linalg.norm(y_true))                                            # L2 norm of true trace
            error_norm = float(np.linalg.norm(error))                                            # L2 norm of error vector
            relative_l2_error = float(error_norm / (true_norm + eps))                           # Relative L2 error

            correlation = self._safe_correlation(y_true, y_pred)                                 # Robust Pearson correlation

            peak_idx_true = int(np.argmax(np.abs(y_true)))                                       # Peak location in true trace
            peak_idx_pred = int(np.argmax(np.abs(y_pred)))                                       # Peak location in predicted trace

            peak_arrival_time_true_s = float(peak_idx_true * dt)                                 # Peak arrival time of true trace
            peak_arrival_time_pred_s = float(peak_idx_pred * dt)                                 # Peak arrival time of predicted trace
            peak_arrival_time_abs_error_s = float(abs(peak_arrival_time_pred_s - peak_arrival_time_true_s))  # Absolute peak-arrival error

            peak_amplitude_true = float(np.max(np.abs(y_true)))                                  # Peak amplitude of true trace
            peak_amplitude_pred = float(np.max(np.abs(y_pred)))                                  # Peak amplitude of predicted trace
            peak_amplitude_abs_error = float(abs(peak_amplitude_pred - peak_amplitude_true))     # Absolute peak-amplitude error
            peak_amplitude_relative_error = float(peak_amplitude_abs_error / (peak_amplitude_true + eps))  # Relative peak-amplitude error

            total_energy_true = float(np.sum(y_true ** 2) * dt)                                  # Total energy of true trace
            total_energy_pred = float(np.sum(y_pred ** 2) * dt)                                  # Total energy of predicted trace
            total_energy_relative_error = float(abs(total_energy_pred - total_energy_true) / (total_energy_true + eps))  # Relative energy error

            fft_true = np.fft.rfft(y_true)                                                       # FFT of true trace
            fft_pred = np.fft.rfft(y_pred)                                                       # FFT of predicted trace
            freqs = np.fft.rfftfreq(len(y_true), d=dt)                                           # FFT frequency vector

            if len(freqs) > 1:                                                                   # Compute dominant frequency only when spectrum is meaningful
                dominant_idx_true = int(np.argmax(np.abs(fft_true[1:])) + 1)
                dominant_idx_pred = int(np.argmax(np.abs(fft_pred[1:])) + 1)
                dominant_frequency_true_Hz = float(freqs[dominant_idx_true])
                dominant_frequency_pred_Hz = float(freqs[dominant_idx_pred])
            else:
                dominant_frequency_true_Hz = 0.0
                dominant_frequency_pred_Hz = 0.0

            dominant_frequency_abs_error_Hz = float(abs(dominant_frequency_pred_Hz - dominant_frequency_true_Hz))  # Absolute dominant-frequency error

            return {
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "relative_l2_error": relative_l2_error,
                "correlation": correlation,
                "peak_arrival_time_true_s": peak_arrival_time_true_s,
                "peak_arrival_time_pred_s": peak_arrival_time_pred_s,
                "peak_arrival_time_abs_error_s": peak_arrival_time_abs_error_s,
                "peak_amplitude_true": peak_amplitude_true,
                "peak_amplitude_pred": peak_amplitude_pred,
                "peak_amplitude_abs_error": peak_amplitude_abs_error,
                "peak_amplitude_relative_error": peak_amplitude_relative_error,
                "total_energy_true": total_energy_true,
                "total_energy_pred": total_energy_pred,
                "total_energy_relative_error": total_energy_relative_error,
                "dominant_frequency_true_Hz": dominant_frequency_true_Hz,
                "dominant_frequency_pred_Hz": dominant_frequency_pred_Hz,
                "dominant_frequency_abs_error_Hz": dominant_frequency_abs_error_Hz,
            }

        def _build_receiver_coordinates(self, metadata, x_true_np):
            nz, nx = x_true_np.shape                                                             # Read grid size from one velocity model
            receiver_x = metadata["receiver_x"].astype(np.int32)                                 # Read receiver x indices from metadata
            receiver_z = metadata["receiver_z"].astype(np.int32)                                 # Read receiver z indices from metadata

            receiver_coords_norm = np.column_stack(                                              # Build normalized receiver coordinates in [0, 1]
                [
                    receiver_x.astype(np.float32) / max(1, nx - 1),
                    receiver_z.astype(np.float32) / max(1, nz - 1),
                ]
            ).astype(np.float32)

            return receiver_x, receiver_z, receiver_coords_norm                                  # Return raw indices and normalized coordinates

        def _predict_full_gather(self, x_true_tensor, receiver_coords_np):
            n_receivers = int(receiver_coords_np.shape[0])                                       # Read number of receiver queries
            x_batch = x_true_tensor.unsqueeze(0).repeat(n_receivers, 1, 1, 1).to(self.device)   # Repeat same velocity model for all receiver queries
            receiver_coord_batch = torch.from_numpy(receiver_coords_np).float().to(self.device)  # Convert receiver coordinates to tensor

            with torch.no_grad():                                                                # Disable gradients during inference
                y_pred = self.model(
                    x_batch,
                    receiver_coord_batch,
                )

            return y_pred.detach().cpu().numpy()                                                 # Return predicted gather with shape (n_receivers, n_time)

        def evaluate_sample(self, sample_index=0, shuffle_seed=42):
            if sample_index < 0 or sample_index >= len(self.dataset):                            # Validate split-local sample index
                raise IndexError(f"sample_index={sample_index} is out of range [0, {len(self.dataset) - 1}]")

            x_true_tensor, y_true_tensor, metadata = self.dataset[sample_index]                  # Read one full sample from selected split
            x_true_np = x_true_tensor.squeeze(0).cpu().numpy()                                   # Convert velocity model to NumPy array
            y_true_np = y_true_tensor.cpu().numpy()                                              # Convert simulated gather to NumPy array

            receiver_x, receiver_z, receiver_coords_correct = self._build_receiver_coordinates(  # Build correct receiver-coordinate array
                metadata=metadata,
                x_true_np=x_true_np,
            )

            n_receivers = int(len(receiver_x))                                                   # Number of receivers in this gather
            rng = np.random.default_rng(shuffle_seed)                                            # Create reproducible random generator
            shuffle_perm = rng.permutation(n_receivers).astype(int)                              # Build receiver permutation for coordinate shuffle
            receiver_coords_shuffled = receiver_coords_correct[shuffle_perm].copy()              # Apply permutation to coordinates only

            y_pred_correct = self._predict_full_gather(                                          # Predict gather using correct coordinates
                x_true_tensor=x_true_tensor,
                receiver_coords_np=receiver_coords_correct,
            )

            y_pred_shuffled = self._predict_full_gather(                                         # Predict gather using shuffled coordinates
                x_true_tensor=x_true_tensor,
                receiver_coords_np=receiver_coords_shuffled,
            )

            dt = float(metadata["dt"])                                                           # Read dt from metadata
            dx = float(metadata["dx"])                                                           # Read dx from metadata
            dz = float(metadata["dz"])                                                           # Read dz from metadata
            isx = int(metadata["source_x"])                                                      # Read source x index
            isz = int(metadata["source_z"])                                                      # Read source z index
            mapped_sample_id = int(metadata["sample_id"])                                        # Read mapped sample id
            model_type = self._decode_string(metadata["model_type"])                             # Decode model type label

            rows = []                                                                            # Create list to store one metrics row per receiver

            for rec in range(n_receivers):                                                       # Loop over all receivers in the gather
                metrics_correct = self._compute_trace_metrics(                                   # Compute metrics for correct coordinates
                    y_true=y_true_np[rec],
                    y_pred=y_pred_correct[rec],
                    dt=dt,
                )

                metrics_shuffled = self._compute_trace_metrics(                                  # Compute metrics for shuffled coordinates
                    y_true=y_true_np[rec],
                    y_pred=y_pred_shuffled[rec],
                    dt=dt,
                )

                donor_receiver_id = int(shuffle_perm[rec])                                       # Receiver id from which the shuffled coordinates were taken

                rows.append(
                    {
                        "split": self.split,
                        "split_sample_index": int(sample_index),
                        "sample_id": int(mapped_sample_id),
                        "receiver_id": int(rec + 1),
                        "receiver_index_0_based": int(rec),
                        "shuffled_from_receiver_id": int(donor_receiver_id + 1),
                        "receiver_x": int(receiver_x[rec]),
                        "receiver_z": int(receiver_z[rec]),
                        "dt": dt,
                        "dx": dx,
                        "dz": dz,
                        "source_x": isx,
                        "source_z": isz,
                        "model_type": model_type,
                        "mse_correct": float(metrics_correct["mse"]),
                        "mse_shuffled": float(metrics_shuffled["mse"]),
                        "mae_correct": float(metrics_correct["mae"]),
                        "mae_shuffled": float(metrics_shuffled["mae"]),
                        "rmse_correct": float(metrics_correct["rmse"]),
                        "rmse_shuffled": float(metrics_shuffled["rmse"]),
                        "relative_l2_error_correct": float(metrics_correct["relative_l2_error"]),
                        "relative_l2_error_shuffled": float(metrics_shuffled["relative_l2_error"]),
                        "correlation_correct": float(metrics_correct["correlation"]),
                        "correlation_shuffled": float(metrics_shuffled["correlation"]),
                        "peak_arrival_time_abs_error_s_correct": float(metrics_correct["peak_arrival_time_abs_error_s"]),
                        "peak_arrival_time_abs_error_s_shuffled": float(metrics_shuffled["peak_arrival_time_abs_error_s"]),
                        "peak_amplitude_relative_error_correct": float(metrics_correct["peak_amplitude_relative_error"]),
                        "peak_amplitude_relative_error_shuffled": float(metrics_shuffled["peak_amplitude_relative_error"]),
                        "total_energy_relative_error_correct": float(metrics_correct["total_energy_relative_error"]),
                        "total_energy_relative_error_shuffled": float(metrics_shuffled["total_energy_relative_error"]),
                        "dominant_frequency_abs_error_Hz_correct": float(metrics_correct["dominant_frequency_abs_error_Hz"]),
                        "dominant_frequency_abs_error_Hz_shuffled": float(metrics_shuffled["dominant_frequency_abs_error_Hz"]),
                    }
                )

            metrics_df = pd.DataFrame(rows)                                                      # Convert per-receiver rows to DataFrame

            overall_summary_df = pd.DataFrame(                                                   # Build one-row overall summary table
                [
                    {
                        "split": self.split,
                        "split_sample_index": int(sample_index),
                        "sample_id": int(mapped_sample_id),
                        "n_receivers": int(n_receivers),
                        "mse_mean_correct": float(metrics_df["mse_correct"].mean()),
                        "mse_mean_shuffled": float(metrics_df["mse_shuffled"].mean()),
                        "rmse_mean_correct": float(metrics_df["rmse_correct"].mean()),
                        "rmse_mean_shuffled": float(metrics_df["rmse_shuffled"].mean()),
                        "relative_l2_mean_correct": float(metrics_df["relative_l2_error_correct"].mean()),
                        "relative_l2_mean_shuffled": float(metrics_df["relative_l2_error_shuffled"].mean()),
                        "correlation_mean_correct": float(metrics_df["correlation_correct"].mean()),
                        "correlation_mean_shuffled": float(metrics_df["correlation_shuffled"].mean()),
                        "peak_arrival_abs_error_mean_correct": float(metrics_df["peak_arrival_time_abs_error_s_correct"].mean()),
                        "peak_arrival_abs_error_mean_shuffled": float(metrics_df["peak_arrival_time_abs_error_s_shuffled"].mean()),
                        "dominant_frequency_abs_error_mean_correct": float(metrics_df["dominant_frequency_abs_error_Hz_correct"].mean()),
                        "dominant_frequency_abs_error_mean_shuffled": float(metrics_df["dominant_frequency_abs_error_Hz_shuffled"].mean()),
                    }
                ]
            )

            result_dict = {
                "split": self.split,
                "sample_index": int(sample_index),
                "sample_id": int(mapped_sample_id),
                "x_true": x_true_np,
                "y_true": y_true_np,
                "y_pred_correct": y_pred_correct,
                "y_pred_shuffled": y_pred_shuffled,
                "metadata": metadata,
                "model_type": model_type,
                "receiver_x": receiver_x,
                "receiver_z": receiver_z,
                "receiver_coords_correct": receiver_coords_correct,
                "receiver_coords_shuffled": receiver_coords_shuffled,
                "shuffle_perm": shuffle_perm,
                "metrics_df": metrics_df,
                "overall_summary_df": overall_summary_df,
            }

            self.last_result_dict = result_dict                                                  # Store last review result dictionary
            self.last_metrics_df = metrics_df                                                    # Store last per-receiver metrics table
            self.last_overall_summary_df = overall_summary_df                                    # Store last overall summary table

            print("\n" + "=" * 120)
            print("Receiver-conditioned sample evaluated successfully")
            print(f"Split                            : {self.split}")
            print(f"Split sample index               : {sample_index}")
            print(f"Mapped sample_id                 : {mapped_sample_id}")
            print(f"Velocity model shape             : {x_true_np.shape}")
            print(f"True gather shape                : {y_true_np.shape}")
            print(f"Predicted gather shape (correct) : {y_pred_correct.shape}")
            print(f"Predicted gather shape (shuffle) : {y_pred_shuffled.shape}")
            print(f"Mean correlation correct         : {overall_summary_df['correlation_mean_correct'].iloc[0]:.6f}")
            print(f"Mean correlation shuffled        : {overall_summary_df['correlation_mean_shuffled'].iloc[0]:.6f}")
            print(f"Mean RMSE correct                : {overall_summary_df['rmse_mean_correct'].iloc[0]:.6f}")
            print(f"Mean RMSE shuffled               : {overall_summary_df['rmse_mean_shuffled'].iloc[0]:.6f}")
            print("=" * 120)
            print("\n")

            return result_dict                                                                   # Return full review result dictionary

        def plot_metric_summary(self, result_dict=None, save=True, figure_path=None):
            if result_dict is None:                                                              # Reuse last result if user does not provide one
                result_dict = self.last_result_dict

            if result_dict is None:                                                              # Stop if no result has been evaluated yet
                raise ValueError("result_dict is None. Run evaluate_sample() first or pass a valid result dictionary.")

            metrics_df = result_dict["metrics_df"]                                               # Read per-receiver metrics dataframe
            sample_index = int(result_dict["sample_index"])                                      # Read split sample index
            sample_id = int(result_dict["sample_id"])                                            # Read mapped sample id

            receiver_ids = metrics_df["receiver_id"].to_numpy(dtype=int)                         # Read 1-based receiver ids

            fig, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)           # Create summary figure
            ax1, ax2, ax3, ax4 = axes.flatten()

            ax1.plot(receiver_ids, metrics_df["rmse_correct"].to_numpy(dtype=float), lw=1.0, ls="-", color=[0, 0, 1], label="Correct coords")      # Plot RMSE for correct coordinates
            ax1.plot(receiver_ids, metrics_df["rmse_shuffled"].to_numpy(dtype=float), lw=1.0, ls="--", color=[1, 0, 0], label="Shuffled coords")   # Plot RMSE for shuffled coordinates
            ax1.set_title("RMSE by Receiver", fontsize=10, fontweight="bold", color=(0, 0, 1))
            ax1.set_xlabel("Receiver ID")
            ax1.set_ylabel("RMSE")
            ax1.grid(visible=True, axis="x")
            ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0, fontsize=8, frameon=False)

            ax2.plot(receiver_ids, metrics_df["correlation_correct"].to_numpy(dtype=float), lw=1.0, ls="-", color=[0, 0, 1], label="Correct coords")     # Plot correlation for correct coordinates
            ax2.plot(receiver_ids, metrics_df["correlation_shuffled"].to_numpy(dtype=float), lw=1.0, ls="--", color=[1, 0, 0], label="Shuffled coords")  # Plot correlation for shuffled coordinates
            ax2.set_title("Correlation by Receiver", fontsize=10, fontweight="bold", color=(0, 0, 1))
            ax2.set_xlabel("Receiver ID")
            ax2.set_ylabel("Correlation")
            ax2.grid(visible=True, axis="x")
            ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0, fontsize=8, frameon=False)

            ax3.plot(receiver_ids, metrics_df["relative_l2_error_correct"].to_numpy(dtype=float), lw=1.0, ls="-", color=[0, 0, 1], label="Correct coords")     # Plot relative L2 error for correct coordinates
            ax3.plot(receiver_ids, metrics_df["relative_l2_error_shuffled"].to_numpy(dtype=float), lw=1.0, ls="--", color=[1, 0, 0], label="Shuffled coords")  # Plot relative L2 error for shuffled coordinates
            ax3.set_title("Relative L2 Error by Receiver", fontsize=10, fontweight="bold", color=(0, 0, 1))
            ax3.set_xlabel("Receiver ID")
            ax3.set_ylabel("Relative L2 Error")
            ax3.grid(visible=True, axis="x")
            ax3.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0, fontsize=8, frameon=False)

            ax4.axis("off")                                                                     # Use last panel as text summary
            summary_df = result_dict["overall_summary_df"]                                      # Read overall summary dataframe
            summary_text = (
                f"Receiver-Conditioned Review Summary\n\n"
                f"Split = {summary_df['split'].iloc[0]}\n"
                f"Split sample index = {summary_df['split_sample_index'].iloc[0]}\n"
                f"Mapped sample_id = {summary_df['sample_id'].iloc[0]}\n"
                f"N receivers = {summary_df['n_receivers'].iloc[0]}\n\n"
                f"Mean RMSE correct = {summary_df['rmse_mean_correct'].iloc[0]:.6f}\n"
                f"Mean RMSE shuffled = {summary_df['rmse_mean_shuffled'].iloc[0]:.6f}\n"
                f"Mean Corr correct = {summary_df['correlation_mean_correct'].iloc[0]:.6f}\n"
                f"Mean Corr shuffled = {summary_df['correlation_mean_shuffled'].iloc[0]:.6f}\n"
                f"Mean Rel. L2 correct = {summary_df['relative_l2_mean_correct'].iloc[0]:.6f}\n"
                f"Mean Rel. L2 shuffled = {summary_df['relative_l2_mean_shuffled'].iloc[0]:.6f}"
            )
            ax4.text(0.03, 0.97, summary_text, ha="left", va="top", fontsize=10, fontweight="bold", color=(0, 0, 1), transform=ax4.transAxes)

            fig.suptitle(
                f"Receiver-Conditioned CNN Metric Summary | sample_index = {sample_index} | mapped_sample_id = {sample_id}",
                fontsize=12,
                fontweight="bold",
                color=(0, 0, 1),
            )

            if figure_path is None:                                                              # Define default output figure path
                figure_path = self.outputs_dir / f"2.receiver_conditioned_cnn_metric_summary_sample_{sample_index}_Mapped_sample_{sample_id}.svg"
            else:
                figure_path = Path(figure_path)

            if save:
                plt.savefig(figure_path, dpi=200, bbox_inches="tight")

                print("\n" + "=" * 120)
                print("Receiver-conditioned metric summary figure saved successfully")
                print(figure_path)
                print("=" * 120)
                print("\n")

            plt.show()
            return fig

        def plot_prediction_review(self, result_dict=None, receivers_to_plot=None, save=True, figure_path=None):
            if result_dict is None:                                                              # Reuse last result if user does not provide one
                result_dict = self.last_result_dict

            if result_dict is None:                                                              # Stop if no result has been evaluated yet
                raise ValueError("result_dict is None. Run evaluate_sample() first or pass a valid result dictionary.")

            x_true_np = result_dict["x_true"]                                                    # Read velocity model
            y_true_np = result_dict["y_true"]                                                    # Read simulated gather
            y_pred_correct = result_dict["y_pred_correct"]                                       # Read predictions with correct coordinates
            y_pred_shuffled = result_dict["y_pred_shuffled"]                                     # Read predictions with shuffled coordinates
            metrics_df = result_dict["metrics_df"]                                               # Read per-receiver metrics dataframe
            sample_index = int(result_dict["sample_index"])                                      # Read split sample index
            sample_id = int(result_dict["sample_id"])                                            # Read mapped sample id

            metadata = result_dict["metadata"]                                                   # Read metadata dictionary
            receiver_x = result_dict["receiver_x"]                                               # Read receiver x array
            receiver_z = result_dict["receiver_z"]                                               # Read receiver z array
            shuffle_perm = result_dict["shuffle_perm"]                                           # Read permutation used for shuffled coordinates

            isx = int(metadata["source_x"])                                                      # Read source x index
            isz = int(metadata["source_z"])                                                      # Read source z index
            model_type = self._decode_string(metadata["model_type"])                             # Read model type label

            n_receivers = int(y_true_np.shape[0])                                                # Read total number of receivers

            if receivers_to_plot is None:                                                        # Select default trace-review receivers if user does not provide them
                n_default = min(5, n_receivers)
                receivers_to_plot = np.linspace(0, n_receivers - 1, n_default, dtype=int).tolist()
            else:
                receivers_to_plot = [int(r) for r in receivers_to_plot]

            receivers_to_plot = receivers_to_plot[:5]                                            # Keep at most five receivers to preserve layout

            fig = plt.figure(figsize=(16, 12))
            fig.suptitle(
                f"Receiver-Conditioned CNN Trace Review | Sample = {sample_index} | Mapped sample_id = {sample_id}",
                fontsize=12,
                fontweight="bold",
                color=(0, 0, 1),
            )

            ax0 = plt.subplot(3, 2, 1)
            im = ax0.imshow(x_true_np, cmap="Spectral", aspect="auto")
            ax0.set_title(f"Input Velocity Model = {model_type}", fontsize=10, fontweight="normal", color=(0, 0, 1))
            ax0.set_xlabel("ix")
            ax0.set_ylabel("iz")
            ax0.grid(False)
            ax0.axis("off")

            cbar0 = fig.colorbar(im, ax=ax0, pad=0.01, fraction=0.03)
            cbar0.ax.tick_params(labelsize=8)
            cbar0.set_label("Velocity")

            ax0.scatter(receiver_x, receiver_z, marker="^", s=20, linewidths=0.5, color=(0, 0, 0))
            for k in range(len(receiver_x)):
                ax0.text(
                    receiver_x[k],
                    receiver_z[k] * 1.0,
                    f"ST{k+1}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    color=(0, 0, 0),
                )

            ax0.scatter([isx], [isz], marker="*", s=100, color=(0, 0, 0))
            ax0.text(
                float(isx) * 1.05,
                float(isz),
                "Source",
                ha="left",
                va="center",
                fontweight="bold",
                color=(0, 0, 0),
            )

            for i, rec in enumerate(receivers_to_plot, start=2):
                ax = plt.subplot(3, 2, i)

                donor_id = int(shuffle_perm[rec]) + 1                                            # Read donor receiver id used in the shuffled-coordinate case
                rec_metrics = metrics_df.loc[metrics_df["receiver_index_0_based"] == rec].iloc[0]

                ax.plot(y_true_np[rec], lw=1.0, ls="-", color=[0, 0, 0], label=f"Simulated - ST{rec+1}")                       # Plot simulated trace
                ax.plot(y_pred_correct[rec], lw=1.0, ls="-", color=[0, 0, 1], label=f"Predicted correct - ST{rec+1}")          # Plot prediction using correct coordinates
                ax.plot(y_pred_shuffled[rec], lw=1.0, ls="--", color=[1, 0, 0], label=f"Predicted shuffled - from ST{donor_id}")  # Plot prediction using shuffled coordinates

                ax.set_title(
                    (
                        f"ST{rec+1} | CorrC={float(rec_metrics['correlation_correct']):.2f}, "
                        f"CorrS={float(rec_metrics['correlation_shuffled']):.2f} | "
                        f"RMSEC={float(rec_metrics['rmse_correct']):.2f}, "
                        f"RMSES={float(rec_metrics['rmse_shuffled']):.2f}"
                    ),
                    fontsize=10,
                    fontweight="normal",
                    color=(0, 0, 1),
                )
                ax.set_xlabel("Time index", fontsize=8)
                ax.set_ylabel("Amplitude", fontsize=8)
                ax.tick_params(axis="x", labelsize=8)
                ax.tick_params(axis="y", labelsize=8)
                ax.grid(visible=True, axis="x")
                ax.set_xlim(0, y_true_np.shape[1])
                ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0, fontsize=8, frameon=False)

            plt.tight_layout()

            if figure_path is None:                                                              # Define default prediction-review path
                figure_path = self.outputs_dir / f"3.receiver_conditioned_cnn_prediction_review_sample_{sample_index}_Mapped_sample_{sample_id}.svg"
            else:
                figure_path = Path(figure_path)

            if save:
                plt.savefig(figure_path, dpi=200, bbox_inches="tight")

                print("\n" + "=" * 120)
                print("Receiver-conditioned prediction review figure saved successfully")
                print(figure_path)
                print("=" * 120)
                print("\n")

            plt.show()
            return fig

        def run_all(
            self,
            sample_index=0,
            shuffle_seed=42,
            receivers_to_plot_traces=None,
            save=True,
        ):
            result_dict = self.evaluate_sample(                                                  # Evaluate one sample using correct and shuffled coordinates
                sample_index=sample_index,
                shuffle_seed=shuffle_seed,
            )

            self.plot_metric_summary(                                                            # Plot metric summary
                result_dict=result_dict,
                save=save,
            )

            self.plot_prediction_review(                                                         # Plot trace review
                result_dict=result_dict,
                receivers_to_plot=receivers_to_plot_traces,
                save=save,
            )

            return result_dict, self.last_metrics_df, self.last_overall_summary_df               # Return reviewed outputs

else:

    class ReceiverConditionedShuffledCoordsReviewer:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"PyTorch is not available in this environment. Original error: {TORCH_IMPORT_ERROR}"
            )
            
            
##############################################################################################################################################
##############################################################################################################################################
################################       Baseline CNN Spectrogram Comparator (n receptors FourierOnet)       ###################################
##############################################################################################################################################
##############################################################################################################################################
if TORCH_AVAILABLE:
    class ReceiverConditionedCNNSpectrogramComparator:
        """
        Spectrogram comparator for receiver-conditioned CNN results.

        What this class does
        --------------------
        This class is designed specifically to work with the result_dict returned by:
            ReceiverConditionedShuffledCoordsReviewer.run_all()

        It does NOT run the model again.
        It only consumes already computed arrays:
        - y_true
        - y_pred_correct
        - y_pred_shuffled
        - metadata
        - receiver_x
        - receiver_z
        - shuffle_perm

        For each selected receiver, it plots:
        1. Simulated spectrogram
        2. Predicted spectrogram using correct coordinates
        3. Predicted spectrogram using shuffled coordinates
        4. Absolute dB difference: simulated vs correct
        5. Absolute dB difference: simulated vs shuffled
        """

        def __init__(
            self,
            result_dict,
            NFFT=128,
            noverlap=96,
            cmap_main="viridis",
            cmap_diff="magma",
            db_floor=-120.0,
            outputs_dir=None,
        ):
            if not isinstance(result_dict, dict):                                                # Validate result_dict type
                raise TypeError("result_dict must be a dictionary.")

            required_keys = [                                                                     # Define required keys for this comparator
                "y_true",
                "y_pred_correct",
                "y_pred_shuffled",
                "metadata",
                "shuffle_perm",
            ]

            missing_keys = [key for key in required_keys if key not in result_dict]               # Detect missing keys
            if len(missing_keys) > 0:
                raise ValueError(
                    f"result_dict is missing required keys: {missing_keys}"
                )

            self.result_dict = result_dict                                                        # Store full result dictionary
            self.NFFT = int(NFFT)                                                                 # Store spectrogram window length
            self.noverlap = int(noverlap)                                                         # Store overlap length
            self.cmap_main = cmap_main                                                            # Store colormap for simulated and predicted panels
            self.cmap_diff = cmap_diff                                                            # Store colormap for absolute-difference panels
            self.db_floor = float(db_floor)                                                       # Store minimum dB floor

            if self.NFFT <= 0:                                                                    # Validate NFFT
                raise ValueError("NFFT must be positive.")

            if self.noverlap < 0:                                                                 # Validate overlap lower bound
                raise ValueError("noverlap cannot be negative.")

            if self.noverlap >= self.NFFT:                                                        # Validate overlap upper bound
                raise ValueError("noverlap must be smaller than NFFT.")

            if outputs_dir is None:                                                               # Define output directory
                self.outputs_dir = Path.cwd()
            else:
                self.outputs_dir = Path(outputs_dir)

            self.outputs_dir.mkdir(parents=True, exist_ok=True)                                   # Create output directory if needed

            self.metadata = self.result_dict["metadata"]                                          # Store metadata dictionary
            self.y_true = np.asarray(self.result_dict["y_true"], dtype=float)                     # Store simulated gather
            self.y_pred_correct = np.asarray(self.result_dict["y_pred_correct"], dtype=float)     # Store predictions with correct coordinates
            self.y_pred_shuffled = np.asarray(self.result_dict["y_pred_shuffled"], dtype=float)   # Store predictions with shuffled coordinates

            self.receiver_x = np.asarray(                                                         # Recover receiver x positions
                self.result_dict.get("receiver_x", self.metadata["receiver_x"]),
                dtype=int,
            )

            self.receiver_z = np.asarray(                                                         # Recover receiver z positions
                self.result_dict.get("receiver_z", self.metadata["receiver_z"]),
                dtype=int,
            )

            self.shuffle_perm = np.asarray(self.result_dict["shuffle_perm"], dtype=int)           # Store receiver permutation used in shuffled prediction
            self.dt = float(self.metadata["dt"])                                                  # Read time step from metadata
            self.sample_index = int(self.result_dict.get("sample_index", -1))                     # Store split-local sample index
            self.sample_id = int(self.result_dict.get("sample_id", self.metadata.get("sample_id", -1)))  # Store mapped sample id
            self.split = str(self.result_dict.get("split", "unknown"))                            # Store split name

            self.model_type = self._decode_string(                                                # Decode model type string
                self.result_dict.get("model_type", self.metadata.get("model_type", "unknown"))
            )

            if self.y_true.ndim != 2:                                                             # Validate y_true shape
                raise ValueError(f"y_true must have shape (n_receivers, n_time), got {self.y_true.shape}")

            if self.y_pred_correct.shape != self.y_true.shape:                                    # Validate correct-prediction shape
                raise ValueError(
                    f"y_pred_correct must have shape {self.y_true.shape}, got {self.y_pred_correct.shape}"
                )

            if self.y_pred_shuffled.shape != self.y_true.shape:                                   # Validate shuffled-prediction shape
                raise ValueError(
                    f"y_pred_shuffled must have shape {self.y_true.shape}, got {self.y_pred_shuffled.shape}"
                )

            self.n_receivers = int(self.y_true.shape[0])                                          # Store number of receivers
            self.n_time = int(self.y_true.shape[1])                                               # Store number of time samples

            if len(self.receiver_x) != self.n_receivers:                                          # Validate receiver_x length
                raise ValueError("receiver_x length must match the number of receivers.")

            if len(self.receiver_z) != self.n_receivers:                                          # Validate receiver_z length
                raise ValueError("receiver_z length must match the number of receivers.")

            if len(self.shuffle_perm) != self.n_receivers:                                        # Validate shuffle permutation length
                raise ValueError("shuffle_perm length must match the number of receivers.")

        def _decode_string(self, value):
            if isinstance(value, bytes):                                                          # Decode HDF5 byte strings when needed
                return value.decode("utf-8")
            return str(value)

        def _compute_spectrogram(self, trace):
            trace = np.asarray(trace, dtype=float).ravel()                                        # Convert trace to 1D float array
            eps = 1e-20                                                                           # Small constant to avoid log of zero
            fs = 1.0 / float(self.dt)                                                             # Compute sampling frequency from dt

            Pxx, frequencies, times = mlab.specgram(                                              # Compute spectrogram with matplotlib.mlab
                x=trace,
                NFFT=self.NFFT,
                Fs=fs,
                noverlap=self.noverlap,
            )

            Pxx_dB = 10.0 * np.log10(Pxx + eps)                                                   # Convert spectrogram power to dB
            Pxx_dB = np.maximum(Pxx_dB, self.db_floor)                                            # Clip very small values to dB floor

            return Pxx, Pxx_dB, frequencies, times                                                # Return full spectrogram outputs

        def _default_receivers_to_plot(self, n_receivers_to_plot=4):
            n_receivers_to_plot = int(n_receivers_to_plot)                                        # Convert requested number of receivers to integer

            if n_receivers_to_plot <= 0:                                                          # Validate requested receiver count
                raise ValueError("n_receivers_to_plot must be positive.")

            if self.n_receivers <= n_receivers_to_plot:                                           # Use all receivers if there are only a few
                return list(range(self.n_receivers))

            receivers = np.linspace(                                                              # Select evenly spaced receivers
                0,
                self.n_receivers - 1,
                n_receivers_to_plot,
                dtype=int,
            )

            return list(np.unique(receivers))                                                     # Return unique receiver indices

        def plot_spectrogram_comparison(
            self,
            receivers_to_plot=None,
            fmax=None,
            common_color_scale=True,
            common_diff_scale=True,
            save=True,
            figure_path=None,
        ):
            if receivers_to_plot is None:                                                         # Select default receivers if user does not provide them
                receivers_to_plot = self._default_receivers_to_plot(n_receivers_to_plot=4)
            else:
                receivers_to_plot = [int(r) for r in receivers_to_plot]

            for receiver_id in receivers_to_plot:                                                 # Validate requested receiver ids
                if receiver_id < 0 or receiver_id >= self.n_receivers:
                    raise IndexError(
                        f"receiver_id={receiver_id} is out of range [0, {self.n_receivers - 1}]"
                    )

            spectrogram_results = []                                                              # Create list to store per-receiver spectrogram results

            for rec in receivers_to_plot:                                                         # Loop over selected receivers
                _, Pxx_true_dB, frequencies, times = self._compute_spectrogram(self.y_true[rec])                         # Compute simulated spectrogram
                _, Pxx_correct_dB, _, _ = self._compute_spectrogram(self.y_pred_correct[rec])                            # Compute correct-coordinate spectrogram
                _, Pxx_shuffled_dB, _, _ = self._compute_spectrogram(self.y_pred_shuffled[rec])                          # Compute shuffled-coordinate spectrogram

                Pxx_abs_diff_correct_dB = np.abs(Pxx_true_dB - Pxx_correct_dB)                                           # Compute absolute dB difference for correct coordinates
                Pxx_abs_diff_shuffled_dB = np.abs(Pxx_true_dB - Pxx_shuffled_dB)                                         # Compute absolute dB difference for shuffled coordinates

                if fmax is not None:                                                                                      # Restrict frequencies if a maximum frequency is requested
                    mask_f = frequencies <= float(fmax)

                    if not np.any(mask_f):
                        raise ValueError(f"fmax={fmax} leaves no frequencies to plot.")

                    frequencies_plot = frequencies[mask_f]
                    Pxx_true_plot = Pxx_true_dB[mask_f, :]
                    Pxx_correct_plot = Pxx_correct_dB[mask_f, :]
                    Pxx_shuffled_plot = Pxx_shuffled_dB[mask_f, :]
                    Pxx_abs_diff_correct_plot = Pxx_abs_diff_correct_dB[mask_f, :]
                    Pxx_abs_diff_shuffled_plot = Pxx_abs_diff_shuffled_dB[mask_f, :]
                else:
                    frequencies_plot = frequencies
                    Pxx_true_plot = Pxx_true_dB
                    Pxx_correct_plot = Pxx_correct_dB
                    Pxx_shuffled_plot = Pxx_shuffled_dB
                    Pxx_abs_diff_correct_plot = Pxx_abs_diff_correct_dB
                    Pxx_abs_diff_shuffled_plot = Pxx_abs_diff_shuffled_dB

                spectrogram_results.append(
                    {
                        "receiver_id": int(rec + 1),
                        "receiver_index_0_based": int(rec),
                        "receiver_x": int(self.receiver_x[rec]),
                        "receiver_z": int(self.receiver_z[rec]),
                        "shuffled_from_receiver_id": int(self.shuffle_perm[rec] + 1),
                        "frequencies": frequencies_plot,
                        "times": times,
                        "Pxx_true_dB": Pxx_true_plot,
                        "Pxx_correct_dB": Pxx_correct_plot,
                        "Pxx_shuffled_dB": Pxx_shuffled_plot,
                        "Pxx_abs_diff_correct_dB": Pxx_abs_diff_correct_plot,
                        "Pxx_abs_diff_shuffled_dB": Pxx_abs_diff_shuffled_plot,
                    }
                )

            if common_color_scale:                                                                # Build common color scale for simulated and prediction panels
                all_main_db = np.concatenate(
                    [
                        np.concatenate(
                            [
                                item["Pxx_true_dB"].ravel(),
                                item["Pxx_correct_dB"].ravel(),
                                item["Pxx_shuffled_dB"].ravel(),
                            ]
                        )
                        for item in spectrogram_results
                    ]
                )

                vmin_main = np.percentile(all_main_db, 5)
                vmax_main = np.percentile(all_main_db, 95)
            else:
                vmin_main = None
                vmax_main = None

            if common_diff_scale:                                                                 # Build common color scale for absolute-difference panels
                all_diff_db = np.concatenate(
                    [
                        np.concatenate(
                            [
                                item["Pxx_abs_diff_correct_dB"].ravel(),
                                item["Pxx_abs_diff_shuffled_dB"].ravel(),
                            ]
                        )
                        for item in spectrogram_results
                    ]
                )

                vmin_diff = np.percentile(all_diff_db, 5)
                vmax_diff = np.percentile(all_diff_db, 95)
            else:
                vmin_diff = None
                vmax_diff = None

            nrows = len(receivers_to_plot)                                                        # One row per selected receiver
            ncols = 5                                                                             # Five panels: true, correct, shuffled, diff-correct, diff-shuffled

            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(26, 4.0 * nrows),
                constrained_layout=True,
            )

            axes = np.atleast_2d(axes)                                                            # Ensure axes is always 2D

            fig.suptitle(
                (
                    f"Receiver-Conditioned CNN Spectrogram Comparison | Split = {self.split} | "
                    f"sample_index = {self.sample_index} | mapped_sample_id = {self.sample_id}"
                ),
                fontsize=12,
                fontweight="bold",
                color=(0, 0, 1),
            )

            last_im_main = None                                                                   # Placeholder for main colorbar mappable
            last_im_diff = None                                                                   # Placeholder for difference colorbar mappable

            for i, item in enumerate(spectrogram_results):                                        # Loop over selected receivers
                rec_id = item["receiver_id"]
                rec_x = item["receiver_x"]
                rec_z = item["receiver_z"]
                donor_id = item["shuffled_from_receiver_id"]
                frequencies_plot = item["frequencies"]
                times = item["times"]

                ax = axes[i, 0]                                                                   # Panel 1: simulated spectrogram
                last_im_main = ax.imshow(
                    item["Pxx_true_dB"],
                    origin="lower",
                    aspect="auto",
                    extent=[times[0], times[-1], frequencies_plot[0], frequencies_plot[-1]],
                    cmap=self.cmap_main,
                    vmin=vmin_main,
                    vmax=vmax_main,
                )
                ax.set_title(f"Simulated | ST{rec_id} | ix={rec_x}, iz={rec_z}", fontsize=10, fontweight="bold", color=(0, 0, 1))
                ax.set_xlabel("Time (s)", fontsize=8)
                ax.set_ylabel("Frequency (Hz)", fontsize=8)
                ax.tick_params(axis="x", labelsize=8)
                ax.tick_params(axis="y", labelsize=8)

                ax = axes[i, 1]                                                                   # Panel 2: prediction with correct coordinates
                ax.imshow(
                    item["Pxx_correct_dB"],
                    origin="lower",
                    aspect="auto",
                    extent=[times[0], times[-1], frequencies_plot[0], frequencies_plot[-1]],
                    cmap=self.cmap_main,
                    vmin=vmin_main,
                    vmax=vmax_main,
                )
                ax.set_title(f"Predicted correct | ST{rec_id}", fontsize=10, fontweight="bold", color=(0, 0, 1))
                ax.set_xlabel("Time (s)", fontsize=8)
                ax.set_ylabel("Frequency (Hz)", fontsize=8)
                ax.tick_params(axis="x", labelsize=8)
                ax.tick_params(axis="y", labelsize=8)

                ax = axes[i, 2]                                                                   # Panel 3: prediction with shuffled coordinates
                ax.imshow(
                    item["Pxx_shuffled_dB"],
                    origin="lower",
                    aspect="auto",
                    extent=[times[0], times[-1], frequencies_plot[0], frequencies_plot[-1]],
                    cmap=self.cmap_main,
                    vmin=vmin_main,
                    vmax=vmax_main,
                )
                ax.set_title(f"Predicted shuffled | ST{rec_id} from ST{donor_id}", fontsize=10, fontweight="bold", color=(0, 0, 1))
                ax.set_xlabel("Time (s)", fontsize=8)
                ax.set_ylabel("Frequency (Hz)", fontsize=8)
                ax.tick_params(axis="x", labelsize=8)
                ax.tick_params(axis="y", labelsize=8)

                ax = axes[i, 3]                                                                   # Panel 4: absolute dB difference for correct coordinates
                last_im_diff = ax.imshow(
                    item["Pxx_abs_diff_correct_dB"],
                    origin="lower",
                    aspect="auto",
                    extent=[times[0], times[-1], frequencies_plot[0], frequencies_plot[-1]],
                    cmap=self.cmap_diff,
                    vmin=vmin_diff,
                    vmax=vmax_diff,
                )
                ax.set_title(f"|Sim - Correct| dB | ST{rec_id}", fontsize=10, fontweight="bold", color=(0, 0, 1))
                ax.set_xlabel("Time (s)", fontsize=8)
                ax.set_ylabel("Frequency (Hz)", fontsize=8)
                ax.tick_params(axis="x", labelsize=8)
                ax.tick_params(axis="y", labelsize=8)

                ax = axes[i, 4]                                                                   # Panel 5: absolute dB difference for shuffled coordinates
                ax.imshow(
                    item["Pxx_abs_diff_shuffled_dB"],
                    origin="lower",
                    aspect="auto",
                    extent=[times[0], times[-1], frequencies_plot[0], frequencies_plot[-1]],
                    cmap=self.cmap_diff,
                    vmin=vmin_diff,
                    vmax=vmax_diff,
                )
                ax.set_title(f"|Sim - Shuffled| dB | ST{rec_id}", fontsize=10, fontweight="bold", color=(0, 0, 1))
                ax.set_xlabel("Time (s)", fontsize=8)
                ax.set_ylabel("Frequency (Hz)", fontsize=8)
                ax.tick_params(axis="x", labelsize=8)
                ax.tick_params(axis="y", labelsize=8)

            if last_im_main is not None:                                                          # Add colorbar for main panels
                cbar_main = fig.colorbar(
                    last_im_main,
                    ax=axes[:, :3].ravel().tolist(),
                    pad=0.01,
                    fraction=0.02,
                    shrink=0.92,
                )
                cbar_main.set_label("Power Spectral Density (dB)")

            if last_im_diff is not None:                                                          # Add colorbar for difference panels
                cbar_diff = fig.colorbar(
                    last_im_diff,
                    ax=axes[:, 3:].ravel().tolist(),
                    pad=0.01,
                    fraction=0.02,
                    shrink=0.92,
                )
                cbar_diff.set_label("Absolute Difference (dB)")

            if save:                                                                              # Save figure if requested
                if figure_path is None:
                    figure_path = self.outputs_dir / (
                        f"4.receiver_conditioned_cnn_spectrogram_comparison_sample_"
                        f"{self.sample_index}_Mapped_sample_{self.sample_id}.svg"
                    )
                else:
                    figure_path = Path(figure_path)

                plt.savefig(figure_path, dpi=200, bbox_inches="tight")

                print("\n" + "=" * 120)
                print("Receiver-conditioned spectrogram comparison figure saved successfully")
                print(figure_path)
                print("=" * 120)
                print("\n")

            plt.show()                                                                            # Display spectrogram figure

            return {
                "sample_index": self.sample_index,
                "sample_id": self.sample_id,
                "split": self.split,
                "receivers_to_plot": receivers_to_plot,
                "metadata": self.metadata,
                "spectrogram_results": spectrogram_results,
            }

else:

    class ReceiverConditionedCNNSpectrogramComparator:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"PyTorch is not available in this environment. Original error: {TORCH_IMPORT_ERROR}"
            )