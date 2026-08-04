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


                        .PY SCRIPT FOR CLASS FOR PAPER.
"""
#-------------------- Libraries ----------------------
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
from pathlib import Path
import h5py

##############################################################################################################################################
##############################################################################################################################################
#############################################         Read and Keys H5 file       ############################################################
##############################################################################################################################################
##############################################################################################################################################

class Keys_h5_file():
    def __init__(self, h5_path):
        self.h5_path = h5_path        
    
    @staticmethod
    def _fmt_val(v, max_len=80):
        """Format an HDF5 attribute value safely for printing."""
        if isinstance(v, bytes):
            try:
                v = v.decode("utf-8")
            except UnicodeDecodeError:
                v = repr(v)
        if isinstance(v, str):
            return v if len(v) <= max_len else v[:max_len] + "..."
        if isinstance(v, np.ndarray):
            return f"ndarray shape={v.shape} dtype={v.dtype}"
        if isinstance(v, (list, tuple)):
            return str(v)[:max_len] + ("..." if len(str(v)) > max_len else "")
        return str(v)[:max_len]

    def print_h5_structure(self):
        """
        Print groups, datasets, shapes, dtypes, and attributes stored in an HDF5 file.
        """
        h5_path = self.h5_path
        h5_path = str(h5_path)

        with h5py.File(h5_path, "r") as h5f:
            print("\n" + "=" * 80)
            print("ROOT ATTRIBUTES")
            print("=" * 80)
            if len(h5f.attrs) == 0:
                print("  (none)")
            else:
                for key, value in h5f.attrs.items():
                    print(f"  {key}: {self._fmt_val(value)}")

            print("\n" + "=" * 80)
            print("H5 STRUCTURE (tree)")
            print("=" * 80)

            def visitor(name, obj):
                indent = "  " * name.count("/")
                try:
                    if isinstance(obj, h5py.Group):
                        print(f"{indent}[G] {name}/")
                        for ak, av in obj.attrs.items():
                            print(f"{indent}    @{ak}: {self._fmt_val(av)}")

                    elif isinstance(obj, h5py.Dataset):
                        print(f"{indent}[D] {name}  shape={obj.shape}  dtype={obj.dtype}")
                        for ak, av in obj.attrs.items():
                            print(f"{indent}    @{ak}: {self._fmt_val(av)}")

                    elif isinstance(obj, h5py.SoftLink):
                        print(f"{indent}[L] {name} -> {obj.path}")

                    else:
                        print(f"{indent}[?] {name}  type={type(obj).__name__}")

                except Exception as e:
                    print(f"{indent}[!] {name}  <error: {e}>")

            h5f.visititems(visitor)

            print("\n" + "=" * 80)
            print("TOP-LEVEL KEYS")
            print("=" * 80)
            for key in h5f.keys():
                print(f"  {key}")


##############################################################################################################################################
##############################################################################################################################################
######################################################    batch loss per epoch   #############################################################
##############################################################################################################################################
##############################################################################################################################################

class PlotBatchLoss():
    def __init__(self, h5_path, save_dir=None, text_size=20, initial_gray = 0.3):
        self.h5_path = Path(h5_path)
        self.save_dir = Path(save_dir) if save_dir else self.h5_path.parent
        self.text_size = text_size
        self.initial_gray = initial_gray

    def _load_data(self):
        with h5py.File(str(self.h5_path), "r") as f:
            g = f["panel_4_train_batch_loss_per_epoch"]
            epochs = g["train_batch_epochs"][:]
            losses = g["train_batch_losses"][:]
            steps_in_epoch = g["train_batch_step_in_epoch"][:]
            best_epoch = f.attrs.get("best_epoch", None)
        return epochs, steps_in_epoch, losses, best_epoch

    def plot(self, figsize=(10, 6), dpi=150, show=True):
        text_size = self.text_size
        epochs, steps, losses, best_epoch = self._load_data()
        unique_epochs = np.unique(epochs)
        n_epochs = len(unique_epochs)
        initial_gray = self.initial_gray

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        for i, ep in enumerate(unique_epochs):
            if best_epoch is not None and ep == best_epoch:
                continue
            mask = epochs == ep
            s = steps[mask]
            l = losses[mask]
            gray_val = initial_gray * i / (n_epochs - 1) if n_epochs > 1 else 0.25
            ax.plot(s, l, color=(gray_val, gray_val, gray_val), lw=0.7)

        if best_epoch is not None:
            mask = epochs == best_epoch
            ax.plot(steps[mask], losses[mask], color=(0, 0, 1), lw=1.5, zorder=5)

        ax.set_xlabel("Batch index inside epoch", fontsize=text_size)
        ax.set_ylabel("Batch loss", fontsize=text_size)
        ax.tick_params(axis="both", labelsize=text_size - 2)
        ax.set_xlim(0, max(s))

        handles = [
            plt.Line2D([], [], color=(0.3, 0.3, 0.3), lw=0.5, label="Other epochs"),
        ]
        if best_epoch is not None:
            handles.append(
                plt.Line2D([], [], color=(0, 0, 1), lw=2.5, label=f"Best epoch: {best_epoch}")
            )
        ax.legend(handles=handles, frameon=False, fontsize=text_size - 4)
        # ax.grid(False, axis="x", alpha=0.3)

        save_path = self.save_dir / "Paper_train_batch_loss.svg"
        save_path_pdf = self.save_dir / "fig6.pdf"
        fig.savefig(save_path, bbox_inches="tight")
        fig.savefig(save_path_pdf, bbox_inches="tight", dpi=dpi)

        print("=" * 120)
        print(f"Saved: {save_path}")
        print(f"Saved: {save_path_pdf}")
        print("=" * 120)

        if show:
            plt.show()
        else:
            plt.close(fig)


##############################################################################################################################################
##############################################################################################################################################
#############################################         Plot epoch loss            ############################################################
##############################################################################################################################################
##############################################################################################################################################

class PlotEpochLoss():
    def __init__(self, h5_path, save_dir=None, text_size = 20):
        self.h5_path = Path(h5_path)
        self.save_dir = Path(save_dir) if save_dir else self.h5_path.parent
        self.text_size = text_size

    def _load_data(self):
        with h5py.File(str(self.h5_path), "r") as f:
            g = f["panel_1_epoch_loss"]
            epochs = g["epochs"][:]
            train_losses = g["train_losses"][:]
            val_losses = g["val_losses"][:]
            best_epoch = g.attrs.get("best_epoch", None)
            best_val_loss = g.attrs.get("best_val_loss", None)
        return epochs, train_losses, val_losses, best_epoch, best_val_loss

    def plot(self, figsize=(10, 6), dpi=150, show=True):
        text_size = self.text_size
        epochs, train_losses, val_losses, best_epoch, best_val_loss = self._load_data()

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.plot(epochs, train_losses,color = (0.5,0.5,0.5), ls = '-', lw = 1.0, label="Train loss")
        ax.plot(epochs, val_losses, color = (0, 0 ,0), ls = '-', lw = 1.0, label="Val loss")

        if best_epoch is not None:
            ax.axvline(best_epoch, color= (0,0,1), ls="--", linewidth=1.5)
            ax.plot(best_epoch, best_val_loss,marker = '^', markersize = 10, markerfacecolor = (0,0,1), markeredgecolor = (0,0,1))
            ax.annotate(f"Best epoch: {best_epoch}\nVal loss: {best_val_loss:.3f}",
                        xy=(best_epoch, best_val_loss),
                        xytext=(10, 50), textcoords="offset points",
                        fontsize= text_size - 5, color= (0,0,0),
                        fontweight = 'bold',
                        bbox=dict(boxstyle="round,pad=0.3", fc= (1,1,1), alpha=0.5))

        ax.set_xlabel("Epoch", fontsize=text_size)
        ax.set_ylabel("Loss", fontsize=text_size)
        ax.tick_params(axis="both", labelsize=text_size - 2)
        ax.set_xlim(0, max(epochs))
        
        ax.legend(frameon=False, fontsize = text_size - 4)
        ax.grid(True, axis="x", alpha=0.3)

        save_path = self.save_dir / "Paper_epoch_loss.svg"
        save_path_pdf = self.save_dir / "fig7.pdf"
        fig.savefig(save_path, bbox_inches="tight")
        fig.savefig(save_path_pdf, bbox_inches="tight", dpi = dpi)
        
        print("=" * 120)
        print(f"Saved: {save_path}")
        print(f"Saved: {save_path_pdf}")
        print("=" * 120)

        if show:
            plt.show()
        else:
            plt.close(fig)

##############################################################################################################################################
##############################################################################################################################################
#############################################         Plot CFL of simultaions            #####################################################
##############################################################################################################################################
##############################################################################################################################################

class CFLsimul_plot():
    def __init__(self, h5_path, save_dir=None, text_size = 20):
        self.h5_path = Path(h5_path)
        self.save_dir = Path(save_dir) if save_dir else self.h5_path.parent
        self.text_size = text_size

    def _load_dataCFL(self):
        with h5py.File(str(self.h5_path), "r") as f:
            if "CFL" in f:
                cfl = f["CFL"][()]
            elif "metadata" in f and "CFL" in f["metadata"]:
                cfl = f["metadata/CFL"][()]
            else:
                raise KeyError("CFL dataset not found. Expected 'CFL' or 'metadata/CFL'.")
        return cfl

    def plotCFL(self, figsize=(10, 6), dpi=150, show=True):
        text_size = self.text_size
        CFL = self._load_dataCFL()

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.plot(CFL,color = (0.5,0.5,0.5), ls = '-', lw = 1.0, label="CFL")

        ax.set_xlabel("Simulation", fontsize=text_size)
        ax.set_ylabel("CFL", fontsize=text_size)
        ax.tick_params(axis="both", labelsize=text_size - 2)
        ax.set_xlim(0, len(CFL) - 1)
        
        # ax.legend(frameon=False, fontsize = text_size - 4)
        # ax.grid(True, axis="x", alpha=0.3)

        save_path = self.save_dir / "Paper_CFL.svg"
        save_path_pdf = self.save_dir / "Paper_CFL.pdf"
        fig.savefig(save_path, bbox_inches="tight")
        fig.savefig(save_path_pdf, bbox_inches="tight", dpi = dpi)
        
        print("=" * 120)
        print(f"Saved: {save_path}")
        print(f"Saved: {save_path_pdf}")
        print("=" * 120)

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plotCFLdistribution(self, bins=20, cfl_limit=0.30, show_kde=True, show_gaussian=False, figsize=(10, 6), dpi=150, show=True):
        text_size = self.text_size
        CFL = self._load_dataCFL()
        CFL = np.asarray(CFL, dtype=float)
        CFL = CFL[np.isfinite(CFL)]

        cfl_mean = np.mean(CFL)
        cfl_std = np.std(CFL)
        cfl_min = np.min(CFL)
        cfl_max = np.max(CFL)
        n_simul = len(CFL)

        x_range = cfl_max - cfl_min
        x_pad = 0.05 * x_range if x_range > 0 else 0.01
        x = np.linspace(cfl_min - x_pad, cfl_max + x_pad, 400)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.hist(CFL, bins=bins, density=True, color=(0.75,0.75,0.75), edgecolor=(0,0,0), linewidth=0.7, alpha=0.85, label="CFL histogram")

        if show_kde:
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(CFL)
                ax.plot(x, kde(x), color=(0,0,1), ls="-", lw=2.0, label="KDE density")
            except Exception:
                print("KDE density was not plotted because scipy gaussian_kde is not available.")

        ax.axvline(cfl_mean, color=(0,0,1), ls=":", lw=1.5)
        ax.axvline(cfl_limit, color=(0,0,0), ls=":", lw=1.5)

        ax.set_xlabel("CFL", fontsize=text_size)
        ax.set_ylabel("Probability density", fontsize=text_size)
        ax.tick_params(axis="both", labelsize=text_size - 2)
        ax.set_xlim(cfl_min - x_pad, cfl_max + x_pad)

        textstr = (
            f"N = {n_simul}\n"
            f"Mean = {cfl_mean:.3f}\n"
            f"Std = {cfl_std:.3f}\n"
            f"Min = {cfl_min:.3f}\n"
            f"Max = {cfl_max:.3f}"
        )
        ax.text(0.47, 0.95, textstr,
                transform=ax.transAxes,
                fontsize=text_size - 5,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=dict(boxstyle="round,pad=0.3", fc=(1,1,1), alpha=0.5))

        ax.legend(frameon=False, fontsize=text_size - 4)

        save_path = self.save_dir / "Paper_CFL_distribution.svg"
        save_path_pdf = self.save_dir / "Paper_CFL_distribution.pdf"
        fig.savefig(save_path, bbox_inches="tight")
        fig.savefig(save_path_pdf, bbox_inches="tight", dpi = dpi)

        print("=" * 120)
        print(f"Saved: {save_path}")
        print(f"Saved: {save_path_pdf}")
        print("=" * 120)

        if show:
            plt.show()
        else:
            plt.close(fig)
