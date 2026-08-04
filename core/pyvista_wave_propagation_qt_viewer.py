"""
Qt-based PyVista viewer for stored wavefield snapshots.

This module uses pyvistaqt to embed the VTK/PyVista renderer inside a Qt
application. The side panel, buttons, slider, playback controls, and Matplotlib
plot are Qt widgets, which is closer to specialized seismic viewers than a
plain pv.Plotter window.
"""

import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyvista as pv
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pyvistaqt import QtInteractor
from qtpy import QtCore, QtGui, QtWidgets


_VISUAL_FLOAT_DTYPE = np.float32
_BLENDER_VIEWPORT_BACKGROUND = "#2B2B2D"
_BLENDER_PLOT_BACKGROUND = "#28282B"
_BLENDER_PLOT_TEXT = "#D8D8DA"


class _QtColorLegend(QtWidgets.QWidget):
    """Flat Qt color legend that avoids VTK 3D text rendering."""

    def __init__(self, cmap, clim, title="Amplitude", parent=None):
        super().__init__(parent)
        self._cmap = str(cmap)
        self._clim = tuple(float(value) for value in clim)
        self._title = str(title)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setMinimumHeight(54)

    def set_scale(self, cmap, clim):
        self._cmap = str(cmap)
        self._clim = tuple(float(value) for value in clim)
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        bounds = self.rect().adjusted(0, 0, -1, -1)
        painter.fillRect(bounds, QtGui.QColor(32, 32, 34, 232))
        painter.setPen(QtGui.QPen(QtGui.QColor("#5A5A5E"), 1))
        painter.drawRect(bounds)

        title_font = QtGui.QFont("Segoe UI", 9)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QtGui.QColor("#E8E8EA"))
        painter.drawText(8, 14, self._title)

        gradient_rect = QtCore.QRectF(8, 21, max(self.width() - 16, 20), 12)
        gradient = QtGui.QLinearGradient(gradient_rect.topLeft(), gradient_rect.topRight())
        try:
            cmap = colormaps.get_cmap(self._cmap)
        except (KeyError, ValueError):
            cmap = colormaps.get_cmap("viridis")
        for step in range(33):
            red, green, blue, _ = cmap(step / 32.0)
            gradient.setColorAt(step / 32.0, QtGui.QColor.fromRgbF(red, green, blue, 1.0))
        painter.fillRect(gradient_rect, gradient)
        painter.setPen(QtGui.QPen(QtGui.QColor("#BEBEC2"), 1))
        painter.drawRect(gradient_rect)

        tick_font = QtGui.QFont("Segoe UI", 8)
        painter.setFont(tick_font)
        low, high = self._clim
        for tick_id in range(5):
            fraction = tick_id / 4.0
            x_pos = gradient_rect.left() + gradient_rect.width() * fraction
            painter.drawLine(QtCore.QPointF(x_pos, gradient_rect.bottom()), QtCore.QPointF(x_pos, gradient_rect.bottom() + 4))
            value = low + (high - low) * fraction
            label = f"{value:.3g}"
            label_rect = QtCore.QRectF(x_pos - 34, 37, 68, 13)
            alignment = QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop
            painter.drawText(label_rect, alignment, label)

def _apply_blender_plot_theme(axis):
    """Apply the UI plot palette without changing plotted values or limits."""
    axis.set_facecolor(_BLENDER_PLOT_BACKGROUND)
    axis.tick_params(axis="both", colors=_BLENDER_PLOT_TEXT)
    axis.xaxis.label.set_color(_BLENDER_PLOT_TEXT)
    axis.yaxis.label.set_color(_BLENDER_PLOT_TEXT)
    for spine in axis.spines.values():
        spine.set_color("#66666A")


# ==================================================================
# INTERNAL HELPER CLASSES FOR OPTIMIZATION AND NEW FEATURES
# ==================================================================

class _MeshCache:
    """LRU cache for wavefield mesh objects to avoid recomputation."""

    def __init__(self, max_size=50):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, mesh):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = mesh
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def clear(self):
        self.cache.clear()


def _mesh_cache_key(
    snapshot_id,
    vertical_scale,
    normalize,
    deform_surface,
    normalize_colors,
    lod_factor,
):
    """Build a cache key from every option that changes mesh geometry or scalars."""
    return (
        int(snapshot_id),
        round(float(vertical_scale), 6),
        bool(normalize),
        bool(deform_surface),
        bool(normalize_colors),
        int(lod_factor),
    )


class PyVistaWavePropagationQtViewer:
    """
    Rich Qt application for interactively inspecting PyVista wavefield snapshots.

    The class accepts the `snapshots` dictionary produced by the simulation and
    delegates the user interface to Qt: camera buttons, frame slider,
    play/pause, deformation toggle, scalar controls, and seismogram plots.
    """

    def __init__(self, snapshots, seismogram_data=None, scene_label=None):
        self.snapshots = snapshots

        self.wavefield_snapshots = np.asarray(
            self.snapshots["wavefield_snapshots"],
            dtype=_VISUAL_FLOAT_DTYPE,
        )
        self.snapshot_time_indices = np.asarray(
            self.snapshots["snapshot_time_indices"],
            dtype=int,
        )
        self.snapshot_times = np.asarray(
            self.snapshots["snapshot_times"],
            dtype=float,
        )
        self.velocity_model = np.asarray(
            self.snapshots["velocity_model"],
            dtype=_VISUAL_FLOAT_DTYPE,
        )

        self.dx = float(self.snapshots["dx"])
        self.dz = float(self.snapshots["dz"])
        self.dt = float(self.snapshots["dt"])
        self.nx = int(self.snapshots["nx"])
        self.nz = int(self.snapshots["nz"])
        self.nt = int(self.snapshots["nt"])
        self.source_x = int(self.snapshots["source_x"])
        self.source_z = int(self.snapshots["source_z"])
        #--------------------------------------------------------
        self.source_x_all = np.asarray(self.snapshots.get("source_x_all", [self.source_x]), dtype=int)  # Read all rupture source x indices or fall back to the legacy single source
        self.source_z_all = np.asarray(self.snapshots.get("source_z_all", [self.source_z]), dtype=int)  # Read all rupture source z indices or fall back to the legacy single source
        self.n_source_events = int(self.snapshots.get("n_source_events", len(self.source_x_all)))       # Read the source-event count when available
        #--------------------------------------------------------
        self.receiver_x = np.asarray(self.snapshots["receiver_x"], dtype=int)
        self.receiver_z = np.asarray(self.snapshots["receiver_z"], dtype=int)
        self.model_type = str(self.snapshots.get("model_type", "unknown"))

        self._validate_snapshots()
        self._build_coordinate_grids()

        self.seismogram_data = self._prepare_seismogram_data(seismogram_data)
        self.scene_label = "" if scene_label is None else str(scene_label)

        self._mesh_cache = _MeshCache(max_size=50)

    # ------------------------------------------------------------------
    # Snapshot setup and mesh helpers
    # ------------------------------------------------------------------
    def _validate_snapshots(self):
        """
        Validate the minimum snapshot structure required for the Qt viewer.
        """
        if self.wavefield_snapshots.ndim != 3:
            raise ValueError(
                "'wavefield_snapshots' must have shape (n_snapshots, nz, nx). "
                f"Got {self.wavefield_snapshots.shape}."
            )

        n_snapshots, nz_snap, nx_snap = self.wavefield_snapshots.shape

        if nz_snap != self.nz or nx_snap != self.nx:
            raise ValueError(
                "Wavefield snapshot shape is inconsistent with nx and nz. "
                f"Expected ({self.nz}, {self.nx}), got ({nz_snap}, {nx_snap})."
            )

        if self.velocity_model.shape != (self.nz, self.nx):
            raise ValueError(
                "'velocity_model' shape is inconsistent with nx and nz. "
                f"Expected ({self.nz}, {self.nx}), got {self.velocity_model.shape}."
            )

        if len(self.snapshot_time_indices) != n_snapshots:
            raise ValueError(
                "'snapshot_time_indices' length must match the number of snapshots."
            )

        if len(self.snapshot_times) != n_snapshots:
            raise ValueError(
                "'snapshot_times' length must match the number of snapshots."
            )

        if len(self.receiver_x) != len(self.receiver_z):
            raise ValueError("'receiver_x' and 'receiver_z' must have the same length.")
        #--------------------------------------------------------
        if len(self.source_x_all) != len(self.source_z_all):                                      # Validate that all rupture source arrays have matching length
            raise ValueError("'source_x_all' and 'source_z_all' must have the same length.")      # Stop if source x and z arrays are inconsistent
        if self.n_source_events != len(self.source_x_all):                                       # Validate that the stored source-event count matches the source arrays
            raise ValueError("'n_source_events' must match the length of 'source_x_all'.")        # Stop if the source-event metadata is inconsistent
        #--------------------------------------------------------

    def _build_coordinate_grids(self):
        """
        Build physical coordinate grids for the 3D surface.
        """
        x = np.arange(self.nx, dtype=_VISUAL_FLOAT_DTYPE) * self.dx
        z = np.arange(self.nz, dtype=_VISUAL_FLOAT_DTYPE) * self.dz
        self.X, self.Z = np.meshgrid(x, z)

    def inspect_snapshots(self):
        """
        Print the available keys and basic information about each snapshot entry.
        """
        print("\nAvailable keys inside snapshots:")
        print("-" * 80)

        for key, value in self.snapshots.items():
            if hasattr(value, "shape"):
                print(f"{key:30s} -> shape = {value.shape}")
            else:
                print(f"{key:30s} -> type  = {type(value)}")

    def print_amplitude_summary(self):
        """
        Print a compact amplitude summary of the stored wavefield snapshots.
        """
        amps = np.max(np.abs(self.wavefield_snapshots), axis=(1, 2))

        print("\nWavefield snapshot amplitude summary")
        print("-" * 80)
        print(f"Number of snapshots             : {len(amps)}")
        print(f"First amplitudes                : {amps[:10]}")
        print(f"Maximum amplitude               : {amps.max():.6e}")
        print(f"Snapshot with maximum amplitude : {int(amps.argmax())}")
        print("-" * 80)

        return amps

    def _scale_wavefield_for_geometry(self, field, vertical_scale=120.0, normalize=True):
        """
        Convert pressure amplitude into vertical elevation.
        """
        field = np.asarray(field, dtype=_VISUAL_FLOAT_DTYPE)

        if normalize:
            max_abs = float(np.max(np.abs(field)))
            if max_abs < 1e-20:
                return np.zeros_like(field, dtype=_VISUAL_FLOAT_DTYPE)
            return np.asarray(vertical_scale * field / max_abs, dtype=_VISUAL_FLOAT_DTYPE)

        return np.asarray(vertical_scale * field, dtype=_VISUAL_FLOAT_DTYPE)

    def _scale_wavefield_for_colors(self, field, normalize_colors=True):
        """
        Convert pressure amplitude into scalar values for coloring.
        """
        field = np.asarray(field, dtype=_VISUAL_FLOAT_DTYPE)

        if normalize_colors:
            max_abs = float(np.max(np.abs(field)))
            if max_abs < 1e-20:
                return np.zeros_like(field, dtype=_VISUAL_FLOAT_DTYPE)
            return np.asarray(field / max_abs, dtype=_VISUAL_FLOAT_DTYPE)

        return field

    def _build_surface_mesh(
        self,
        field,
        vertical_scale=120.0,
        normalize=True,
        deform_surface=False,
        normalize_colors=True,
        lod_factor=1,
    ):
        """
        Build a PyVista StructuredGrid from one wavefield snapshot.

        Parameters
        ----------
        lod_factor : int
            Level-of-Detail factor. 1 = full resolution, 2 = 50% vertices, 4 = 25% vertices.
        """
        field = np.asarray(field, dtype=_VISUAL_FLOAT_DTYPE)

        if lod_factor > 1:
            field = field[::lod_factor, ::lod_factor]
            x_grid = self.X[::lod_factor, ::lod_factor]
            z_grid = self.Z[::lod_factor, ::lod_factor]
            velocity_model = self.velocity_model[::lod_factor, ::lod_factor]
        else:
            x_grid = self.X
            z_grid = self.Z
            velocity_model = self.velocity_model

        if deform_surface:
            y_surface = self._scale_wavefield_for_geometry(
                field=field,
                vertical_scale=vertical_scale,
                normalize=normalize,
            )
        else:
            y_surface = np.zeros_like(field, dtype=_VISUAL_FLOAT_DTYPE)

        color_field = self._scale_wavefield_for_colors(
            field=field,
            normalize_colors=normalize_colors,
        )

        surface = pv.StructuredGrid(x_grid, y_surface, z_grid)
        surface["Amplitude"] = np.asarray(color_field, dtype=_VISUAL_FLOAT_DTYPE).ravel(order="F")
        surface["RawAmplitude"] = np.asarray(field, dtype=_VISUAL_FLOAT_DTYPE).ravel(order="F")
        surface["Velocity"] = np.asarray(velocity_model, dtype=_VISUAL_FLOAT_DTYPE).ravel(order="F")

        return surface

    def _add_source_and_receivers(
        self,
        plotter,
        marker_elevation=0.0,
        point_size=12,
        show_labels=False,
    ):
        """
        Add source and receiver locations to the PyVista scene.
        """
        #--------------------------------------------------------
        source_points = np.column_stack(                                                         # Build one PyVista point for each rupture source event
            [
                self.source_x_all.astype(_VISUAL_FLOAT_DTYPE) * self.dx,                         # Convert all source x indices to physical coordinates
                np.full(len(self.source_x_all), marker_elevation, dtype=_VISUAL_FLOAT_DTYPE),    # Keep all source markers on the same display elevation
                self.source_z_all.astype(_VISUAL_FLOAT_DTYPE) * self.dz,                         # Convert all source z indices to physical coordinates
            ]
        )
        #--------------------------------------------------------

        receiver_points = np.column_stack(
            [
                self.receiver_x.astype(_VISUAL_FLOAT_DTYPE) * self.dx,
                np.full(len(self.receiver_x), marker_elevation, dtype=_VISUAL_FLOAT_DTYPE),
                self.receiver_z.astype(_VISUAL_FLOAT_DTYPE) * self.dz,
            ]
        )

        plotter.add_points(
            source_points,                                                                       # Plot all rupture source markers or the legacy single source marker
            color="black",
            point_size=point_size * 1.5,
            render_points_as_spheres=True,
        )
        plotter.add_points(
            receiver_points,
            color="blue",
            point_size=point_size,
            render_points_as_spheres=True,
        )

        if show_labels:
            #--------------------------------------------------------
            source_labels = ["Source"] if len(source_points) == 1 else [f"R{k + 1}" for k in range(len(source_points))]  # Keep the old label for one source and rupture labels for many
            plotter.add_point_labels(
                source_points,                                                                   # Label all rupture source markers or the legacy single source marker
                source_labels,                                                                   # Use labels that match the number of source markers
                font_size=14,
                text_color="black",
                point_color="black",
                point_size=0,
                always_visible=True,
            )
            #--------------------------------------------------------

            receiver_labels = [f"ST{k + 1}" for k in range(len(receiver_points))]
            plotter.add_point_labels(
                receiver_points,
                receiver_labels,
                font_size=8,
                text_color="blue",
                point_color="blue",
                point_size=0,
                always_visible=False,
            )

    def _prepare_seismogram_data(self, seismogram_data):
        """
        Validate optional surface seismogram data used by the Qt side plot.
        """
        if seismogram_data is None:
            return None

        seismograms = np.asarray(seismogram_data["surface_seismograms"], dtype=_VISUAL_FLOAT_DTYPE)
        if seismograms.ndim != 2:
            raise ValueError(
                "'surface_seismograms' must have shape (n_receivers, nt), "
                f"got {seismograms.shape}."
            )

        n_receivers, nt = seismograms.shape
        dt = float(seismogram_data.get("dt", self.dt))
        time_axis = np.arange(nt, dtype=_VISUAL_FLOAT_DTYPE) * dt

        receiver_x = np.asarray(
            seismogram_data.get("receiver_x", np.arange(n_receivers)),
            dtype=int,
        )
        receiver_z = np.asarray(
            seismogram_data.get("receiver_z", np.zeros(n_receivers, dtype=int)),
            dtype=int,
        )

        selected_receivers = seismogram_data.get("selected_receivers")
        if selected_receivers is None:
            selected_receivers = [0, n_receivers // 2, n_receivers - 1]

        selected_receivers = np.asarray(selected_receivers, dtype=int)
        selected_receivers = selected_receivers[
            (selected_receivers >= 0) & (selected_receivers < n_receivers)
        ]
        selected_receivers = np.unique(selected_receivers)

        if len(selected_receivers) == 0:
            raise ValueError("No valid receiver indices were selected for plotting.")

        if len(selected_receivers) > 3:
            selected_receivers = selected_receivers[:3]

        return {
            "surface_seismograms": seismograms,
            "dt": dt,
            "time_axis": time_axis,
            "receiver_x": receiver_x,
            "receiver_z": receiver_z,
            "selected_receivers": selected_receivers,
        }

    def show_qt(
        self,
        start_snapshot=0,
        end_snapshot=None,
        step=1,
        vertical_scale=120.0,
        normalize=True,
        deform_surface=False,
        normalize_colors=True,
        cmap="turbo",
        clim=None,
        show_source_receivers=True,
        show_labels=False,
        show_edges=False,
        point_size=12,
        window_size=(1600, 950),
        frame_rate=10,
        mesh_update_mode="replace",
        orthographic=True,
        app_title="AI Surface Seismogram - PyVista Qt Viewer",
        comparison_viewers=None,
        presentation_sample_keys=None,
        presentation_sample_loader=None,
        snapshot_export_dir=None,
        presentation_lighting_enabled=True,
        presentation_shadows_enabled=True,
        presentation_grid_visible=True,
        presentation_mesh_visible=True,
        presentation_layer_outlines_visible=True,
        presentation_domain_thickness=None,
        lod_factor=1,
    ):
        """
        Open the Qt viewer and start the Qt event loop when needed.
        """
        app = QtWidgets.QApplication.instance()
        owns_app = app is None

        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        try:
            app.setStyle("Fusion")
        except Exception:
            pass

        window = _WavePropagationQtWindow(
            viewer=self,
            start_snapshot=start_snapshot,
            end_snapshot=end_snapshot,
            step=step,
            vertical_scale=vertical_scale,
            normalize=normalize,
            deform_surface=deform_surface,
            normalize_colors=normalize_colors,
            cmap=cmap,
            clim=clim,
            show_source_receivers=show_source_receivers,
            show_labels=show_labels,
            show_edges=show_edges,
            point_size=point_size,
            frame_rate=frame_rate,
            mesh_update_mode=mesh_update_mode,
            orthographic=orthographic,
            title=app_title,
            comparison_viewers=comparison_viewers,
            presentation_sample_keys=presentation_sample_keys,
            presentation_sample_loader=presentation_sample_loader,
            snapshot_export_dir=snapshot_export_dir,
            presentation_lighting_enabled=presentation_lighting_enabled,
            presentation_shadows_enabled=presentation_shadows_enabled,
            presentation_grid_visible=presentation_grid_visible,
            presentation_mesh_visible=presentation_mesh_visible,
            presentation_layer_outlines_visible=presentation_layer_outlines_visible,
            presentation_domain_thickness=presentation_domain_thickness,
            lod_factor=lod_factor,
        )

        window.resize(int(window_size[0]), int(window_size[1]))
        window.show()

        if owns_app:
            app.exec()

        return window


#--------------------------------------------------------
class _MultiSimulationComparisonWidget(QtWidgets.QWidget):                                         # Widget that compares up to two full PyVista simulations in one tab
    """
    Qt widget for synchronized two-simulation PyVista playback with two traces per simulation.
    """

    def __init__(
        self,
        comparison_viewers,
        start_snapshot,
        end_snapshot,
        step,
        vertical_scale,
        normalize,
        deform_surface,
        normalize_colors,
        cmap,
        clim,
        show_source_receivers,
        show_labels,
        show_edges,
        point_size,
        frame_rate,
        mesh_update_mode,
        orthographic,
        lod_factor=1,
        grid_visible=True,
        mesh_visible=True,
        lighting_enabled=True,
        shadows_enabled=True,
        layer_outlines_visible=True,
        domain_thickness=None,
        parent=None,
    ):
        super().__init__(parent)

        self.viewers = list(comparison_viewers)[:2]                                                # Keep only the first two simulations for lighter synchronized comparison
        if len(self.viewers) == 0:                                                                 # Stop early if the caller did not provide any simulation to compare
            raise ValueError("comparison_viewers must contain at least one viewer.")                # Keep the comparison tab explicit and predictable
        self.viewer_labels = [viewer.scene_label if viewer.scene_label else f"Simulation {viewer_id + 1}" for viewer_id, viewer in enumerate(self.viewers)]  # Use the real selected simulation labels in the comparison tab

        self.vertical_scale = float(vertical_scale)                                                # Store the shared vertical scaling used by every PyVista view
        self.normalize = bool(normalize)                                                           # Store the shared geometry normalization flag
        self.deform_surface = bool(deform_surface)                                                 # Store the shared surface deformation flag
        self.normalize_colors = bool(normalize_colors)                                             # Store the shared color normalization flag
        self.cmap = str(cmap)                                                                      # Store the shared colormap name
        self.base_clim = clim                                                                      # Store the color limits received from the single-view tab
        self.show_source_receivers = bool(show_source_receivers)                                   # Store whether source and receiver markers should be shown
        self.show_labels = bool(show_labels)                                                       # Store whether marker labels should be shown in the overview
        self.show_edges = bool(mesh_visible)                                                       # Store whether mesh edges should be shown in each overview scene
        self.grid_visible = bool(grid_visible)                                                     # Store whether bounds/grid should be shown in comparison scenes
        self.lighting_enabled = bool(lighting_enabled)                                             # Store whether presentation-style lighting is active
        self.shadows_enabled = bool(shadows_enabled)                                               # Store whether VTK shadows should be requested
        self.layer_outlines_visible = bool(layer_outlines_visible)                                 # Store whether velocity-layer outlines are visible in comparison scenes
        self.domain_thickness = None if domain_thickness is None else float(domain_thickness)      # Optional fixed presentation block thickness
        if self.domain_thickness is not None and self.domain_thickness <= 0.0:                     # Validate user-provided thickness before rendering
            raise ValueError("domain_thickness must be positive or None.")                         # Keep invalid geometry from reaching PyVista
        self.point_size = int(point_size)                                                          # Store the marker size used in the smaller comparison views
        self.frame_rate = max(float(frame_rate), 1.0)                                              # Store the synchronized playback frame rate
        self.mesh_update_mode = str(mesh_update_mode).lower()                                      # Store the mesh update strategy used by every comparison view
        self.orthographic = bool(orthographic)                                                     # Store the shared camera projection mode
        self.lod_factor = max(int(lod_factor), 1)                                                   # Store the shared render LOD used in each comparison view
        self.current_frame = 0                                                                     # Store the current synchronized comparison frame

        if self.mesh_update_mode not in ("replace", "inplace"):                                    # Validate the mesh update mode before building any PyVista widget
            raise ValueError("mesh_update_mode must be 'replace' or 'inplace'.")                    # Match the accepted modes used by the single-simulation viewer

        self.snapshot_ids_by_viewer = [                                                            # Build a compatible snapshot index list for each simulation
            self._select_snapshot_ids_for_viewer(                                                   # Select the requested snapshot range for one simulation
                viewer=viewer,                                                                      # Use the current comparison viewer object
                start_snapshot=start_snapshot,                                                      # Use the shared comparison start snapshot
                end_snapshot=end_snapshot,                                                          # Use the shared comparison end snapshot
                step=step,                                                                          # Use the shared comparison snapshot stride
            )
            for viewer in self.viewers                                                              # Repeat the snapshot selection for every simulation in the grid
        ]
        self.max_frames = min(len(snapshot_ids) for snapshot_ids in self.snapshot_ids_by_viewer)    # Keep playback synchronized by using the shortest available frame count
        if self.max_frames <= 0:                                                                   # Validate that at least one synchronized frame exists
            raise ValueError("No synchronized comparison frames are available.")                     # Stop if the comparison range is empty

        self.clims = [self._resolve_color_limits_for_viewer(viewer, ids) for viewer, ids in zip(self.viewers, self.snapshot_ids_by_viewer)]  # Resolve color limits for every simulation
        self.plotters = []                                                                         # Store every QtInteractor used in the comparison grid
        self.surfaces = []                                                                         # Store the current PyVista surface for each simulation
        self.mesh_actors = []                                                                      # Store the current wavefield actor for each simulation
        self.side_surfaces_by_viewer = []                                                          # Store the thickened-domain side surfaces for each comparison scene
        self.side_actors_by_viewer = []                                                            # Store side actors for every comparison scene
        self.layer_outline_polydata_by_viewer = []                                                  # Store layer-outline geometry for every comparison scene
        self.layer_outline_actors_by_viewer = []                                                    # Store layer-outline actors for every comparison scene
        self.trace_time_lines = []                                                                 # Store the vertical time cursors for all trace axes
        self.trace_axes = []                                                                       # Store trace axes for later layout and updates
        self.velocity_overlay_cmap = "viridis"                                                     # Use viridis as the default colormap for the 2D velocity-model overlays
        self.velocity_overlay_canvases = []                                                        # Store the velocity-model overlay canvases placed over each PyVista view
        self.scalar_legends = []                                                                    # Flat Qt legends replacing VTK scalar-bar text

        self._build_ui()                                                                           # Build the comparison tab user interface
        self._position_scalar_legends()
        self._build_scenes()                                                                       # Build the initial PyVista scene in each populated cell
        self._connect_events()                                                                     # Connect the compact shared controls
        self._update_frame(0)                                                                      # Draw the first synchronized frame

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_scalar_legends()

    def _position_scalar_legends(self):
        for legend, plotter in zip(self.scalar_legends, self.plotters):
            parent = plotter.interactor
            width = min(max(int(parent.width() * 0.62), 180), 420)
            legend.setGeometry(
                max((parent.width() - width) // 2, 10),
                max(parent.height() - 60, 8),
                width,
                54,
            )
            legend.raise_()
    def _select_snapshot_ids_for_viewer(self, viewer, start_snapshot, end_snapshot, step):          # Select valid snapshot ids for one comparison viewer
        n_snapshots = int(viewer.wavefield_snapshots.shape[0])                                      # Read the number of snapshots stored by this simulation
        start_snapshot = int(np.clip(int(start_snapshot), 0, n_snapshots - 1))                      # Clamp the start snapshot to this simulation range
        end_snapshot = n_snapshots - 1 if end_snapshot is None else int(end_snapshot)               # Use the last snapshot when no explicit end is provided
        end_snapshot = int(np.clip(end_snapshot, start_snapshot, n_snapshots - 1))                  # Clamp the end snapshot to this simulation range
        step = max(int(step), 1)                                                                    # Keep the snapshot stride positive
        return np.arange(start_snapshot, end_snapshot + 1, step, dtype=int)                         # Return the valid snapshot ids for this viewer

    def _resolve_color_limits_for_viewer(self, viewer, snapshot_ids):                               # Resolve color limits for one comparison viewer
        if self.base_clim is not None:                                                             # Reuse explicit color limits when the single viewer already resolved them
            return self.base_clim                                                                  # Keep color limits consistent with the main tab

        if self.normalize_colors:                                                                  # Normalized colors always live in the same range
            return (-1.0, 1.0)                                                                      # Use the normalized amplitude range

        values = viewer.wavefield_snapshots[snapshot_ids].ravel()                                  # Read wavefield values from the selected snapshots
        vmin = float(np.percentile(values, 2.0))                                                    # Use a robust lower percentile to avoid extreme color outliers
        vmax = float(np.percentile(values, 98.0))                                                   # Use a robust upper percentile to avoid extreme color outliers
        abs_max = max(abs(vmin), abs(vmax))                                                        # Keep the color scale symmetric around zero
        if abs_max < 1e-20:                                                                        # Avoid a zero-width color range
            return (-1.0, 1.0)                                                                      # Fall back to a stable default range
        return (-abs_max, abs_max)                                                                 # Return symmetric pressure-amplitude limits

    def _build_ui(self):                                                                           # Build the two-simulation comparison layout and the right trace panel
        root_layout = QtWidgets.QHBoxLayout(self)                                                  # Put simulations on the left and traces on the right
        root_layout.setContentsMargins(6, 6, 6, 6)                                                 # Keep a compact margin around the comparison tab
        root_layout.setSpacing(8)                                                                  # Separate the simulation grid from the trace panel

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, self)                              # Put the synchronized scenes above the trace comparison
        root_layout.addWidget(self.splitter)                                                       # Fill the comparison tab with the splitter

        self.grid_container = QtWidgets.QWidget(self.splitter)                                     # Create the left container for the two PyVista comparison views
        self.grid_layout = QtWidgets.QGridLayout(self.grid_container)                              # Arrange the simulation views in a compact 1x2 grid
        self.grid_layout.setContentsMargins(8, 8, 8, 4)                                            # Keep a quiet frame around the render views
        self.grid_layout.setSpacing(8)                                                             # Keep a small visual gap between simulation cells
        self.grid_layout.setColumnStretch(0, 1)
        self.grid_layout.setColumnStretch(1, 1)
        self.grid_container.setMinimumHeight(470)
        self.splitter.addWidget(self.grid_container)                                               # Add the simulation grid to the splitter

        for cell_id in range(2):                                                                   # Build only two cells so the comparison stays responsive
            row = 0                                                                                # Keep both comparison simulations on the same row
            col = cell_id                                                                          # Place each comparison simulation in one column
            cell = QtWidgets.QFrame(self.grid_container)                                           # Create one framed cell for a PyVista view
            cell.setFrameShape(QtWidgets.QFrame.StyledPanel)                                       # Use a light frame to separate dense 3D views
            cell.setStyleSheet("QFrame { background-color: #2B2B2D; border: 1px solid #161618; }") # Give each PyVista cell a clean publication-style frame
            cell_layout = QtWidgets.QVBoxLayout(cell)                                              # Stack the small label above the PyVista interactor
            cell_layout.setContentsMargins(0, 0, 0, 0)                                             # Let the render view use nearly all available space
            cell_layout.setSpacing(0)                                                              # Keep the label attached to the scene
            label_text = self.viewer_labels[cell_id] if cell_id < len(self.viewer_labels) else "No sample"  # Use the selected simulation name instead of the local cell number
            label = QtWidgets.QLabel(label_text, cell)                                             # Label the cell with the selected simulation name
            label.setAlignment(QtCore.Qt.AlignCenter)                                              # Center the compact label above the scene
            label.setMinimumHeight(24)                                                             # Give the simulation label enough height to remain readable
            label.setStyleSheet("QLabel { color: #D8D8DA; background-color: #232325; border-bottom: 1px solid #161618; font-size: 11px; font-weight: 700; }") # Keep labels consistent with the main app
            cell_layout.addWidget(label)                                                           # Add the label to the cell

            if cell_id < len(self.viewers):                                                        # Populate only cells with available simulations
                plotter = QtInteractor(cell)                                                       # Create one full PyVista interactor for this simulation
                plotter.set_background(_BLENDER_VIEWPORT_BACKGROUND)                                                    # Keep the overview background consistent with the main viewer
                plotter.interactor.setMinimumHeight(430)
                cell_layout.addWidget(plotter.interactor, stretch=1)                               # Place the PyVista view inside the grid cell
                self.plotters.append(plotter)                                                      # Store the interactor for synchronized updates
                #--------------------------------------------------------
                velocity_overlay = self._build_velocity_overlay(                                    # Build a static 2D velocity-model overlay for this simulation cell
                    viewer=self.viewers[cell_id],                                                   # Use the velocity model that belongs to the current compared simulation
                    parent_widget=plotter.interactor,                                               # Parent the overlay to the PyVista render widget
                )
                self.velocity_overlay_canvases.append(velocity_overlay)                             # Store the overlay so it can be kept above the PyVista renderer
                legend = _QtColorLegend(self.cmap, self.clims[cell_id], parent=plotter.interactor)
                self.scalar_legends.append(legend)
                legend.raise_()
                #--------------------------------------------------------
            else:                                                                                  # Leave the second cell visible when only one simulation exists
                empty_label = QtWidgets.QLabel("No sample", cell)                                  # Show that this grid position has no simulation
                empty_label.setAlignment(QtCore.Qt.AlignCenter)                                    # Center the empty-cell message
                empty_label.setStyleSheet("color: #777777; font-size: 10px;")                      # Make the empty-cell message quiet
                cell_layout.addWidget(empty_label, stretch=1)                                      # Fill the empty cell with the placeholder label

            self.grid_layout.addWidget(cell, row, col)                                             # Add the cell to its comparison-grid position

        self.trace_panel = QtWidgets.QWidget(self.splitter)                                        # Create the lower panel for compact controls and four comparison traces
        self.trace_panel.setObjectName("comparison_trace_panel")                                    # Give the trace panel a scoped Qt style name
        self.trace_panel.setMinimumHeight(310)                                                     # Reserve enough vertical space for the trace comparison
        self.trace_panel.setStyleSheet(                                                            # Keep controls readable while leaving Matplotlib with a light figure background
            """
            QWidget#comparison_trace_panel {
                background-color: #28282B;
                border-top: 1px solid #161618;
            }
            QWidget#comparison_trace_panel QLabel {
                color: #D8D8DA;
                font-size: 10px;
                font-weight: 600;
            }
            QWidget#comparison_trace_panel QCheckBox {
                color: #D8D8DA;
                font-size: 10px;
                font-weight: 600;
                spacing: 4px;
            }
            QWidget#comparison_trace_panel QPushButton,
            QWidget#comparison_trace_panel QComboBox,
            QWidget#comparison_trace_panel QDoubleSpinBox {
                color: #D8D8DA;
                background-color: #333336;
                border: 1px solid #4B4B4F;
                border-radius: 3px;
                min-height: 22px;
                padding: 3px 6px;
            }
            QWidget#comparison_trace_panel QPushButton {
                color: #ffffff;
                background-color: #363639;
                border: 1px solid #4B4B4F;
            }
            QWidget#comparison_trace_panel QPushButton:hover,
            QWidget#comparison_trace_panel QComboBox:hover,
            QWidget#comparison_trace_panel QDoubleSpinBox:hover {
                border: 1px solid #68686D;
            }
            QWidget#comparison_trace_panel QComboBox QAbstractItemView {
                color: #D8D8DA;
                background-color: #333336;
                selection-color: #D8D8DA;
                selection-background-color: #E58A2B;
                border: 1px solid #4B4B4F;
                outline: 0;
            }
            QWidget#comparison_trace_panel QSlider::groove:horizontal {
                height: 4px;
                background: #48484C;
            }
            QWidget#comparison_trace_panel QSlider::handle:horizontal {
                width: 12px;
                margin: -5px 0;
                border-radius: 6px;
                background: #334155;
            }
            """
        )
        trace_layout = QtWidgets.QVBoxLayout(self.trace_panel)                                     # Stack controls over the trace canvas
        trace_layout.setContentsMargins(10, 6, 10, 8)                                              # Give the trace panel a little breathing room
        trace_layout.setSpacing(6)                                                                 # Keep controls close to the trace canvas

        trace_title = QtWidgets.QLabel("Trace Comparison", self.trace_panel)                       # Add a compact title for the right-side trace panel
        trace_title.setObjectName("comparison_trace_title")                                        # Give the title a stable style name
        trace_title.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)                     # Align the title with the trace content
        trace_title.setStyleSheet("font-size: 12px; font-weight: 700; color: #D8D8DA;")            # Make the trace panel title readable
        trace_layout.addWidget(trace_title)                                                        # Place the trace title above the compact controls

        controls_layout = QtWidgets.QGridLayout()                                                  # Use a compact grid for the shared comparison controls
        controls_layout.setContentsMargins(0, 0, 0, 0)                                             # Avoid extra control padding
        controls_layout.setHorizontalSpacing(6)                                                    # Keep controls readable without using much width
        controls_layout.setVerticalSpacing(3)                                                      # Keep the controls shallow so traces get most vertical space

        self.compare_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)                              # Create the shared synchronized frame slider
        self.compare_slider.setRange(0, self.max_frames - 1)                                       # Match the slider to the synchronized frame count
        self.compare_slider.setValue(0)                                                            # Start at the first synchronized frame
        self.compare_play_button = QtWidgets.QPushButton("Play")                                   # Create one shared play button for the compared simulations
        self.compare_play_button.setMinimumWidth(82)                                               # Keep the play button easy to click in the compact toolbar
        self.compare_cmap_combo = QtWidgets.QComboBox()                                            # Create one shared colormap selector for all simulations
        self.compare_cmap_combo.addItems(["seismic", "gray", "RdBu_r", "viridis", "turbo", "coolwarm"])  # Match the main viewer colormap choices
        self.compare_cmap_combo.setView(QtWidgets.QListView())
        self.compare_cmap_combo.view().setStyleSheet(
            "QListView { color: #D8D8DA; background: #303033; selection-background-color: #E58A2B; selection-color: #171719; }"
        )
        if self.cmap in [self.compare_cmap_combo.itemText(i) for i in range(self.compare_cmap_combo.count())]:  # Use the requested colormap when it exists
            self.compare_cmap_combo.setCurrentText(self.cmap)                                      # Initialize the comparison colormap
        self.compare_fps_spin = QtWidgets.QDoubleSpinBox()                                         # Create one shared playback-speed control
        self.compare_fps_spin.setRange(0.2, 60.0)                                                  # Match the main viewer playback-speed range
        self.compare_fps_spin.setDecimals(1)                                                       # Keep the frame-rate control compact
        self.compare_fps_spin.setSingleStep(1.0)                                                   # Use whole-FPS stepping by default
        self.compare_fps_spin.setValue(self.frame_rate)                                            # Initialize with the current frame rate
        self.compare_fps_spin.setMaximumWidth(92)                                                  # Keep the FPS control compact so traces keep most of the width
        self.compare_scale_spin = QtWidgets.QDoubleSpinBox()                                       # Create one shared vertical-scale control
        self.compare_scale_spin.setRange(0.0, 5000.0)                                              # Match the main viewer vertical-scale range
        self.compare_scale_spin.setDecimals(1)                                                     # Keep the scale control compact
        self.compare_scale_spin.setSingleStep(50.0)                                                # Match the main viewer scale stepping
        self.compare_scale_spin.setValue(self.vertical_scale)                                      # Initialize with the current vertical scale
        self.compare_scale_spin.setMaximumWidth(100)                                               # Keep the scale control compact so the trace panel has room
        self.compare_color_min_spin = QtWidgets.QDoubleSpinBox()
        self.compare_color_min_spin.setRange(-1.0e12, 1.0e12)
        self.compare_color_min_spin.setDecimals(4)
        self.compare_color_min_spin.setSingleStep(0.05)
        self.compare_color_min_spin.setKeyboardTracking(False)
        self.compare_color_min_spin.setValue(float(self.clims[0][0]))
        self.compare_color_min_spin.setMaximumWidth(94)
        self.compare_color_max_spin = QtWidgets.QDoubleSpinBox()
        self.compare_color_max_spin.setRange(-1.0e12, 1.0e12)
        self.compare_color_max_spin.setDecimals(4)
        self.compare_color_max_spin.setSingleStep(0.05)
        self.compare_color_max_spin.setKeyboardTracking(False)
        self.compare_color_max_spin.setValue(float(self.clims[0][1]))
        self.compare_color_max_spin.setMaximumWidth(94)
        self.compare_grid_check = QtWidgets.QCheckBox("Grid")
        self.compare_grid_check.setChecked(self.grid_visible)
        self.compare_mesh_check = QtWidgets.QCheckBox("Mesh")
        self.compare_mesh_check.setChecked(self.show_edges)
        self.compare_layer_check = QtWidgets.QCheckBox("Layers")
        self.compare_layer_check.setChecked(self.layer_outlines_visible)
        self.compare_lighting_check = QtWidgets.QCheckBox("Lighting")
        self.compare_lighting_check.setChecked(self.lighting_enabled)

        fps_label = QtWidgets.QLabel("FPS")                                                        # Build a visible label for the playback-speed control
        cmap_label = QtWidgets.QLabel("Cmap")                                                      # Build a visible label for the colormap selector
        scale_label = QtWidgets.QLabel("Scale")                                                    # Build a visible label for the vertical-scale control
        cmin_label = QtWidgets.QLabel("Min")
        cmax_label = QtWidgets.QLabel("Max")
        controls_layout.addWidget(self.compare_play_button, 0, 0)                                  # Put play first for fast access
        controls_layout.addWidget(fps_label, 0, 1)                                                 # Label the frame-rate spin box
        controls_layout.addWidget(self.compare_fps_spin, 0, 2)                                     # Add the frame-rate spin box
        controls_layout.addWidget(cmap_label, 0, 3)                                                # Label the colormap selector
        controls_layout.addWidget(self.compare_cmap_combo, 0, 4)                                   # Add the colormap selector
        controls_layout.addWidget(scale_label, 0, 5)                                               # Label the vertical-scale control
        controls_layout.addWidget(self.compare_scale_spin, 0, 6)                                   # Add the vertical-scale control
        controls_layout.addWidget(cmin_label, 0, 7)
        controls_layout.addWidget(self.compare_color_min_spin, 0, 8)
        controls_layout.addWidget(cmax_label, 0, 9)
        controls_layout.addWidget(self.compare_color_max_spin, 0, 10)
        controls_layout.addWidget(self.compare_grid_check, 1, 0)
        controls_layout.addWidget(self.compare_mesh_check, 1, 1)
        controls_layout.addWidget(self.compare_layer_check, 1, 2)
        controls_layout.addWidget(self.compare_lighting_check, 1, 3)
        controls_layout.addWidget(self.compare_slider, 2, 0, 1, 11)                                 # Put the time slider below the compact controls
        trace_layout.addLayout(controls_layout)                                                    # Add shared controls above the trace canvas

        self.trace_figure = Figure(figsize=(12.8, 3.2), facecolor=_BLENDER_PLOT_BACKGROUND, tight_layout=True)   # Create a wide light-background figure for up to four comparison traces
        self.trace_canvas = FigureCanvas(self.trace_figure)                                       # Embed the trace figure in Qt
        trace_layout.addWidget(self.trace_canvas, stretch=1)                                      # Give the trace canvas most of the right panel
        self._build_trace_axes()                                                                  # Draw the first and middle trace for each simulation

        self.splitter.addWidget(self.trace_panel)                                                  # Add the trace panel to the splitter
        self.splitter.setStretchFactor(0, 5)                                                       # Give the simulation grid substantial space
        self.splitter.setStretchFactor(1, 2)                                                       # Keep traces readable without shrinking the scenes

        self.timer = QtCore.QTimer(self)                                                           # Create the synchronized comparison playback timer
        self.timer.setInterval(max(int(1000 / self.frame_rate), 1))                                # Initialize timer interval from FPS

    def _build_velocity_overlay(self, viewer, parent_widget):                                      # Build one static velocity-model overlay in the PyVista view
        #--------------------------------------------------------
        overlay_figure = Figure(                                                                   # Create a small Matplotlib figure for the 2D velocity model
            figsize=(1.85, 1.35),                                                                  # Keep the overlay compact inside the animation window
            dpi=100,                                                                               # Use a stable pixel density for the overlay
            facecolor=_BLENDER_PLOT_BACKGROUND,                                                                  # Use a white figure background for readability over the 3D view
            tight_layout=True,                                                                     # Reduce unused margins around the velocity image
        )
        overlay_canvas = FigureCanvas(overlay_figure)                                              # Embed the velocity figure as a Qt canvas
        overlay_canvas.setParent(parent_widget)                                                    # Attach the overlay directly to the PyVista render widget
        overlay_canvas.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)                  # Let mouse events pass through to the PyVista interactor
        overlay_canvas.setGeometry(12, 12, 185, 135)                                               # Place the overlay in the upper-left corner of the PyVista viewport
        overlay_canvas.setStyleSheet("background-color: #28282B; border: 1px solid #55555A;")        # Add a thin border so the overlay stays visible on light backgrounds

        overlay_axis = overlay_figure.add_subplot(111)                                             # Create the 2D velocity-model axis
        overlay_axis.imshow(                                                                       # Draw the velocity model as a static 2D image
            viewer.velocity_model,                                                                 # Use the current simulation velocity field
            cmap=self.velocity_overlay_cmap,                                                       # Use viridis by default for all 2D velocity overlays
            origin="upper",                                                                       # Keep array orientation consistent with the stored model
            aspect="auto",                                                                        # Fill the compact overlay without changing its fixed size
        )
        overlay_axis.set_title("Velocity", fontsize=7, fontweight="bold", pad=1, color=_BLENDER_PLOT_TEXT)                  # Add a short label so the inset is self-explanatory
        overlay_axis.set_xticks([])                                                               # Hide x ticks to keep the overlay compact
        overlay_axis.set_yticks([])                                                               # Hide z ticks to keep the overlay compact
        for spine in overlay_axis.spines.values():                                                 # Keep a visible frame around the velocity image
            spine.set_linewidth(0.6)                                                              # Use a thin axis frame
            spine.set_color("#77777C")                                                           # Use a dark frame for contrast
        overlay_canvas.draw_idle()                                                                # Draw the velocity overlay once because it is static
        overlay_canvas.raise_()                                                                   # Keep the overlay above the PyVista renderer
        return overlay_canvas                                                                     # Return the overlay canvas for later raise calls
        #--------------------------------------------------------

    def _comparison_scalar_bar_args(self):                                                        # Build compact scalar-bar settings for comparison PyVista views
        #--------------------------------------------------------
        return {                                                                                   # Return one shared scalar-bar configuration dictionary
            "title": "Amplitude",                                                                  # Keep the same amplitude meaning as the single-simulation tab
            "vertical": False,                                                                     # Use a horizontal scalar bar to preserve vertical render space
            "position_x": 0.28,                                                                    # Center the scalar bar near the bottom of the viewport
            "position_y": 0.045,                                                                   # Keep the scalar bar close to the bottom edge
            "width": 0.44,                                                                         # Keep the scalar bar compact inside each comparison cell
            "height": 0.045,                                                                       # Keep the scalar bar thin so it does not cover the wavefield
            "title_font_size": 8,                                                                  # Use small title text for the comparison viewport
            "label_font_size": 7,                                                                  # Use small tick labels for the comparison viewport
            "color": _BLENDER_PLOT_TEXT,                                                                     # Use black scalar-bar text on the white render background
            "fmt": "%.2f",                                                                        # Keep labels short so they fit in the compact bar
        }
        #--------------------------------------------------------

    def _build_trace_axes(self):                                                                   # Build the four-trace comparison plot
        axes = self.trace_figure.subplots(len(self.viewers), 2, sharex=True)                       # Create two trace axes for every available simulation
        axes = np.asarray(axes, dtype=object)                                                       # Normalize Matplotlib axes into an array for indexing
        if axes.ndim == 1:                                                                          # Handle the single-simulation comparison edge case
            axes = axes.reshape(1, -1)                                                             # Keep indexing as [simulation, trace_column]

        self.trace_axes = axes                                                                     # Store trace axes for later reference
        self.trace_time_lines = []                                                                 # Reset time cursor storage before plotting traces
        trace_line_colors = ["#F0F0F2", "#B8DFF2"]                                                 # Keep waveform colors neutral for publication-style trace comparison
        cursor_colors = ["#F0A15A", "#E58A2B"]                                                     # Use distinct cursor colors for the compared simulations

        for viewer_id, viewer in enumerate(self.viewers):                                          # Loop over every simulation shown in the comparison grid
            viewer_label = self.viewer_labels[viewer_id]                                           # Read the real selected simulation label for this trace row
            seismogram_data = viewer.seismogram_data                                               # Read the prepared seismogram data for this simulation
            if seismogram_data is None:                                                            # Skip trace plotting when no seismogram data is available
                for col in range(2):                                                               # Mark both trace columns as unavailable
                    axes[viewer_id, col].text(0.5, 0.5, "No trace data", ha="center", va="center")  # Show a compact missing-data message
                    axes[viewer_id, col].set_axis_off()                                            # Hide axes that have no trace data
                self.trace_time_lines.append([])                                                   # Keep a matching empty cursor list for this simulation
                continue                                                                           # Move to the next simulation

            seismograms = seismogram_data["surface_seismograms"]                                  # Read all receiver traces for this simulation
            time_axis = seismogram_data["time_axis"]                                               # Read the seismogram time axis
            n_receivers = int(seismograms.shape[0])                                                # Count available receiver traces
            receiver_ids = [0, n_receivers // 2]                                                   # Select the first receiver and the receiver at the middle of the array
            selected_traces = seismograms[receiver_ids]                                            # Read the two selected traces
            abs_limit = float(np.percentile(np.abs(selected_traces), 99.0))                        # Use a robust amplitude limit for the two traces
            if abs_limit < 1e-20:                                                                  # Avoid a flat y-axis when traces are all zeros
                abs_limit = 1.0                                                                    # Use a stable default trace amplitude range
            abs_limit *= 1.15                                                                      # Add a small visual margin around the trace amplitudes

            row_lines = []                                                                         # Store the two time cursors for this simulation
            for col, receiver_id in enumerate(receiver_ids):                                       # Plot the first and middle trace columns
                ax = axes[viewer_id, col]                                                          # Select the current trace axis
                trace = seismograms[int(receiver_id)]

                _apply_blender_plot_theme(ax)                                                        # Use a white axis background inside the light trace panel
                ax.plot(time_axis, trace, color=trace_line_colors[col], linewidth=0.65, alpha=0.94) # Draw the trace as a compact neutral waveform
                time_line = ax.axvline(                                                            # Add the synchronized time cursor
                    float(viewer.snapshot_times[int(self.snapshot_ids_by_viewer[viewer_id][0])]),  # Use this simulation's first selected snapshot time
                    color=cursor_colors[viewer_id % len(cursor_colors)],                           # Use a distinct cursor color for each compared simulation
                    linewidth=1.0,                                                                 # Keep the cursor readable in small axes
                    linestyle="--",                                                               # Make the cursor visually distinct from the waveform
                    alpha=0.72,                                                                    # Keep the cursor present without overpowering the trace
                )
                row_lines.append(time_line)                                                        # Store the cursor line for playback updates
                ax.set_ylim(-abs_limit, abs_limit)                                                 # Use the same amplitude range for the two traces in one simulation
                ax.grid(True, axis="x", linestyle="--", alpha=0.16)                                # Add a light time grid for reading arrivals
                ax.grid(True, axis="y", linestyle=":", alpha=0.10)                                 # Add a very light amplitude grid
                ax.tick_params(axis="both", labelsize=7, width=0.6, length=3)                      # Keep tick labels readable in the trace panel
                ax.set_ylabel("Amp.", fontsize=7)                                                   # Keep the y-axis label compact and physically meaningful
                if viewer_id == 0:                                                                 # Put column labels only on the first row to reduce clutter
                    ax.set_title("Receiver 1" if col == 0 else f"Receiver {int(receiver_id) + 1}", fontsize=8, fontweight="bold", pad=4) # Label the selected receiver columns
                ax.text(                                                                           # Add a clear per-axis simulation and receiver label
                    0.015,                                                                         # Place the label near the left edge of the axis
                    0.86,                                                                          # Place the label near the top of the axis
                    f"{viewer_label} | ST{int(receiver_id) + 1}",                                  # Show the real selected simulation and station number
                    transform=ax.transAxes,                                                        # Position the label in axis-relative coordinates
                    fontsize=7,                                                                    # Keep the label compact inside each subplot
                    fontweight="bold",                                                            # Make the simulation label easy to scan
                    color=_BLENDER_PLOT_TEXT,                                                              # Use dark text on the white trace background
                    ha="left",                                                                    # Anchor the label from the left
                    va="center",                                                                  # Vertically center the label box
                    bbox={                                                                         # Give the label a subtle white backing
                        "boxstyle": "round,pad=0.18",                                             # Use a small rounded text box
                        "facecolor": "#38383C",                                                   # Keep the label box consistent with the plot background
                        "edgecolor": "#66666A",                                                   # Add a quiet outline so the label is visible over traces
                        "alpha": 0.86,                                                            # Keep the label readable without feeling heavy
                    },
                )
                for spine in ax.spines.values():                                                   # Thin the axis spines for dense plotting
                    spine.set_linewidth(0.6)                                                       # Use compact but visible spine weight
                    spine.set_color("#555555")                                                     # Use a softer axis frame than pure black
            self.trace_time_lines.append(row_lines)                                                # Store this simulation's trace cursors

        for ax in axes[-1, :]:                                                                      # Label time only on the bottom row
            ax.set_xlabel("Time [s]", fontsize=7)                                                   # Add a compact time-axis label

        self.trace_canvas.draw_idle()                                                              # Draw the trace panel once after building all axes

    def _build_case_surfaces(self, viewer, field):
        field = np.asarray(field, dtype=_VISUAL_FLOAT_DTYPE)
        lod = max(int(self.lod_factor), 1)
        if lod > 1:
            field = field[::lod, ::lod]
            x_grid = viewer.X[::lod, ::lod]
            z_grid = viewer.Z[::lod, ::lod]
        else:
            x_grid = viewer.X
            z_grid = viewer.Z

        y_top = viewer._scale_wavefield_for_geometry(
            field=field,
            vertical_scale=self.vertical_scale,
            normalize=self.normalize,
        )
        thickness = self._resolve_case_domain_thickness(viewer)
        y_back = y_top - thickness
        color_field = viewer._scale_wavefield_for_colors(
            field=field,
            normalize_colors=self.normalize_colors,
        )

        top_surface = pv.StructuredGrid(x_grid, y_top, z_grid)
        top_surface["Amplitude"] = np.asarray(color_field, dtype=_VISUAL_FLOAT_DTYPE).ravel(order="F")
        back_surface = pv.StructuredGrid(x_grid, y_back, z_grid)
        back_surface["Amplitude"] = np.asarray(color_field, dtype=_VISUAL_FLOAT_DTYPE).ravel(order="F")

        def side_from_edges(x_edge, y_edge, z_edge, color_edge):
            side_surface = pv.StructuredGrid(
                np.column_stack([x_edge, x_edge]),
                np.column_stack([y_edge, y_edge - thickness]),
                np.column_stack([z_edge, z_edge]),
            )
            side_surface["Amplitude"] = np.column_stack([color_edge, color_edge]).ravel(order="F")
            return side_surface

        side_surfaces = [
            back_surface,
            side_from_edges(x_grid[0, :], y_top[0, :], z_grid[0, :], color_field[0, :]),
            side_from_edges(x_grid[-1, :], y_top[-1, :], z_grid[-1, :], color_field[-1, :]),
            side_from_edges(x_grid[:, 0], y_top[:, 0], z_grid[:, 0], color_field[:, 0]),
            side_from_edges(x_grid[:, -1], y_top[:, -1], z_grid[:, -1], color_field[:, -1]),
        ]
        return top_surface, side_surfaces

    def _build_case_layer_outline_polydata(self, viewer, field):
        field = np.asarray(field, dtype=_VISUAL_FLOAT_DTYPE)
        lod = max(int(self.lod_factor), 1)
        if lod > 1:
            field = field[::lod, ::lod]
            x_grid = viewer.X[::lod, ::lod]
            z_grid = viewer.Z[::lod, ::lod]
            velocity_model = viewer.velocity_model[::lod, ::lod]
        else:
            x_grid = viewer.X
            z_grid = viewer.Z
            velocity_model = viewer.velocity_model

        y_top = viewer._scale_wavefield_for_geometry(
            field=field,
            vertical_scale=self.vertical_scale,
            normalize=self.normalize,
        )
        velocity_values = np.asarray(velocity_model, dtype=_VISUAL_FLOAT_DTYPE)
        finite_values = velocity_values[np.isfinite(velocity_values)]
        if finite_values.size == 0:
            return pv.PolyData()

        unique_values = np.unique(np.round(finite_values, decimals=6))
        if len(unique_values) > 1 and len(unique_values) <= 16:
            contour_levels = 0.5 * (unique_values[:-1] + unique_values[1:])
        else:
            contour_levels = np.percentile(finite_values, [20.0, 40.0, 60.0, 80.0])
            contour_levels = np.unique(np.round(contour_levels, decimals=6))

        contour_levels = contour_levels[
            (contour_levels > float(np.nanmin(finite_values)))
            & (contour_levels < float(np.nanmax(finite_values)))
        ]
        if len(contour_levels) == 0:
            return pv.PolyData()

        contour_grid = pv.StructuredGrid(x_grid, np.zeros_like(x_grid, dtype=_VISUAL_FLOAT_DTYPE), z_grid)
        contour_grid["Velocity"] = np.asarray(velocity_model, dtype=_VISUAL_FLOAT_DTYPE).ravel(order="F")
        try:
            contour_lines = contour_grid.contour(isosurfaces=contour_levels, scalars="Velocity")
        except Exception:
            return pv.PolyData()

        if contour_lines.n_points == 0 or contour_lines.lines.size == 0:
            return pv.PolyData()

        contour_points = np.asarray(contour_lines.points)
        contour_cells = np.asarray(contour_lines.lines, dtype=np.int_)
        x_min = float(np.nanmin(x_grid))
        x_max = float(np.nanmax(x_grid))
        z_min = float(np.nanmin(z_grid))
        dx = float(abs(x_grid[0, 1] - x_grid[0, 0])) if x_grid.shape[1] > 1 else 1.0
        dz = float(abs(z_grid[1, 0] - z_grid[0, 0])) if z_grid.shape[0] > 1 else 1.0
        y_offset = max(float(self.vertical_scale) * 0.004, float(dx) * 0.006)
        side_offset = max(min(dx, dz) * 0.30, 1.0e-6)
        thickness = self._resolve_case_domain_thickness(viewer)

        def sample_y(x_value, z_value):
            col = np.clip((float(x_value) - x_min) / dx, 0.0, y_top.shape[1] - 1.0)
            row = np.clip((float(z_value) - z_min) / dz, 0.0, y_top.shape[0] - 1.0)
            col0 = int(np.floor(col))
            row0 = int(np.floor(row))
            col1 = min(col0 + 1, y_top.shape[1] - 1)
            row1 = min(row0 + 1, y_top.shape[0] - 1)
            tx = col - col0
            tz = row - row0
            y00 = y_top[row0, col0]
            y01 = y_top[row0, col1]
            y10 = y_top[row1, col0]
            y11 = y_top[row1, col1]
            return float(
                (1.0 - tx) * (1.0 - tz) * y00
                + tx * (1.0 - tz) * y01
                + (1.0 - tx) * tz * y10
                + tx * tz * y11
            )

        def smooth_polyline(coords):
            coords = np.asarray(coords, dtype=_VISUAL_FLOAT_DTYPE)
            if len(coords) < 7:
                return coords
            smoothed = coords.copy()
            for _ in range(3):
                next_coords = smoothed.copy()
                next_coords[1:-1] = (
                    0.25 * smoothed[:-2]
                    + 0.50 * smoothed[1:-1]
                    + 0.25 * smoothed[2:]
                )
                smoothed = next_coords
            return smoothed

        def side_offset_point(point):
            x_value, y_value, z_value = point
            if abs(x_value - x_min) <= 1.5 * dx:
                x_value = x_min - side_offset
            elif abs(x_value - x_max) <= 1.5 * dx:
                x_value = x_max + side_offset
            elif abs(z_value - z_min) <= 1.5 * dz:
                z_value = z_min - side_offset
            elif abs(z_value - float(np.nanmax(z_grid))) <= 1.5 * dz:
                z_value = float(np.nanmax(z_grid)) + side_offset
            return (float(x_value), float(y_value), float(z_value))

        outline_points = []
        outline_lines = []

        cursor = 0
        while cursor < len(contour_cells):
            n_points = int(contour_cells[cursor])
            point_ids = contour_cells[cursor + 1 : cursor + 1 + n_points]
            cursor += n_points + 1
            if n_points < 2:
                continue

            front_coords = []
            back_coords = []
            for point_id in point_ids:
                x_value = float(contour_points[int(point_id), 0])
                z_value = float(contour_points[int(point_id), 2])
                y_value = sample_y(x_value, z_value)
                front_coords.append((x_value, y_value + y_offset, z_value))
                back_coords.append((x_value, y_value - thickness - y_offset, z_value))

            front_coords = smooth_polyline(front_coords)
            back_coords = smooth_polyline(back_coords)
            front_ids = []
            back_ids = []
            for front_point, back_point in zip(front_coords, back_coords):
                front_ids.append(len(outline_points))
                outline_points.append(tuple(front_point))
                back_ids.append(len(outline_points))
                outline_points.append(tuple(back_point))

            outline_lines.extend([len(front_ids), *front_ids])
            outline_lines.extend([len(back_ids), *back_ids])

            first_point = np.asarray(outline_points[front_ids[0]])
            last_point = np.asarray(outline_points[front_ids[-1]])
            closed_line = np.linalg.norm(first_point[[0, 2]] - last_point[[0, 2]]) < max(dx, dz) * 1.5
            if not closed_line:
                for front_id, back_id in ((front_ids[0], back_ids[0]), (front_ids[-1], back_ids[-1])):
                    side_front = side_offset_point(outline_points[front_id])
                    side_back = side_offset_point(outline_points[back_id])
                    side_front_id = len(outline_points)
                    outline_points.append(side_front)
                    side_back_id = len(outline_points)
                    outline_points.append(side_back)
                    outline_lines.extend([2, front_id, side_front_id])
                    outline_lines.extend([2, side_front_id, side_back_id])
                    outline_lines.extend([2, side_back_id, back_id])

        def add_side_face_interfaces(edge_col, x_value, x_direction):
            x_side = float(x_value + x_direction * side_offset)
            edge_velocity = velocity_model[:, edge_col]
            edge_y = y_top[:, edge_col]
            edge_z = z_grid[:, edge_col]
            for level in contour_levels:
                edge_crossings = np.where(
                    (edge_velocity[:-1] - level) * (edge_velocity[1:] - level) <= 0.0
                )[0]
                for row in edge_crossings:
                    v0 = float(edge_velocity[row])
                    v1 = float(edge_velocity[row + 1])
                    if abs(v1 - v0) < 1.0e-12:
                        continue
                    ratio = float(np.clip((float(level) - v0) / (v1 - v0), 0.0, 1.0))
                    z_value = float((1.0 - ratio) * edge_z[row] + ratio * edge_z[row + 1])
                    y_value = float((1.0 - ratio) * edge_y[row] + ratio * edge_y[row + 1])
                    front_id = len(outline_points)
                    outline_points.append((x_side, y_value + y_offset, z_value))
                    back_id = len(outline_points)
                    outline_points.append((x_side, y_value - thickness - y_offset, z_value))
                    outline_lines.extend([2, front_id, back_id])

        add_side_face_interfaces(0, x_min, -1.0)
        add_side_face_interfaces(-1, x_max, 1.0)

        if not outline_points:
            return pv.PolyData()

        outline_polydata = pv.PolyData(np.asarray(outline_points, dtype=_VISUAL_FLOAT_DTYPE))
        outline_polydata.lines = np.asarray(outline_lines, dtype=np.int_)
        return outline_polydata

    def _add_case_layer_outline_actor(self, plotter, polydata, viewer_id):
        self.layer_outline_polydata_by_viewer[viewer_id] = polydata
        self.layer_outline_actors_by_viewer[viewer_id] = None
        if not self.layer_outlines_visible or polydata.n_points == 0:
            return
        self.layer_outline_actors_by_viewer[viewer_id] = plotter.add_mesh(
            polydata,
            name="comparison_layer_outlines",
            color="#4b5560",
            opacity=0.24,
            line_width=0.35,
            render_lines_as_tubes=False,
            lighting=False,
            show_scalar_bar=False,
        )

    def _remove_case_layer_outline_actor(self, plotter, viewer_id):
        try:
            plotter.remove_actor("comparison_layer_outlines", reset_camera=False, render=False)
        except Exception:
            pass
        self.layer_outline_actors_by_viewer[viewer_id] = None
        self.layer_outline_polydata_by_viewer[viewer_id] = None

    def _update_case_layer_outline_actor(self, viewer_id, field):
        if not self.layer_outlines_visible:
            return
        plotter = self.plotters[viewer_id]
        new_polydata = self._build_case_layer_outline_polydata(self.viewers[viewer_id], field)
        polydata = self.layer_outline_polydata_by_viewer[viewer_id]
        actor = self.layer_outline_actors_by_viewer[viewer_id]
        if polydata is None or actor is None:
            self._add_case_layer_outline_actor(plotter, new_polydata, viewer_id)
            return
        polydata.copy_from(new_polydata)
        polydata.Modified()
        actor.mapper.SetInputData(polydata)
        actor.mapper.Update()
        actor.mapper.Modified()

    def _add_case_side_actors(self, plotter, side_surfaces, viewer_id):
        self.side_surfaces_by_viewer[viewer_id] = list(side_surfaces)
        self.side_actors_by_viewer[viewer_id] = []
        for side_id, side_surface in enumerate(side_surfaces):
            actor = plotter.add_mesh(
                side_surface,
                name=f"comparison_side_{side_id}",
                scalars="Amplitude",
                cmap=self.cmap,
                clim=self.clims[viewer_id],
                opacity=1.0,
                show_edges=self.show_edges,
                edge_color="#111827",
                line_width=0.32,
                smooth_shading=True,
                lighting=self.lighting_enabled,
                ambient=0.42 if self.lighting_enabled else 1.0,
                diffuse=0.62 if self.lighting_enabled else 0.0,
                specular=0.08 if self.lighting_enabled else 0.0,
                specular_power=14.0,
                show_scalar_bar=False,
            )
            self.side_actors_by_viewer[viewer_id].append(actor)

    def _add_case_wavefield_actor(self, plotter, surface, viewer_id, show_scalar_bar=True):
        return plotter.add_mesh(
            surface,
            name="wavefield_surface",
            scalars="Amplitude",
            cmap=self.cmap,
            clim=self.clims[viewer_id],
            show_edges=self.show_edges,
            edge_color="#111827",
            line_width=0.30,
            smooth_shading=True,
            lighting=self.lighting_enabled,
            ambient=0.36 if self.lighting_enabled else 1.0,
            diffuse=0.70 if self.lighting_enabled else 0.0,
            specular=0.10 if self.lighting_enabled else 0.0,
            specular_power=18.0,
            show_scalar_bar=False,
            scalar_bar_args=self._comparison_scalar_bar_args(),
        )

    def _configure_case_lighting(self, plotter, viewer):
        try:
            plotter.remove_all_lights()
        except Exception:
            pass
        if not self.lighting_enabled:
            try:
                renderer = plotter.renderer
                if hasattr(renderer, "UseShadowsOff"):
                    renderer.UseShadowsOff()
            except Exception:
                pass
            return

        x_mid = 0.5 * (viewer.nx - 1) * viewer.dx
        z_mid = 0.5 * (viewer.nz - 1) * viewer.dz
        x_span = (viewer.nx - 1) * viewer.dx
        z_span = (viewer.nz - 1) * viewer.dz
        span = max(x_span, z_span)
        thickness = self._resolve_case_domain_thickness(viewer)
        focal_point = (x_mid, -0.2 * thickness, z_mid)
        for position, intensity in [
            ((x_mid - 0.9 * x_span, 2.3 * span, z_mid - 0.85 * z_span), 0.58),
            ((x_mid + 0.9 * x_span, 0.85 * span, z_mid + 0.70 * z_span), 0.22),
        ]:
            try:
                plotter.add_light(
                    pv.Light(
                        position=position,
                        focal_point=focal_point,
                        color="white",
                        intensity=float(intensity),
                    )
                )
            except Exception:
                pass

        try:
            renderer = plotter.renderer
            if self.shadows_enabled and hasattr(renderer, "UseShadowsOn"):
                renderer.UseShadowsOn()
            elif hasattr(renderer, "UseShadowsOff"):
                renderer.UseShadowsOff()
        except Exception:
            pass

    def _resolve_case_domain_thickness(self, viewer):                                              # Resolve the comparison block thickness for one simulation
        if self.domain_thickness is not None:                                                      # Use the explicit notebook value when provided
            return float(self.domain_thickness)                                                    # Keep comparison and presentation scenes visually consistent
        return max(float(self.vertical_scale) * 1.75, float(viewer.dx) * 16.0)                     # Preserve the original automatic thickness

    def _build_scenes(self):                                                                       # Build all PyVista scenes for the comparison grid
        for viewer_id, (viewer, plotter) in enumerate(zip(self.viewers, self.plotters)):           # Loop over every populated grid cell
            snapshot_id = int(self.snapshot_ids_by_viewer[viewer_id][0])                           # Use the first synchronized snapshot for the initial surface
            surface, side_surfaces = self._build_case_surfaces(                                    # Build the same thickened presentation domain used by the main view
                viewer=viewer,
                field=viewer.wavefield_snapshots[snapshot_id],
            )
            self.surfaces.append(surface)                                                          # Store the current surface for this simulation
            self.side_surfaces_by_viewer.append([])
            self.side_actors_by_viewer.append([])
            self._add_case_side_actors(plotter, side_surfaces, viewer_id)
            actor = self._add_case_wavefield_actor(plotter, surface, viewer_id, show_scalar_bar=True)
            self.mesh_actors.append(actor)                                                         # Store the actor for update operations
            self.layer_outline_polydata_by_viewer.append(None)
            self.layer_outline_actors_by_viewer.append(None)
            self._add_case_layer_outline_actor(
                plotter,
                self._build_case_layer_outline_polydata(viewer, viewer.wavefield_snapshots[snapshot_id]),
                viewer_id,
            )

            if self.show_source_receivers:                                                         # Add markers when requested by the main viewer configuration
                viewer._add_source_and_receivers(                                                  # Draw source and receiver markers on this full PyVista view
                    plotter=plotter,                                                               # Use the current cell plotter
                    marker_elevation=float(self.vertical_scale) * 0.08,                            # Match the presentation-domain marker elevation
                    point_size=self.point_size,                                                    # Use compact markers for the two-simulation overview
                    show_labels=self.show_labels,                                                  # Keep labels disabled by default for readability
                )

            if self.grid_visible:
                plotter.show_bounds(
                    grid="front",
                    location="outer",
                    xtitle="X",
                    ytitle="Amplitude",
                    ztitle="Z",
                    font_size=8,
                )

            try:                                                                                   # Apply optional PyVista rendering enhancements when available
                plotter.enable_anti_aliasing()                                                     # Smooth edges in each comparison cell
            except Exception:                                                                      # Keep the viewer robust across different PyVista/VTK versions
                pass                                                                               # Continue even if these rendering options are unavailable
            self._configure_case_lighting(plotter, viewer)

            self._set_case_oblique_camera(viewer, plotter)                                         # Set the same oblique camera style as the main viewer
            if self.orthographic:                                                                  # Apply orthographic projection when requested
                plotter.enable_parallel_projection()                                               # Use parallel projection for stable visual comparison
            plotter.render()                                                                       # Render the initial comparison scene

    def _set_case_oblique_camera(self, viewer, plotter):                                           # Set the oblique camera for one comparison cell
        x_mid = 0.5 * (viewer.nx - 1) * viewer.dx                                                  # Compute the physical x midpoint
        z_mid = 0.5 * (viewer.nz - 1) * viewer.dz                                                  # Compute the physical z midpoint
        x_span = (viewer.nx - 1) * viewer.dx                                                       # Compute the physical x span
        z_span = (viewer.nz - 1) * viewer.dz                                                       # Compute the physical z span
        span = max(x_span, z_span)                                                                 # Use the larger domain span to place the camera
        plotter.camera_position = [                                                                # Use a strong three-quarter vertical X-Z section view
            (x_mid - 1.86 * x_span, 1.72 * span, z_mid - 0.22 * z_span),                          # Emphasize the side face while keeping depth vertical
            (x_mid, 0.0, z_mid),                                                                   # Set the focal point at the model center
            (0.0, 0.0, -1.0),                                                                      # Keep shallow depths visually above deeper zones
        ]
        plotter.camera.zoom(1.02)                                                                  # Fill the comparison cell with the vertical section

    def _copy_case_surface_data(self, target_surface, source_surface, actor=None):                 # Copy PyVista data while keeping the existing actor alive
        target_surface.copy_from(source_surface)                                                   # Copy points, topology, and arrays into the persistent mesh
        target_surface.Modified()                                                                  # Mark the dataset as modified for VTK
        points = target_surface.GetPoints()                                                        # Read the VTK point container when present
        if points is not None:                                                                     # Mark point coordinates as changed
            points.Modified()                                                                      # Notify VTK that geometry points changed
            point_data = points.GetData()                                                          # Read the raw coordinate array
            if point_data is not None:                                                             # Mark the raw point array as changed too
                point_data.Modified()                                                              # Ensure renderers do not reuse stale coordinate buffers
        vtk_point_data = target_surface.GetPointData()                                             # Read scalar/vector arrays attached to the points
        if vtk_point_data is not None:                                                             # Mark every scalar array as modified
            vtk_point_data.Modified()                                                              # Notify VTK that point data changed
            for array_id in range(vtk_point_data.GetNumberOfArrays()):                             # Loop through all point arrays
                array = vtk_point_data.GetArray(array_id)                                          # Read the current VTK array
                if array is not None:                                                              # Mark the array when it exists
                    array.Modified()                                                               # Ensure scalar colors refresh without actor replacement
        if actor is not None:                                                                      # Refresh the mapper without replacing the actor
            actor.mapper.SetInputData(target_surface)                                              # Keep the mapper attached to the persistent dataset
            actor.mapper.Update()                                                                  # Force mapper update
            actor.mapper.Modified()                                                                # Mark the mapper as modified

    def _replace_case_surface_data(self, viewer_id, new_surface, new_side_surfaces):               # Refresh one comparison PyVista surface
        self._copy_case_surface_data(                                                              # Update the front wavefield face in place
            self.surfaces[viewer_id],
            new_surface,
            actor=self.mesh_actors[viewer_id],
        )
        for side_surface, new_side_surface, side_actor in zip(
            self.side_surfaces_by_viewer[viewer_id],
            new_side_surfaces,
            self.side_actors_by_viewer[viewer_id],
        ):
            self._copy_case_surface_data(side_surface, new_side_surface, actor=side_actor)         # Update each block side without removing its actor

    def _update_frame(self, frame_id):                                                             # Update every simulation and trace cursor to one synchronized frame
        frame_id = int(np.clip(frame_id, 0, self.max_frames - 1))                                  # Clamp the requested frame to the synchronized range
        self.current_frame = frame_id                                                              # Store the current synchronized frame

        for viewer_id, viewer in enumerate(self.viewers):                                          # Loop over every simulation in the comparison tab
            snapshot_id = int(self.snapshot_ids_by_viewer[viewer_id][frame_id])                    # Read this simulation's snapshot id at the synchronized frame
            field = viewer.wavefield_snapshots[snapshot_id]                                       # Read the full wavefield snapshot for this simulation
            cache_key = ("comparison_presentation", viewer_id) + _mesh_cache_key(
                snapshot_id=snapshot_id,
                vertical_scale=self.vertical_scale,
                normalize=self.normalize,
                deform_surface=self.deform_surface,
                normalize_colors=self.normalize_colors,
                lod_factor=self.lod_factor,
            )
            new_surface = viewer._mesh_cache.get(cache_key)
            if new_surface is None:
                new_surface = self._build_case_surfaces(viewer=viewer, field=field)
                viewer._mesh_cache.put(cache_key, new_surface)
            surface, side_surfaces = new_surface
            self._replace_case_surface_data(viewer_id, surface, side_surfaces)                     # Refresh this simulation's scene
            self._update_case_layer_outline_actor(viewer_id, field)                                # Keep layer outlines synchronized with the animated wavefield

            time_value = float(viewer.snapshot_times[snapshot_id])                                 # Read this simulation's physical time at the current frame
            for line in self.trace_time_lines[viewer_id]:                                          # Update both trace cursors for this simulation
                line.set_xdata([time_value, time_value])                                           # Move the trace time cursor to the current simulation time

            self.plotters[viewer_id].render()                                                      # Render the refreshed comparison cell
            self.plotters[viewer_id].update()                                                      # Flush the QtInteractor update
            if viewer_id < len(self.velocity_overlay_canvases):                                    # Keep the velocity-model overlay above the refreshed PyVista renderer
                self.velocity_overlay_canvases[viewer_id].raise_()                                 # Raise the static velocity overlay after every render update
            if viewer_id < len(self.scalar_legends):
                self.scalar_legends[viewer_id].raise_()

        if self.compare_slider.value() != frame_id:                                                # Keep the shared slider synchronized with playback
            self.compare_slider.blockSignals(True)                                                 # Avoid recursive updates while moving the slider programmatically
            self.compare_slider.setValue(frame_id)                                                 # Move the slider to the current frame
            self.compare_slider.blockSignals(False)                                                # Restore slider signals

        self.trace_canvas.draw_idle()                                                              # Redraw the 18 trace cursors

    def _connect_events(self):                                                                     # Connect compact comparison controls
        self.compare_slider.valueChanged.connect(self._update_frame)                               # Let the slider drive all simulations
        self.compare_play_button.clicked.connect(self._toggle_playback)                            # Let the play button control synchronized playback
        self.compare_fps_spin.valueChanged.connect(self._set_frame_rate)                           # Let the FPS control update timer speed
        self.compare_cmap_combo.currentTextChanged.connect(self._set_colormap)                     # Let the colormap control update all simulations
        self.compare_scale_spin.valueChanged.connect(self._set_vertical_scale)                     # Let the scale control update all simulations
        self.compare_color_min_spin.valueChanged.connect(self._set_color_limits_from_controls)
        self.compare_color_max_spin.valueChanged.connect(self._set_color_limits_from_controls)
        self.compare_grid_check.toggled.connect(self._set_grid_visible)
        self.compare_mesh_check.toggled.connect(self._set_mesh_visible)
        self.compare_layer_check.toggled.connect(self._set_layer_outlines_visible)
        self.compare_lighting_check.toggled.connect(self._set_lighting_enabled)
        self.timer.timeout.connect(self._timer_tick)                                               # Advance all simulations on each timer tick

    def _toggle_playback(self):                                                                    # Toggle synchronized comparison playback
        if self.timer.isActive():                                                                  # Stop playback when the timer is active
            self.timer.stop()                                                                      # Stop the synchronized timer
            self.compare_play_button.setText("Play")                                               # Restore the play label
        else:                                                                                      # Start playback when the timer is inactive
            self.timer.start()                                                                     # Start the synchronized timer
            self.compare_play_button.setText("Pause")                                              # Show that playback is running

    def _timer_tick(self):                                                                         # Advance the synchronized comparison frame
        next_frame = (self.current_frame + 1) % self.max_frames                                    # Loop playback through the synchronized frame range
        self._update_frame(next_frame)                                                             # Draw the next synchronized frame

    def _set_frame_rate(self, value):                                                              # Update synchronized comparison playback speed
        self.frame_rate = max(float(value), 0.2)                                                   # Store a safe positive frame rate
        self.timer.setInterval(max(int(1000 / self.frame_rate), 1))                                # Update the timer interval in milliseconds

    def _set_vertical_scale(self, value):                                                          # Update vertical scale across every comparison simulation
        self.vertical_scale = float(value)                                                         # Store the new shared vertical scale
        for viewer in self.viewers:
            viewer._mesh_cache.clear()
        self._update_frame(self.current_frame)                                                     # Rebuild every visible surface at the current frame

    def _rebuild_case_actors(self, viewer_id):
        plotter = self.plotters[viewer_id]
        plotter.remove_actor("wavefield_surface", reset_camera=False, render=False)
        for side_id in range(len(self.side_surfaces_by_viewer[viewer_id])):
            plotter.remove_actor(f"comparison_side_{side_id}", reset_camera=False, render=False)
        self._add_case_side_actors(plotter, self.side_surfaces_by_viewer[viewer_id], viewer_id)
        self.mesh_actors[viewer_id] = self._add_case_wavefield_actor(
            plotter,
            self.surfaces[viewer_id],
            viewer_id,
            show_scalar_bar=True,
        )
        self._configure_case_lighting(plotter, self.viewers[viewer_id])
        if viewer_id < len(self.scalar_legends):
            self.scalar_legends[viewer_id].set_scale(self.cmap, self.clims[viewer_id])
            self.scalar_legends[viewer_id].raise_()
        plotter.render()
        plotter.update()

    def _set_color_limits_from_controls(self):
        color_min = float(self.compare_color_min_spin.value())
        color_max = float(self.compare_color_max_spin.value())
        if color_min >= color_max:
            return
        self.clims = [(color_min, color_max) for _ in self.viewers]
        for viewer_id in range(len(self.plotters)):
            self._rebuild_case_actors(viewer_id)

    def _set_grid_visible(self, visible):
        self.grid_visible = bool(visible)
        for viewer, plotter in zip(self.viewers, self.plotters):
            try:
                plotter.remove_bounds_axes()
            except Exception:
                pass
            if self.grid_visible:
                plotter.show_bounds(
                    grid="front",
                    location="outer",
                    xtitle="X",
                    ytitle="Amplitude",
                    ztitle="Z",
                    font_size=8,
                )
            plotter.render()

    def _set_mesh_visible(self, visible):
        self.show_edges = bool(visible)
        for viewer_id in range(len(self.plotters)):
            self._rebuild_case_actors(viewer_id)

    def _set_layer_outlines_visible(self, visible):
        self.layer_outlines_visible = bool(visible)
        for viewer_id, plotter in enumerate(self.plotters):
            if self.layer_outlines_visible:
                snapshot_id = int(self.snapshot_ids_by_viewer[viewer_id][self.current_frame])
                self._add_case_layer_outline_actor(
                    plotter,
                    self._build_case_layer_outline_polydata(
                        self.viewers[viewer_id],
                        self.viewers[viewer_id].wavefield_snapshots[snapshot_id],
                    ),
                    viewer_id,
                )
            else:
                self._remove_case_layer_outline_actor(plotter, viewer_id)
            plotter.render()
            plotter.update()

    def _set_lighting_enabled(self, visible):
        self.lighting_enabled = bool(visible)
        for viewer_id in range(len(self.plotters)):
            self._rebuild_case_actors(viewer_id)

    def _set_colormap(self, cmap):                                                                 # Update colormap across every comparison simulation
        self.cmap = str(cmap)                                                                      # Store the new shared colormap
        for viewer_id in range(len(self.plotters)):                                                # Loop over every comparison plotter
            self._rebuild_case_actors(viewer_id)                                                   # Rebuild all comparison actors with the new colormap
        self._position_scalar_legends()
#--------------------------------------------------------


class _RetiredGradientTabWidget(QtWidgets.QWidget):
    """Retained only for backward compatibility; the app no longer exposes this tab."""

    def __init__(self, parent_window):
        super().__init__(parent_window)
        self.parent_window = parent_window
        self.figure = None
        self.canvas = None
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def update(self, *args, **kwargs):
        return
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        grad_x = grad_z = magnitude = None

        nz, nx = field.shape
        step = 4
        z_indices, x_indices = np.mgrid[0:nz:step, 0:nx:step]
        z_coords = z_indices.ravel() * dz
        x_coords = x_indices.ravel() * dx
        grad_x_sample = grad_x[::step, ::step].ravel()
        grad_z_sample = grad_z[::step, ::step].ravel()

        mask = np.isfinite(grad_x_sample) & np.isfinite(grad_z_sample)
        x_coords = x_coords[mask]
        z_coords = z_coords[mask]
        grad_x_sample = grad_x_sample[mask]
        grad_z_sample = grad_z_sample[mask]

        if len(x_coords) > 0:
            ax.quiver(
                x_coords,
                z_coords,
                grad_x_sample,
                grad_z_sample,
                magnitude[::step, ::step].ravel()[mask],
                cmap="hot",
                scale=20,
                width=0.002,
            )

        ax.set_xlabel("X (m)", fontsize=10)
        ax.set_ylabel("Z (m)", fontsize=10)
        ax.set_title("Pressure Gradient Field (∇P)", fontsize=12, fontweight="bold")
        ax.invert_yaxis()
        self.figure.tight_layout()
        self.canvas.draw_idle()


class _PresentationDomainWidget(QtWidgets.QWidget):
    """Presentation scene with a solid thickened domain block and wavefield."""

    def __init__(
        self,
        viewer,
        snapshot_ids,
        vertical_scale,
        normalize,
        normalize_colors,
        cmap,
        clim,
        show_source_receivers,
        point_size,
        frame_rate,
        orthographic,
        lod_factor,
        sample_keys=None,
        sample_loader=None,
        snapshot_export_dir=None,
        lighting_enabled=True,
        shadows_enabled=True,
        grid_visible=True,
        mesh_visible=True,
        layer_outlines_visible=True,
        domain_thickness=None,
        parent=None,
    ):
        super().__init__(parent)
        self.viewer = viewer
        self.snapshot_ids = np.asarray(snapshot_ids, dtype=int)
        self.vertical_scale = float(vertical_scale)
        self.normalize = bool(normalize)
        self.normalize_colors = bool(normalize_colors)
        self.cmap = str(cmap)
        self.clim = clim
        if self.clim is None:
            self.clim = (-1.0, 1.0) if self.normalize_colors else self._resolve_color_limits()
        self.show_source_receivers = bool(show_source_receivers)
        self.point_size = int(point_size)
        self.frame_rate = max(float(frame_rate), 0.2)
        self.orthographic = bool(orthographic)
        self.lod_factor = max(int(lod_factor), 1)
        self.sample_keys = [] if sample_keys is None else list(sample_keys)
        self.sample_loader = sample_loader
        self.snapshot_export_dir = None if snapshot_export_dir is None else Path(snapshot_export_dir)
        self.domain_thickness_override = None if domain_thickness is None else float(domain_thickness)
        if self.domain_thickness_override is not None and self.domain_thickness_override <= 0.0:
            raise ValueError("domain_thickness must be positive or None.")
        self.domain_thickness = self._resolve_domain_thickness(self.viewer)
        self.snapshot_start = int(self.snapshot_ids[0]) if len(self.snapshot_ids) else 0
        self.snapshot_step = int(self.snapshot_ids[1] - self.snapshot_ids[0]) if len(self.snapshot_ids) > 1 else 1
        self.current_frame = 0
        self.surface = None
        self.mesh_actor = None
        self.side_surfaces = []
        self.side_actors = []
        self.bounds_visible = bool(grid_visible)
        self.axes_visible = True
        self.edges_visible = bool(mesh_visible)
        self.lighting_enabled = bool(lighting_enabled)
        self.shadows_enabled = bool(shadows_enabled)
        self.layer_outlines_visible = bool(layer_outlines_visible)
        self.layer_outline_actor = None
        self.layer_outline_polydata = None
        self.seismogram_time_lines = []
        self.velocity_overlay_cmap = "viridis"
        self.velocity_overlay_canvas = None
        self.scalar_legend = None
        self.velocity_overlay_axis = None
        self.scene_label_widget = None

        self._build_ui()
        self._build_scene()
        self._connect_events()
        self._update_frame(0)
        self._position_overlays()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_overlays()

    def _position_overlays(self):
        if self.velocity_overlay_canvas is not None:
            self.velocity_overlay_canvas.setGeometry(16, 16, 215, 155)
            self.velocity_overlay_canvas.raise_()

        if self.scalar_legend is not None:
            legend_width = min(max(int(self.render_container.width() * 0.48), 320), 560)
            self.scalar_legend.setGeometry(
                max((self.render_container.width() - legend_width) // 2, 14),
                max(self.render_container.height() - 66, 14),
                legend_width,
                54,
            )
            self.scalar_legend.raise_()

        if self.scene_label_widget is not None:
            margin = 14
            self.scene_label_widget.adjustSize()
            label_size = self.scene_label_widget.size()
            container_rect = self.render_container.rect()
            x_pos = max(container_rect.width() - label_size.width() - margin, margin)
            self.scene_label_widget.move(x_pos, margin)
            self.scene_label_widget.raise_()

    def _build_velocity_overlay(self, parent_widget):
        overlay_figure = Figure(
            figsize=(2.15, 1.55),
            dpi=100,
            facecolor=_BLENDER_PLOT_BACKGROUND,
            tight_layout=True,
        )
        overlay_canvas = FigureCanvas(overlay_figure)
        overlay_canvas.setParent(parent_widget)
        overlay_canvas.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        overlay_canvas.setGeometry(16, 16, 215, 155)
        overlay_canvas.setStyleSheet("background-color: #28282B; border: 1px solid #55555A;")

        self.velocity_overlay_canvas = overlay_canvas
        self.velocity_overlay_axis = overlay_figure.add_subplot(111)
        self._draw_velocity_overlay()
        overlay_canvas.raise_()
        return overlay_canvas

    def _draw_velocity_overlay(self):
        if self.velocity_overlay_axis is None:
            return

        self.velocity_overlay_axis.clear()
        self.velocity_overlay_axis.imshow(
            self.viewer.velocity_model,
            cmap=self.velocity_overlay_cmap,
            origin="upper",
            aspect="auto",
        )
        self.velocity_overlay_axis.set_title("Velocity", fontsize=8, fontweight="bold", pad=1, color=_BLENDER_PLOT_TEXT)
        self.velocity_overlay_axis.set_xticks([])
        self.velocity_overlay_axis.set_yticks([])
        for spine in self.velocity_overlay_axis.spines.values():
            spine.set_linewidth(0.7)
            spine.set_color("#77777C")
        if self.velocity_overlay_canvas is not None:
            self.velocity_overlay_canvas.draw_idle()

    def _resolve_color_limits(self):
        values = self.viewer.wavefield_snapshots[self.snapshot_ids].ravel()
        vmin = float(np.percentile(values, 2.0))
        vmax = float(np.percentile(values, 98.0))
        abs_max = max(abs(vmin), abs(vmax))
        if abs_max < 1e-20:
            return (-1.0, 1.0)
        return (-abs_max, abs_max)

    def _resolve_domain_thickness(self, viewer):
        if self.domain_thickness_override is not None:
            return float(self.domain_thickness_override)
        return max(float(self.vertical_scale) * 1.75, float(viewer.dx) * 16.0)

    def _build_ui(self):
        root_layout = QtWidgets.QHBoxLayout(self)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(8)

        # Single Simulation uses a Blender-like editor workspace: viewport and
        # seismic timeline at left, scene/properties dock at right.
        self.workspace_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        self.workspace_splitter.setChildrenCollapsible(False)
        root_layout.addWidget(self.workspace_splitter)
        self.viewport_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, self.workspace_splitter)
        self.viewport_splitter.setChildrenCollapsible(False)
        self.workspace_splitter.addWidget(self.viewport_splitter)

        viewport_editor = QtWidgets.QFrame(self.viewport_splitter)
        viewport_editor.setObjectName("viewportEditor")
        viewport_layout = QtWidgets.QVBoxLayout(viewport_editor)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        viewport_layout.setSpacing(0)
        viewport_header = QtWidgets.QFrame(viewport_editor)
        viewport_header.setObjectName("viewportToolbar")
        viewport_header_layout = QtWidgets.QHBoxLayout(viewport_header)
        viewport_header_layout.setContentsMargins(7, 0, 7, 0)
        viewport_header_layout.setSpacing(3)
        viewport_title = QtWidgets.QLabel("3D Viewport", viewport_header)
        viewport_title.setObjectName("editorType")
        viewport_header_layout.addWidget(viewport_title)
        for label, callback in (("View", self._set_presentation_camera), ("Top", self._set_top_camera), ("Left", self._set_left_camera), ("Right", self._set_right_camera), ("Frame", lambda: self.plotter.reset_camera())):
            action = QtWidgets.QToolButton(viewport_header)
            action.setText(label)
            action.setObjectName("editorMenuButton")
            action.setAutoRaise(True)
            action.clicked.connect(callback)
            viewport_header_layout.addWidget(action)
        viewport_header_layout.addStretch(1)
        mode = QtWidgets.QLabel("Object Mode", viewport_header)
        mode.setObjectName("editorMode")
        viewport_header_layout.addWidget(mode)
        viewport_layout.addWidget(viewport_header)

        self.render_container = QtWidgets.QWidget(self)
        self.render_container.setObjectName("render_container")
        render_layout = QtWidgets.QVBoxLayout(self.render_container)
        render_layout.setContentsMargins(0, 0, 0, 0)
        render_layout.setSpacing(0)

        self.plotter = QtInteractor(self.render_container)
        self.plotter.interactor.setMinimumHeight(620)
        render_layout.addWidget(self.plotter.interactor)
        self.velocity_overlay_canvas = self._build_velocity_overlay(self.plotter.interactor)
        self.scalar_legend = _QtColorLegend(self.cmap, self.clim, parent=self.plotter.interactor)
        self.scalar_legend.raise_()

        if self.viewer.scene_label:
            self.scene_label_widget = QtWidgets.QLabel(self.viewer.scene_label, self.render_container)
            self.scene_label_widget.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.scene_label_widget.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
            self.scene_label_widget.setStyleSheet(
                """
                QLabel {
                    color: #E8E8EA;
                    background-color: rgba(35, 35, 37, 224);
                    border: 1px solid rgba(125, 125, 130, 170);
                    border-radius: 4px;
                    padding: 5px 9px;
                    font-size: 14px;
                    font-weight: 600;
                }
                """
            )
            self.scene_label_widget.adjustSize()

        viewport_layout.addWidget(self.render_container, stretch=1)
        self.viewport_splitter.addWidget(viewport_editor)

        self.side_panel = QtWidgets.QFrame(self)
        self.side_panel.setObjectName("sidePanel")
        self.side_panel.setMinimumWidth(410)
        self.side_panel.setMaximumWidth(520)
        panel_layout = QtWidgets.QVBoxLayout(self.side_panel)
        panel_layout.setContentsMargins(10, 8, 10, 8)
        panel_layout.setSpacing(7)

        outliner_header = QtWidgets.QLabel("OUTLINER  /  SCENE")
        outliner_header.setObjectName("muted")
        panel_layout.addWidget(outliner_header)
        outliner = QtWidgets.QFrame(self.side_panel)
        outliner.setObjectName("outliner")
        outliner.setMaximumHeight(92)
        outliner_layout = QtWidgets.QVBoxLayout(outliner)
        outliner_layout.setContentsMargins(10, 7, 10, 7)
        outliner_layout.setSpacing(3)
        scene_name = self.viewer.scene_label or "WAVEFIELD"
        scene_item = QtWidgets.QLabel(f"Collection  /  {scene_name}")
        scene_item.setObjectName("sceneItem")
        outliner_layout.addWidget(scene_item)
        for label in (">  Wavefield surface", ">  Velocity model overlay", ">  Source and receivers"):
            item = QtWidgets.QLabel(label)
            item.setObjectName("muted")
            outliner_layout.addWidget(item)
        panel_layout.addWidget(outliner)
        properties_header = QtWidgets.QLabel("PROPERTIES  /  VIEWPORT")
        properties_header.setObjectName("muted")
        properties_header.setContentsMargins(0, 5, 0, 0)
        panel_layout.addWidget(properties_header)

        info_group = QtWidgets.QGroupBox("Simulation")
        info_layout = QtWidgets.QFormLayout(info_group)
        self.sample_combo = QtWidgets.QComboBox()
        self.sample_combo.addItems(self.sample_keys)
        if self.viewer.scene_label and self.viewer.scene_label in self.sample_keys:
            self.sample_combo.setCurrentText(self.viewer.scene_label)
        self.sample_combo.setEnabled(bool(self.sample_keys) and self.sample_loader is not None)
        self.info_model = QtWidgets.QLabel(str(self.viewer.model_type))
        self.info_shape = QtWidgets.QLabel(
            f"{self.viewer.nz} x {self.viewer.nx} | {len(self.snapshot_ids)} frames"
        )
        self.info_frame = QtWidgets.QLabel("-")
        self.info_time = QtWidgets.QLabel("-")
        self.info_amp = QtWidgets.QLabel("-")
        info_layout.addRow("Sample", self.sample_combo)
        info_layout.addRow("Model", self.info_model)
        info_layout.addRow("Grid", self.info_shape)
        info_layout.addRow("Frame", self.info_frame)
        info_layout.addRow("Time", self.info_time)
        info_layout.addRow("Max |P|", self.info_amp)
        panel_layout.addWidget(info_group)

        camera_group = QtWidgets.QGroupBox("Camera")
        camera_layout = QtWidgets.QGridLayout(camera_group)
        self.btn_oblique = QtWidgets.QPushButton("Oblique")
        self.btn_top = QtWidgets.QPushButton("Top")
        self.btn_left = QtWidgets.QPushButton("Left")
        self.btn_right = QtWidgets.QPushButton("Right")
        self.btn_fit = QtWidgets.QPushButton("Fit")
        camera_layout.addWidget(self.btn_oblique, 0, 0)
        camera_layout.addWidget(self.btn_top, 0, 1)
        camera_layout.addWidget(self.btn_left, 1, 0)
        camera_layout.addWidget(self.btn_right, 1, 1)
        camera_layout.addWidget(self.btn_fit, 2, 0, 1, 2)
        panel_layout.addWidget(camera_group)

        display_group = QtWidgets.QGroupBox("Display")
        display_layout = QtWidgets.QFormLayout(display_group)
        self.chk_bounds = QtWidgets.QCheckBox("Show grid and ticks")
        self.chk_bounds.setChecked(self.bounds_visible)
        self.chk_axes = QtWidgets.QCheckBox("Show orientation axes")
        self.chk_axes.setChecked(True)
        self.chk_edges = QtWidgets.QCheckBox("Show mesh grid")
        self.chk_edges.setChecked(self.edges_visible)
        self.chk_layer_outlines = QtWidgets.QCheckBox("Show layer outlines")
        self.chk_layer_outlines.setChecked(self.layer_outlines_visible)
        self.chk_lighting = QtWidgets.QCheckBox("Use lighting/shading")
        self.chk_lighting.setChecked(self.lighting_enabled)
        self.scale_spin = QtWidgets.QDoubleSpinBox()
        self.scale_spin.setRange(0.0, 5000.0)
        self.scale_spin.setDecimals(1)
        self.scale_spin.setSingleStep(50.0)
        self.scale_spin.setValue(self.vertical_scale)
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(["seismic", "gray", "RdBu_r", "viridis", "turbo", "coolwarm"])
        if self.cmap in [self.cmap_combo.itemText(i) for i in range(self.cmap_combo.count())]:
            self.cmap_combo.setCurrentText(self.cmap)
        self.color_min_spin = QtWidgets.QDoubleSpinBox()
        self.color_min_spin.setRange(-1.0e12, 1.0e12)
        self.color_min_spin.setDecimals(4)
        self.color_min_spin.setSingleStep(0.05)
        self.color_min_spin.setKeyboardTracking(False)
        self.color_min_spin.setValue(float(self.clim[0]))
        self.color_max_spin = QtWidgets.QDoubleSpinBox()
        self.color_max_spin.setRange(-1.0e12, 1.0e12)
        self.color_max_spin.setDecimals(4)
        self.color_max_spin.setSingleStep(0.05)
        self.color_max_spin.setKeyboardTracking(False)
        self.color_max_spin.setValue(float(self.clim[1]))
        display_layout.addRow(self.chk_bounds)
        display_layout.addRow(self.chk_axes)
        display_layout.addRow(self.chk_edges)
        display_layout.addRow(self.chk_layer_outlines)
        display_layout.addRow(self.chk_lighting)
        display_layout.addRow("Vertical scale", self.scale_spin)
        display_layout.addRow("Color map", self.cmap_combo)
        display_layout.addRow("Color min", self.color_min_spin)
        display_layout.addRow("Color max", self.color_max_spin)
        panel_layout.addWidget(display_group)

        playback_group = QtWidgets.QGroupBox("Playback")
        playback_layout = QtWidgets.QGridLayout(playback_group)
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setRange(0, len(self.snapshot_ids) - 1)
        self.frame_slider.setValue(0)
        self.btn_play = QtWidgets.QPushButton("Play")
        self.btn_save_svg = QtWidgets.QPushButton("Save SVG")
        self.btn_save_svg.setEnabled(self.snapshot_export_dir is not None)
        if self.snapshot_export_dir is None:
            self.btn_save_svg.setToolTip("Set snapshot_export_folder_name in 6.0 to enable SVG export.")
        self.frame_label = QtWidgets.QLabel("-")
        self.speed_spin = QtWidgets.QDoubleSpinBox()
        self.speed_spin.setRange(0.2, 60.0)
        self.speed_spin.setDecimals(1)
        self.speed_spin.setSingleStep(1.0)
        self.speed_spin.setValue(self.frame_rate)
        playback_layout.addWidget(self.frame_slider, 0, 0, 1, 3)
        playback_layout.addWidget(self.btn_play, 1, 0)
        playback_layout.addWidget(self.btn_save_svg, 1, 1)
        playback_layout.addWidget(self.frame_label, 1, 2)
        playback_layout.addWidget(QtWidgets.QLabel("FPS"), 2, 0)
        playback_layout.addWidget(self.speed_spin, 2, 1, 1, 2)
        panel_layout.addWidget(playback_group)
        panel_layout.addStretch(1)
        panel_layout.addStretch(0)
        footer = QtWidgets.QLabel("PROPERTIES  /  Non-destructive viewport controls")
        footer.setObjectName("footerLabel")
        footer.setContentsMargins(14, 4, 14, 2)
        panel_layout.addWidget(footer)
        self.trace_editor = QtWidgets.QFrame(self.viewport_splitter)
        self.trace_editor.setObjectName("traceEditor")
        trace_editor_layout = QtWidgets.QVBoxLayout(self.trace_editor)
        trace_editor_layout.setContentsMargins(8, 0, 8, 8)
        trace_editor_layout.setSpacing(5)
        trace_header = QtWidgets.QFrame(self.trace_editor)
        trace_header.setObjectName("timelineToolbar")
        trace_header_layout = QtWidgets.QHBoxLayout(trace_header)
        trace_header_layout.setContentsMargins(7, 0, 7, 0)
        trace_header_layout.setSpacing(4)
        trace_title = QtWidgets.QLabel("Dope Sheet  /  Seismograms", trace_header)
        trace_title.setObjectName("editorType")
        trace_header_layout.addWidget(trace_title)
        trace_header_layout.addStretch(1)
        trace_status = QtWidgets.QLabel("Playback follows current frame", trace_header)
        trace_status.setObjectName("muted")
        trace_header_layout.addWidget(trace_status)
        trace_editor_layout.addWidget(trace_header)
        self.trace_group = self._build_plot_group()
        self.trace_group.setObjectName("seismogramEditor")
        self.trace_group.setMinimumHeight(220)
        trace_editor_layout.addWidget(self.trace_group, stretch=1)
        self.viewport_splitter.addWidget(self.trace_editor)

        self.workspace_splitter.addWidget(self.side_panel)
        self.workspace_splitter.setStretchFactor(0, 5)
        self.workspace_splitter.setStretchFactor(1, 2)
        self.workspace_splitter.setSizes([1320, 430])
        self.viewport_splitter.setStretchFactor(0, 5)
        self.viewport_splitter.setStretchFactor(1, 2)
        self.viewport_splitter.setSizes([650, 290])

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(max(int(1000 / self.frame_rate), 1))

    def _build_plot_group(self):
        if self.viewer.seismogram_data is None:
            group = QtWidgets.QGroupBox("Wavefield Summary")
            layout = QtWidgets.QVBoxLayout(group)
            self.trace_figure = Figure(figsize=(4.2, 2.8), facecolor=_BLENDER_PLOT_BACKGROUND, tight_layout=True)
            self.trace_canvas = FigureCanvas(self.trace_figure)
            ax = self.trace_figure.add_subplot(111)
            _apply_blender_plot_theme(ax)
            amps = np.max(np.abs(self.viewer.wavefield_snapshots[self.snapshot_ids]), axis=(1, 2))
            ax.plot(np.arange(len(amps)), amps, color="#69C0E8", linewidth=1.2)
            self.summary_time_line = ax.axvline(0, color="#F0A15A", linewidth=1.2, alpha=0.8)
            ax.set_title("Wavefield Amplitude Timeline", fontsize=11, fontweight="bold", color=_BLENDER_PLOT_TEXT)
            ax.set_xlabel("Frame")
            ax.set_ylabel("Max |P|")
            ax.grid(True, alpha=0.25)
            layout.addWidget(self.trace_canvas)
            return group

        group = QtWidgets.QGroupBox("Surface Seismograms")
        layout = QtWidgets.QVBoxLayout(group)
        self.trace_figure = Figure(figsize=(4.8, 4.1), facecolor=_BLENDER_PLOT_BACKGROUND, tight_layout=True)
        self.trace_canvas = FigureCanvas(self.trace_figure)
        self._draw_seismogram_axes()
        layout.addWidget(self.trace_canvas)
        return group

    def _draw_seismogram_axes(self):
        self.trace_figure.clear()
        self.seismogram_time_lines = []
        seismogram_data = self.viewer.seismogram_data
        if seismogram_data is None:
            self.trace_canvas.draw_idle()
            return

        seismograms = seismogram_data["surface_seismograms"]
        time_axis = seismogram_data["time_axis"]
        receiver_x = seismogram_data["receiver_x"]
        receiver_z = seismogram_data["receiver_z"]
        selected_receivers = seismogram_data["selected_receivers"]

        axes = self.trace_figure.subplots(len(selected_receivers), 1, sharex=True)
        axes = np.atleast_1d(axes)
        selected_traces = seismograms[selected_receivers]
        abs_limit = float(np.percentile(np.abs(selected_traces), 99.0))
        if abs_limit < 1e-20:
            abs_limit = 1.0
        abs_limit *= 1.15

        first_time = float(self.viewer.snapshot_times[int(self.snapshot_ids[0])])
        for ax, receiver_id in zip(axes, selected_receivers):
            trace = seismograms[int(receiver_id)]
            _apply_blender_plot_theme(ax)
            ax.plot(time_axis, trace, color="#F0F0F2", linewidth=0.85, alpha=0.92)
            time_line = ax.axvline(
                first_time,
                color="#F0A15A",
                linewidth=1.0,
                linestyle="--",
                alpha=0.65,
            )
            self.seismogram_time_lines.append(time_line)

            if int(receiver_id) < len(receiver_x) and int(receiver_id) < len(receiver_z):
                receiver_label = f"ST{int(receiver_id) + 1}  ix={int(receiver_x[int(receiver_id)])}"
            else:
                receiver_label = f"ST{int(receiver_id) + 1}"
            ax.text(
                0.01,
                0.82,
                receiver_label,
                transform=ax.transAxes,
                fontsize=8,
                fontweight="bold",
                color="#F0F0F2",
                ha="left",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "#38383C",
                    "edgecolor": "none",
                    "alpha": 0.72,
                },
            )
            ax.set_ylim(-abs_limit, abs_limit)
            ax.set_ylabel("Amp.", fontsize=8)
            ax.grid(True, axis="x", linestyle="--", alpha=0.28)
            ax.grid(True, axis="y", linestyle=":", alpha=0.16)
            ax.tick_params(axis="both", labelsize=8, width=0.7, length=3)
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)

        axes[0].set_title("Surface Seismograms", fontsize=11, fontweight="bold", pad=8, color=_BLENDER_PLOT_TEXT)
        axes[-1].set_xlabel("Time [s]", fontsize=9)
        self.trace_canvas.draw_idle()

    def _build_thickened_surfaces(self, field):
        field = np.asarray(field, dtype=_VISUAL_FLOAT_DTYPE)
        lod = max(int(self.lod_factor), 1)
        if lod > 1:
            field = field[::lod, ::lod]
            x_grid = self.viewer.X[::lod, ::lod]
            z_grid = self.viewer.Z[::lod, ::lod]
        else:
            x_grid = self.viewer.X
            z_grid = self.viewer.Z

        y_top = self.viewer._scale_wavefield_for_geometry(
            field=field,
            vertical_scale=self.vertical_scale,
            normalize=self.normalize,
        )
        thickness = float(self.domain_thickness)
        y_back = y_top - thickness

        color_field = self.viewer._scale_wavefield_for_colors(
            field=field,
            normalize_colors=self.normalize_colors,
        )
        top_surface = pv.StructuredGrid(x_grid, y_top, z_grid)
        top_surface["Amplitude"] = np.asarray(color_field, dtype=_VISUAL_FLOAT_DTYPE).ravel(order="F")
        top_surface["RawAmplitude"] = np.asarray(field, dtype=_VISUAL_FLOAT_DTYPE).ravel(order="F")
        back_surface = pv.StructuredGrid(x_grid, y_back, z_grid)
        back_surface["Amplitude"] = np.asarray(color_field, dtype=_VISUAL_FLOAT_DTYPE).ravel(order="F")

        def side_from_edges(x_edge, y_edge, z_edge, color_edge):
            side_surface = pv.StructuredGrid(
                np.column_stack([x_edge, x_edge]),
                np.column_stack([y_edge, y_edge - thickness]),
                np.column_stack([z_edge, z_edge]),
            )
            side_surface["Amplitude"] = np.column_stack([color_edge, color_edge]).ravel(order="F")
            return side_surface

        side_surfaces = [
            back_surface,
            side_from_edges(x_grid[0, :], y_top[0, :], z_grid[0, :], color_field[0, :]),
            side_from_edges(x_grid[-1, :], y_top[-1, :], z_grid[-1, :], color_field[-1, :]),
            side_from_edges(x_grid[:, 0], y_top[:, 0], z_grid[:, 0], color_field[:, 0]),
            side_from_edges(x_grid[:, -1], y_top[:, -1], z_grid[:, -1], color_field[:, -1]),
        ]
        return top_surface, side_surfaces

    def _update_thickened_surface_arrays(self, field):
        """Update the existing VTK topology with the exact current-frame values."""
        field = np.asarray(field, dtype=_VISUAL_FLOAT_DTYPE)
        lod = max(int(self.lod_factor), 1)
        if lod > 1:
            field = field[::lod, ::lod]
        y_top = self.viewer._scale_wavefield_for_geometry(
            field=field,
            vertical_scale=self.vertical_scale,
            normalize=self.normalize,
        )
        color_field = self.viewer._scale_wavefield_for_colors(
            field=field,
            normalize_colors=self.normalize_colors,
        )
        top_y = y_top.ravel(order="F")
        colors = np.asarray(color_field, dtype=_VISUAL_FLOAT_DTYPE).ravel(order="F")
        raw = np.asarray(field, dtype=_VISUAL_FLOAT_DTYPE).ravel(order="F")
        thickness = float(self.domain_thickness)

        self.surface.points[:, 1] = top_y
        self.surface["Amplitude"][:] = colors
        self.surface["RawAmplitude"][:] = raw
        back = self.side_surfaces[0]
        back.points[:, 1] = top_y - thickness
        back["Amplitude"][:] = colors

        edge_specs = (
            (1, y_top[0, :], color_field[0, :]),
            (2, y_top[-1, :], color_field[-1, :]),
            (3, y_top[:, 0], color_field[:, 0]),
            (4, y_top[:, -1], color_field[:, -1]),
        )
        for side_id, edge_y, edge_colors in edge_specs:
            side = self.side_surfaces[side_id]
            edge_y = np.asarray(edge_y, dtype=_VISUAL_FLOAT_DTYPE)
            edge_colors = np.asarray(edge_colors, dtype=_VISUAL_FLOAT_DTYPE)
            side.points[:, 1] = np.concatenate((edge_y, edge_y - thickness))
            side["Amplitude"][:] = np.concatenate((edge_colors, edge_colors))

        self._mark_surface_modified(self.surface, self.mesh_actor)
        for side, actor in zip(self.side_surfaces, self.side_actors):
            self._mark_surface_modified(side, actor)
    def _mark_surface_modified(self, surface, actor):
        surface.Modified()
        points = surface.GetPoints()
        if points is not None:
            points.Modified()
            data = points.GetData()
            if data is not None:
                data.Modified()
        point_data = surface.GetPointData()
        if point_data is not None:
            point_data.Modified()
            for array_id in range(point_data.GetNumberOfArrays()):
                array = point_data.GetArray(array_id)
                if array is not None:
                    array.Modified()
        if actor is not None:
            actor.mapper.SetInputData(surface)
            actor.mapper.Update()
            actor.mapper.Modified()
    def _add_wavefield_actor(self, surface, show_scalar_bar=True):
        self.mesh_actor = self.plotter.add_mesh(
            surface,
            name="presentation_wavefield_surface",
            scalars="Amplitude",
            cmap=self.cmap,
            clim=self.clim,
            opacity=1.0,
            show_edges=self.edges_visible,
            edge_color="#111827",
            line_width=0.35,
            smooth_shading=True,
            lighting=self.lighting_enabled,
            ambient=0.36 if self.lighting_enabled else 1.0,
            diffuse=0.70 if self.lighting_enabled else 0.0,
            specular=0.10 if self.lighting_enabled else 0.0,
            specular_power=18.0,
            show_scalar_bar=False,
            scalar_bar_args={
                "title": "Amplitude",
                "vertical": False,
                "position_x": 0.17,
                "position_y": 0.065,
                "width": 0.66,
                "height": 0.065,
                "title_font_size": 13,
                "label_font_size": 11,
                "color": _BLENDER_PLOT_TEXT,
                "fmt": "%.2f",
            },
        )

    def _add_side_actors(self, side_surfaces):
        self.side_surfaces = list(side_surfaces)
        self.side_actors = []
        for side_id, side_surface in enumerate(side_surfaces):
            actor_name = f"presentation_side_{side_id}"
            actor = self.plotter.add_mesh(
                side_surface,
                name=actor_name,
                scalars="Amplitude",
                cmap=self.cmap,
                clim=self.clim,
                opacity=1.0,
                show_edges=self.edges_visible,
                edge_color="#111827",
                line_width=0.45,
                lighting=self.lighting_enabled,
                ambient=0.42 if self.lighting_enabled else 1.0,
                diffuse=0.62 if self.lighting_enabled else 0.0,
                specular=0.08 if self.lighting_enabled else 0.0,
                specular_power=14.0,
                show_scalar_bar=False,
            )
            self.side_actors.append(actor)

    def _build_layer_outline_polydata(self, field):
        field = np.asarray(field, dtype=_VISUAL_FLOAT_DTYPE)
        lod = max(int(self.lod_factor), 1)
        if lod > 1:
            field = field[::lod, ::lod]
            x_grid = self.viewer.X[::lod, ::lod]
            z_grid = self.viewer.Z[::lod, ::lod]
            velocity_model = self.viewer.velocity_model[::lod, ::lod]
        else:
            x_grid = self.viewer.X
            z_grid = self.viewer.Z
            velocity_model = self.viewer.velocity_model

        y_top = self.viewer._scale_wavefield_for_geometry(
            field=field,
            vertical_scale=self.vertical_scale,
            normalize=self.normalize,
        )
        velocity_values = np.asarray(velocity_model, dtype=_VISUAL_FLOAT_DTYPE)
        finite_values = velocity_values[np.isfinite(velocity_values)]
        if finite_values.size == 0:
            return pv.PolyData()

        unique_values = np.unique(np.round(finite_values, decimals=6))
        if len(unique_values) > 1 and len(unique_values) <= 16:
            contour_levels = 0.5 * (unique_values[:-1] + unique_values[1:])
        else:
            contour_levels = np.percentile(finite_values, [20.0, 40.0, 60.0, 80.0])
            contour_levels = np.unique(np.round(contour_levels, decimals=6))

        contour_levels = contour_levels[
            (contour_levels > float(np.nanmin(finite_values)))
            & (contour_levels < float(np.nanmax(finite_values)))
        ]
        if len(contour_levels) == 0:
            return pv.PolyData()

        template_cache = getattr(self, "_layer_contour_template_cache", None)
        if template_cache is None:
            template_cache = {}
            self._layer_contour_template_cache = template_cache
        template = template_cache.get(lod)
        if template is None:
            contour_grid = pv.StructuredGrid(x_grid, np.zeros_like(x_grid, dtype=_VISUAL_FLOAT_DTYPE), z_grid)
            contour_grid["Velocity"] = np.asarray(velocity_model, dtype=_VISUAL_FLOAT_DTYPE).ravel(order="F")
            try:
                contour_lines = contour_grid.contour(isosurfaces=contour_levels, scalars="Velocity")
            except Exception:
                return pv.PolyData()
            if contour_lines.n_points == 0 or contour_lines.lines.size == 0:
                return pv.PolyData()
            template = (
                # Preserve the contour-level precision passed to VTK.  Rounding it
                # down to the visual array dtype changes the interpolated X/Z
                # positions by a few ULPs and makes the cached first frame differ
                # from the pre-cache rendering.
                np.asarray(contour_levels).copy(),
                np.asarray(contour_lines.points).copy(),
                np.asarray(contour_lines.lines, dtype=np.int_).copy(),
            )
            template_cache[lod] = template
        contour_levels, contour_points, contour_cells = template
        x_min = float(np.nanmin(x_grid))
        z_min = float(np.nanmin(z_grid))
        dx = float(abs(x_grid[0, 1] - x_grid[0, 0])) if x_grid.shape[1] > 1 else 1.0
        dz = float(abs(z_grid[1, 0] - z_grid[0, 0])) if z_grid.shape[0] > 1 else 1.0
        x_max = float(np.nanmax(x_grid))
        z_max = float(np.nanmax(z_grid))
        y_offset = max(float(self.vertical_scale) * 0.004, float(dx) * 0.006)
        side_offset = max(min(dx, dz) * 0.30, 1.0e-6)
        thickness = float(self.domain_thickness)

        def sample_y(x_value, z_value):
            col = np.clip((float(x_value) - x_min) / dx, 0.0, y_top.shape[1] - 1.0)
            row = np.clip((float(z_value) - z_min) / dz, 0.0, y_top.shape[0] - 1.0)
            col0 = int(np.floor(col))
            row0 = int(np.floor(row))
            col1 = min(col0 + 1, y_top.shape[1] - 1)
            row1 = min(row0 + 1, y_top.shape[0] - 1)
            tx = col - col0
            tz = row - row0
            y00 = y_top[row0, col0]
            y01 = y_top[row0, col1]
            y10 = y_top[row1, col0]
            y11 = y_top[row1, col1]
            return float(
                (1.0 - tx) * (1.0 - tz) * y00
                + tx * (1.0 - tz) * y01
                + (1.0 - tx) * tz * y10
                + tx * tz * y11
            )

        outline_points = []
        outline_lines = []

        cursor = 0

        def smooth_polyline(coords):
            coords = np.asarray(coords, dtype=_VISUAL_FLOAT_DTYPE)
            if len(coords) < 7:
                return coords

            smoothed = coords.copy()
            for _ in range(3):
                next_coords = smoothed.copy()
                next_coords[1:-1] = (
                    0.25 * smoothed[:-2]
                    + 0.50 * smoothed[1:-1]
                    + 0.25 * smoothed[2:]
                )
                smoothed = next_coords
            return smoothed

        def side_offset_point(point):
            x_value, y_value, z_value = point
            near_x_min = abs(x_value - x_min) <= 1.5 * dx
            near_x_max = abs(x_value - x_max) <= 1.5 * dx
            near_z_min = abs(z_value - z_min) <= 1.5 * dz
            near_z_max = abs(z_value - z_max) <= 1.5 * dz

            if near_x_min:
                x_value = x_min - side_offset
            elif near_x_max:
                x_value = x_max + side_offset
            elif near_z_min:
                z_value = z_min - side_offset
            elif near_z_max:
                z_value = z_max + side_offset

            return (float(x_value), float(y_value), float(z_value))

        while cursor < len(contour_cells):
            n_points = int(contour_cells[cursor])
            point_ids = contour_cells[cursor + 1 : cursor + 1 + n_points]
            cursor += n_points + 1
            if n_points < 2:
                continue

            front_coords = []
            back_coords = []
            for point_id in point_ids:
                x_value = float(contour_points[int(point_id), 0])
                z_value = float(contour_points[int(point_id), 2])
                y_value = sample_y(x_value, z_value)
                front_coords.append((x_value, y_value + y_offset, z_value))
                back_coords.append((x_value, y_value - thickness - y_offset, z_value))

            front_coords = smooth_polyline(front_coords)
            back_coords = smooth_polyline(back_coords)
            front_ids = []
            back_ids = []
            for front_point, back_point in zip(front_coords, back_coords):
                front_ids.append(len(outline_points))
                outline_points.append(tuple(front_point))
                back_ids.append(len(outline_points))
                outline_points.append(tuple(back_point))

            outline_lines.extend([len(front_ids), *front_ids])
            outline_lines.extend([len(back_ids), *back_ids])

            first_point = np.asarray(outline_points[front_ids[0]])
            last_point = np.asarray(outline_points[front_ids[-1]])
            closed_line = np.linalg.norm(first_point[[0, 2]] - last_point[[0, 2]]) < max(dx, dz) * 1.5
            if not closed_line:
                for front_id, back_id in ((front_ids[0], back_ids[0]), (front_ids[-1], back_ids[-1])):
                    side_front = side_offset_point(outline_points[front_id])
                    side_back = side_offset_point(outline_points[back_id])
                    side_front_id = len(outline_points)
                    outline_points.append(side_front)
                    side_back_id = len(outline_points)
                    outline_points.append(side_back)
                    outline_lines.extend([2, front_id, side_front_id])
                    outline_lines.extend([2, side_front_id, side_back_id])
                    outline_lines.extend([2, side_back_id, back_id])

        def add_side_face_interfaces(edge_col, x_value, x_direction):
            x_side = float(x_value + x_direction * side_offset)
            edge_velocity = velocity_model[:, edge_col]
            edge_y = y_top[:, edge_col]
            edge_z = z_grid[:, edge_col]

            for level in contour_levels:
                edge_crossings = np.where(
                    (edge_velocity[:-1] - level) * (edge_velocity[1:] - level) <= 0.0
                )[0]
                for row in edge_crossings:
                    v0 = float(edge_velocity[row])
                    v1 = float(edge_velocity[row + 1])
                    if abs(v1 - v0) < 1.0e-12:
                        continue

                    ratio = float(np.clip((float(level) - v0) / (v1 - v0), 0.0, 1.0))
                    z_value = float((1.0 - ratio) * edge_z[row] + ratio * edge_z[row + 1])
                    y_value = float((1.0 - ratio) * edge_y[row] + ratio * edge_y[row + 1])
                    front_id = len(outline_points)
                    outline_points.append((x_side, y_value + y_offset, z_value))
                    back_id = len(outline_points)
                    outline_points.append((x_side, y_value - thickness - y_offset, z_value))
                    outline_lines.extend([2, front_id, back_id])

        add_side_face_interfaces(0, x_min, -1.0)
        add_side_face_interfaces(-1, x_max, 1.0)

        if not outline_points:
            return pv.PolyData()

        outline_polydata = pv.PolyData(np.asarray(outline_points, dtype=_VISUAL_FLOAT_DTYPE))
        outline_polydata.lines = np.asarray(outline_lines, dtype=np.int_)
        return outline_polydata

    def _add_layer_outline_actor(self, polydata):
        self.layer_outline_polydata = polydata
        self.layer_outline_actor = None
        if not self.layer_outlines_visible or polydata.n_points == 0:
            return

        self.layer_outline_actor = self.plotter.add_mesh(
            polydata,
            name="presentation_layer_outlines",
            color="#4b5560",
            opacity=0.24,
            line_width=0.35,
            render_lines_as_tubes=False,
            lighting=False,
            show_scalar_bar=False,
        )

    def _update_layer_outline_actor(self, field):
        if not self.layer_outlines_visible:
            return

        new_polydata = self._build_layer_outline_polydata(field)
        if self.layer_outline_polydata is None or self.layer_outline_actor is None:
            self._add_layer_outline_actor(new_polydata)
            return

        self.layer_outline_polydata.copy_from(new_polydata)
        self.layer_outline_polydata.Modified()
        if self.layer_outline_actor is not None:
            self.layer_outline_actor.mapper.SetInputData(self.layer_outline_polydata)
            self.layer_outline_actor.mapper.Update()
            self.layer_outline_actor.mapper.Modified()

    def _remove_layer_outline_actor(self):
        try:
            self.plotter.remove_actor("presentation_layer_outlines", reset_camera=False, render=False)
        except Exception:
            pass
        self.layer_outline_actor = None
        self.layer_outline_polydata = None

    def _configure_lighting(self):
        try:
            self.plotter.remove_all_lights()
        except Exception:
            pass

        if not self.lighting_enabled:
            try:
                renderer = self.plotter.renderer
                if hasattr(renderer, "UseShadowsOff"):
                    renderer.UseShadowsOff()
            except Exception:
                pass
            return

        x_mid = 0.5 * (self.viewer.nx - 1) * self.viewer.dx
        z_mid = 0.5 * (self.viewer.nz - 1) * self.viewer.dz
        x_span = (self.viewer.nx - 1) * self.viewer.dx
        z_span = (self.viewer.nz - 1) * self.viewer.dz
        span = max(x_span, z_span)
        focal_point = (x_mid, -0.2 * self.domain_thickness, z_mid)

        light_specs = [
            ((x_mid - 0.9 * x_span, 2.3 * span, z_mid - 0.85 * z_span), 0.62),
            ((x_mid + 0.9 * x_span, 0.85 * span, z_mid + 0.70 * z_span), 0.24),
        ]
        for position, intensity in light_specs:
            try:
                self.plotter.add_light(
                    pv.Light(
                        position=position,
                        focal_point=focal_point,
                        color="white",
                        intensity=float(intensity),
                    )
                )
            except Exception:
                pass

        try:
            renderer = self.plotter.renderer
            if self.shadows_enabled and hasattr(renderer, "UseShadowsOn"):
                renderer.UseShadowsOn()
            elif hasattr(renderer, "UseShadowsOff"):
                renderer.UseShadowsOff()
        except Exception:
            pass

        try:
            self.plotter.enable_eye_dome_lighting()
        except Exception:
            pass

    def _rebuild_presentation_actors(self):
        self.plotter.remove_actor("presentation_wavefield_surface", reset_camera=False, render=False)
        for side_id in range(len(self.side_surfaces)):
            self.plotter.remove_actor(f"presentation_side_{side_id}", reset_camera=False, render=False)
        self._remove_layer_outline_actor()
        self._add_side_actors(self.side_surfaces)
        self._add_wavefield_actor(self.surface, show_scalar_bar=True)
        if self.layer_outlines_visible:
            snapshot_id = int(self.snapshot_ids[self.current_frame])
            self._add_layer_outline_actor(
                self._build_layer_outline_polydata(self.viewer.wavefield_snapshots[snapshot_id])
            )
        self._configure_lighting()
        self.plotter.render()
        self._position_overlays()

    def _copy_surface_data(self, target_surface, source_surface, actor=None):
        target_surface.copy_from(source_surface)
        target_surface.Modified()

        points = target_surface.GetPoints()
        if points is not None:
            points.Modified()
            point_data = points.GetData()
            if point_data is not None:
                point_data.Modified()

        vtk_point_data = target_surface.GetPointData()
        if vtk_point_data is not None:
            vtk_point_data.Modified()
            for array_id in range(vtk_point_data.GetNumberOfArrays()):
                array = vtk_point_data.GetArray(array_id)
                if array is not None:
                    array.Modified()

        if actor is not None:
            actor.mapper.SetInputData(target_surface)
            actor.mapper.Update()
            actor.mapper.Modified()

    def _build_scene(self):
        first_snapshot = int(self.snapshot_ids[0])
        self.surface, side_surfaces = self._build_thickened_surfaces(
            self.viewer.wavefield_snapshots[first_snapshot]
        )

        self.plotter.set_background(_BLENDER_VIEWPORT_BACKGROUND)
        self._add_side_actors(side_surfaces)
        self._add_wavefield_actor(self.surface, show_scalar_bar=True)
        self._add_layer_outline_actor(
            self._build_layer_outline_polydata(self.viewer.wavefield_snapshots[first_snapshot])
        )
        self._configure_lighting()

        if self.show_source_receivers:
            self.viewer._add_source_and_receivers(
                plotter=self.plotter,
                marker_elevation=float(self.vertical_scale) * 0.08,
                point_size=max(int(self.point_size * 0.9), 4),
                show_labels=False,
            )

        if self.bounds_visible:
            self.plotter.show_bounds(
                grid="front",
                location="outer",
                xtitle="X",
                ytitle="Amplitude",
                ztitle="Z",
                font_size=10,
            )
        if self.axes_visible:
            self.plotter.add_axes(xlabel="X", ylabel="Amplitude", zlabel="Z")

        try:
            self.plotter.enable_anti_aliasing()
        except Exception:
            pass

        self._set_presentation_camera()
        if self.orthographic:
            self.plotter.enable_parallel_projection()
        self.plotter.render()

    def _set_presentation_camera(self):
        x_mid = 0.5 * (self.viewer.nx - 1) * self.viewer.dx
        z_mid = 0.5 * (self.viewer.nz - 1) * self.viewer.dz
        x_span = (self.viewer.nx - 1) * self.viewer.dx
        z_span = (self.viewer.nz - 1) * self.viewer.dz
        span = max(x_span, z_span)

        self.plotter.camera_position = [
            (x_mid - 1.78 * x_span, 1.72 * span, z_mid - 0.22 * z_span),
            (x_mid, 0.0, z_mid),
            (0.0, 0.0, -1.0),
        ]
        self.plotter.camera.zoom(1.04)

    def _refresh_snapshot_ids(self):
        n_snapshots = int(self.viewer.wavefield_snapshots.shape[0])
        start = int(np.clip(self.snapshot_start, 0, n_snapshots - 1))
        step = max(int(self.snapshot_step), 1)
        self.snapshot_ids = np.arange(start, n_snapshots, step, dtype=int)
        self.frame_slider.setRange(0, len(self.snapshot_ids) - 1)
        self.frame_slider.setValue(0)

    def _set_sample(self, sample_key):
        if self.sample_loader is None or not sample_key:
            return
        if self.viewer.scene_label == sample_key:
            return

        was_playing = self.timer.isActive()
        if was_playing:
            self.timer.stop()
            self.btn_play.setText("Play")

        new_viewer = self.sample_loader(str(sample_key))
        if new_viewer is None:
            return

        self.viewer = new_viewer
        self.viewer._mesh_cache.clear()
        self.domain_thickness = self._resolve_domain_thickness(self.viewer)
        self._refresh_snapshot_ids()
        self.info_model.setText(str(self.viewer.model_type))
        self.info_shape.setText(f"{self.viewer.nz} x {self.viewer.nx} | {len(self.snapshot_ids)} frames")
        if self.scene_label_widget is not None:
            self.scene_label_widget.setText(str(self.viewer.scene_label))
        self._draw_velocity_overlay()
        self._draw_seismogram_axes()

        try:
            self.plotter.clear()
        except Exception:
            pass

        self.side_surfaces = []
        self.side_actors = []
        self.surface = None
        self.mesh_actor = None
        self.layer_outline_actor = None
        self.layer_outline_polydata = None
        self._layer_contour_template_cache = {}
        self._build_scene()
        self._update_frame(0)
        self._position_overlays()

        if was_playing:
            self.timer.start()
            self.btn_play.setText("Pause")

    def _update_frame(self, frame_id):
        frame_id = int(np.clip(frame_id, 0, len(self.snapshot_ids) - 1))
        self.current_frame = frame_id
        snapshot_id = int(self.snapshot_ids[frame_id])
        field = self.viewer.wavefield_snapshots[snapshot_id]

        cache_key = ("presentation",) + _mesh_cache_key(
            snapshot_id=snapshot_id,
            vertical_scale=self.vertical_scale,
            normalize=self.normalize,
            deform_surface=True,
            normalize_colors=self.normalize_colors,
            lod_factor=self.lod_factor,
        )
        self._update_thickened_surface_arrays(field)
        self._update_layer_outline_actor(field)

        time_value = float(self.viewer.snapshot_times[snapshot_id])
        max_amp = float(np.max(np.abs(field)))
        self.frame_label.setText(
            f"{frame_id + 1}/{len(self.snapshot_ids)}"
        )
        self.info_frame.setText(f"{frame_id + 1}/{len(self.snapshot_ids)} | snapshot {snapshot_id}")
        self.info_time.setText(f"it = {int(self.viewer.snapshot_time_indices[snapshot_id])} | t = {time_value:.5f} s")
        self.info_amp.setText(f"{max_amp:.6e}")
        if self.viewer.seismogram_data is not None:
            for line in self.seismogram_time_lines:
                line.set_xdata([time_value, time_value])
        elif hasattr(self, "summary_time_line"):
            self.summary_time_line.set_xdata([frame_id, frame_id])
        if hasattr(self, "trace_canvas"):
            self.trace_canvas.draw_idle()
        if self.frame_slider.value() != frame_id:
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(frame_id)
            self.frame_slider.blockSignals(False)

        if self.scalar_legend is not None:
            self.scalar_legend.set_scale(self.cmap, self.clim)
        self.plotter.render()
        self.plotter.update()
        self._position_overlays()

    def _connect_events(self):
        self.frame_slider.valueChanged.connect(self._update_frame)
        self.btn_play.clicked.connect(self._toggle_playback)
        self.btn_save_svg.clicked.connect(self._save_svg_snapshot)
        self.speed_spin.valueChanged.connect(self._set_frame_rate)
        self.timer.timeout.connect(self._timer_tick)
        self.sample_combo.currentTextChanged.connect(self._set_sample)
        self.scale_spin.valueChanged.connect(self._set_vertical_scale)
        self.cmap_combo.currentTextChanged.connect(self._set_colormap)
        self.color_min_spin.valueChanged.connect(self._set_color_limits_from_controls)
        self.color_max_spin.valueChanged.connect(self._set_color_limits_from_controls)
        self.chk_bounds.toggled.connect(self._set_bounds_visible)
        self.chk_axes.toggled.connect(self._set_axes_visible)
        self.chk_edges.toggled.connect(self._set_edges_visible)
        self.chk_layer_outlines.toggled.connect(self._set_layer_outlines_visible)
        self.chk_lighting.toggled.connect(self._set_lighting_enabled)
        self.btn_oblique.clicked.connect(self._set_presentation_camera)
        self.btn_top.clicked.connect(self._set_top_camera)
        self.btn_left.clicked.connect(self._set_left_camera)
        self.btn_right.clicked.connect(self._set_right_camera)
        self.btn_fit.clicked.connect(self.plotter.reset_camera)

    def _toggle_playback(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play.setText("Play")
        else:
            self.timer.start()
            self.btn_play.setText("Pause")

    def _timer_tick(self):
        next_frame = (self.current_frame + 1) % len(self.snapshot_ids)
        self._update_frame(next_frame)

    def _set_frame_rate(self, value):
        self.frame_rate = max(float(value), 0.2)
        self.timer.setInterval(max(int(1000 / self.frame_rate), 1))

    def _set_vertical_scale(self, value):
        self.vertical_scale = float(value)
        self.viewer._mesh_cache.clear()
        self._update_frame(self.current_frame)

    def _set_colormap(self, cmap):
        self.cmap = str(cmap)
        if self.scalar_legend is not None:
            self.scalar_legend.set_scale(self.cmap, self.clim)
        self._rebuild_presentation_actors()

    def _update_scalar_range(self):
        if self.mesh_actor is not None:
            self.mesh_actor.mapper.SetScalarRange(float(self.clim[0]), float(self.clim[1]))
            self.mesh_actor.mapper.Modified()

        for side_actor in self.side_actors:
            side_actor.mapper.SetScalarRange(float(self.clim[0]), float(self.clim[1]))
            side_actor.mapper.Modified()

        try:
            self.plotter.update_scalar_bar_range(self.clim, name="Amplitude")
        except Exception:
            try:
                self.plotter.update_scalar_bar_range(self.clim)
            except Exception:
                pass

        if self.scalar_legend is not None:
            self.scalar_legend.set_scale(self.cmap, self.clim)
        self.plotter.render()
        self.plotter.update()
        self._position_overlays()

    def _set_color_limits_from_controls(self):
        color_min = float(self.color_min_spin.value())
        color_max = float(self.color_max_spin.value())
        if color_min >= color_max:
            return

        self.clim = (color_min, color_max)
        self._update_scalar_range()

    def _safe_filename_token(self, value):
        token = str(value).strip().lower()
        safe_chars = []
        for char in token:
            if char.isalnum():
                safe_chars.append(char)
            elif char in (" ", "-", "_", "."):
                safe_chars.append("_")
        token = "".join(safe_chars).strip("_")
        while "__" in token:
            token = token.replace("__", "_")
        return token or "unknown"

    def _next_export_path(self):
        export_dir = Path(self.snapshot_export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        snapshot_id = int(self.snapshot_ids[self.current_frame])
        time_value = float(self.viewer.snapshot_times[snapshot_id])
        scene = self._safe_filename_token(self.viewer.scene_label or "simulation")
        model = self._safe_filename_token(self.viewer.model_type)
        cmap = self._safe_filename_token(self.cmap)
        clim_min = self._safe_filename_token(f"{float(self.clim[0]):.3f}")
        clim_max = self._safe_filename_token(f"{float(self.clim[1]):.3f}")
        base_name = (
            f"{scene}_{model}_{self.viewer.nz}x{self.viewer.nx}_"
            f"frame_{self.current_frame + 1:04d}_snapshot_{snapshot_id:04d}_"
            f"t_{time_value:.5f}s_cmap_{cmap}_clim_{clim_min}_to_{clim_max}"
        )
        export_path = export_dir / f"{base_name}.svg"
        duplicate_id = 2
        while export_path.exists():
            export_path = export_dir / f"{base_name}_{duplicate_id:02d}.svg"
            duplicate_id += 1
        return export_path

    def _save_svg_snapshot(self):
        if self.snapshot_export_dir is None:
            return

        export_path = self._next_export_path()
        previous_size = tuple(int(value) for value in self.plotter.window_size)
        minimum_svg_size = (1488, 940)
        scale = max(
            1.0,
            minimum_svg_size[0] / max(previous_size[0], 1),
            minimum_svg_size[1] / max(previous_size[1], 1),
        )
        export_size = tuple(
            max(int(round(value * scale)), minimum)
            for value, minimum in zip(previous_size, minimum_svg_size)
        )
        try:
            # GL2PS embeds the current VTK framebuffer in the SVG.  Export from
            # a temporary publication-sized framebuffer, then restore the UI.
            self.plotter.window_size = export_size
            self.plotter.render()
            self.plotter.save_graphic(str(export_path))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "SVG export failed",
                f"Could not save SVG snapshot:\n{export_path}\n\n{exc}",
            )
            return
        finally:
            self.plotter.window_size = previous_size
            self.plotter.render()
            self._position_overlays()

        self.btn_save_svg.setToolTip(str(export_path))
        main_window = self.window()
        if hasattr(main_window, "statusBar"):
            main_window.statusBar().showMessage(
                f"Saved high-resolution SVG snapshot: {export_path}",
                8000,
            )

    def _set_bounds_visible(self, visible):
        self.bounds_visible = bool(visible)
        try:
            self.plotter.remove_bounds_axes()
        except Exception:
            pass
        if self.bounds_visible:
            self.plotter.show_bounds(
                grid="front",
                location="outer",
                xtitle="X",
                ytitle="Amplitude",
                ztitle="Z",
                font_size=10,
            )
        self.plotter.render()

    def _set_axes_visible(self, visible):
        self.axes_visible = bool(visible)
        if self.axes_visible:
            self.plotter.add_axes(xlabel="X", ylabel="Amplitude", zlabel="Z")
        else:
            try:
                self.plotter.hide_axes()
            except Exception:
                pass
        self.plotter.render()

    def _set_edges_visible(self, visible):
        self.edges_visible = bool(visible)
        self._rebuild_presentation_actors()

    def _set_layer_outlines_visible(self, visible):
        self.layer_outlines_visible = bool(visible)
        if self.layer_outlines_visible:
            snapshot_id = int(self.snapshot_ids[self.current_frame])
            self._add_layer_outline_actor(
                self._build_layer_outline_polydata(self.viewer.wavefield_snapshots[snapshot_id])
            )
        else:
            self._remove_layer_outline_actor()
        self.plotter.render()

    def _set_lighting_enabled(self, visible):
        self.lighting_enabled = bool(visible)
        self._rebuild_presentation_actors()

    def _set_top_camera(self):
        x_mid = 0.5 * (self.viewer.nx - 1) * self.viewer.dx
        z_mid = 0.5 * (self.viewer.nz - 1) * self.viewer.dz
        span = max((self.viewer.nx - 1) * self.viewer.dx, (self.viewer.nz - 1) * self.viewer.dz)
        self.plotter.camera_position = [
            (x_mid, 2.2 * span, z_mid),
            (x_mid, 0.0, z_mid),
            (0.0, 0.0, -1.0),
        ]
        self.plotter.render()

    def _set_left_camera(self):
        x_mid = 0.5 * (self.viewer.nx - 1) * self.viewer.dx
        z_mid = 0.5 * (self.viewer.nz - 1) * self.viewer.dz
        span = max((self.viewer.nx - 1) * self.viewer.dx, (self.viewer.nz - 1) * self.viewer.dz)
        self.plotter.camera_position = [
            (x_mid - 2.0 * span, 0.5 * span, z_mid),
            (x_mid, 0.0, z_mid),
            (0.0, 0.0, -1.0),
        ]
        self.plotter.render()

    def _set_right_camera(self):
        x_mid = 0.5 * (self.viewer.nx - 1) * self.viewer.dx
        z_mid = 0.5 * (self.viewer.nz - 1) * self.viewer.dz
        span = max((self.viewer.nx - 1) * self.viewer.dx, (self.viewer.nz - 1) * self.viewer.dz)
        self.plotter.camera_position = [
            (x_mid + 2.0 * span, 0.5 * span, z_mid),
            (x_mid, 0.0, z_mid),
            (0.0, 0.0, -1.0),
        ]
        self.plotter.render()


class _WavePropagationQtWindow(QtWidgets.QMainWindow):
    """
    QMainWindow containing a PyVista render area and a control panel.
    """

    def __init__(
        self,
        viewer,
        start_snapshot,
        end_snapshot,
        step,
        vertical_scale,
        normalize,
        deform_surface,
        normalize_colors,
        cmap,
        clim,
        show_source_receivers,
        show_labels,
        show_edges,
        point_size,
        frame_rate,
        mesh_update_mode,
        orthographic,
        title,
        comparison_viewers,
        presentation_sample_keys=None,
        presentation_sample_loader=None,
        snapshot_export_dir=None,
        presentation_lighting_enabled=True,
        presentation_shadows_enabled=True,
        presentation_grid_visible=True,
        presentation_mesh_visible=True,
        presentation_layer_outlines_visible=True,
        presentation_domain_thickness=None,
        lod_factor=1,
    ):
        super().__init__()

        self.viewer = viewer
        self.vertical_scale = float(vertical_scale)
        self.normalize = bool(normalize)
        self.deform_surface = bool(deform_surface)
        self.normalize_colors = bool(normalize_colors)
        self.cmap = cmap
        self.clim = clim
        self.show_source_receivers = bool(show_source_receivers)
        self.show_labels = bool(show_labels)
        self.show_edges = bool(show_edges)
        self.point_size = int(point_size)
        self.frame_rate = max(float(frame_rate), 1.0)
        self.mesh_update_mode = str(mesh_update_mode).lower()
        self.initial_orthographic = bool(orthographic)
        self.current_frame = 0
        self.lod_factor = max(int(lod_factor), 1)
        #--------------------------------------------------------
        self.comparison_viewers = [] if comparison_viewers is None else list(comparison_viewers)  # Store optional viewers used by the two-simulation comparison tab
        self.comparison_snapshot_step = max(int(step), 1)                                         # Preserve the selected snapshot stride for the comparison tab
        self.presentation_sample_keys = [] if presentation_sample_keys is None else list(presentation_sample_keys)
        self.presentation_sample_loader = presentation_sample_loader
        self.snapshot_export_dir = snapshot_export_dir
        self.presentation_lighting_enabled = bool(presentation_lighting_enabled)
        self.presentation_shadows_enabled = bool(presentation_shadows_enabled)
        self.presentation_grid_visible = bool(presentation_grid_visible)
        self.presentation_mesh_visible = bool(presentation_mesh_visible)
        self.presentation_layer_outlines_visible = bool(presentation_layer_outlines_visible)
        self.presentation_domain_thickness = presentation_domain_thickness
        #--------------------------------------------------------

        if self.mesh_update_mode not in ("replace", "inplace"):
            raise ValueError("mesh_update_mode must be 'replace' or 'inplace'.")

        self.snapshot_ids = self._select_snapshot_ids(
            start_snapshot=start_snapshot,
            end_snapshot=end_snapshot,
            step=step,
        )
        self.clim = self._resolve_color_limits(self.clim)
        self.amplitude_summary = np.max(
            np.abs(self.viewer.wavefield_snapshots[self.snapshot_ids]),
            axis=(1, 2),
        )

        self.surface = None
        self.mesh_actor = None
        self.axes_actor = None
        self.time_line = None
        self.seismogram_time_lines = []
        self.scene_label_widget = None
        #--------------------------------------------------------
        self.velocity_overlay_cmap = "viridis"                                                     # Use viridis as the default colormap for the single-view velocity overlay
        self.velocity_overlay_canvas = None
        self.scalar_legend = None                                                        # Store the single-view velocity overlay canvas when it is created
        #--------------------------------------------------------

        self.setWindowTitle(title)
        self.setMinimumSize(1280, 760)
        self._apply_application_style()
        self.statusBar().showMessage(
            f"{self.viewer.model_type} | {self.viewer.nz} x {self.viewer.nx} grid | "
            f"{len(self.snapshot_ids)} frames | LOD {self.lod_factor}"
        )
        self._build_ui()

    def resizeEvent(self, event):
        """
        Keep the Qt scene label anchored to the render area's upper-right corner.
        """
        super().resizeEvent(event)
        self._position_scene_label()

    def _position_scene_label(self):
        """
        Position the high-DPI Qt label over the PyVista render widget.
        """
        if self.scene_label_widget is None:
            return

        margin = 14
        self.scene_label_widget.adjustSize()
        label_size = self.scene_label_widget.size()
        container_rect = self.render_container.rect()

        x_pos = max(container_rect.width() - label_size.width() - margin, margin)
        y_pos = margin

        self.scene_label_widget.move(x_pos, y_pos)
        self.scene_label_widget.raise_()
        #--------------------------------------------------------
        if self.velocity_overlay_canvas is not None:                                                # Keep the single-view velocity overlay above the PyVista renderer
            self.velocity_overlay_canvas.raise_()                                                   # Raise the static velocity overlay after label positioning
        #--------------------------------------------------------

    def _build_single_velocity_overlay(self, parent_widget):                                        # Build the static 2D velocity overlay for the single-simulation tab
        #--------------------------------------------------------
        overlay_figure = Figure(                                                                   # Create a small Matplotlib figure for the 2D velocity model
            figsize=(2.15, 1.55),                                                                  # Use a slightly larger overlay in the single-simulation view
            dpi=100,                                                                               # Use a stable pixel density for the overlay
            facecolor=_BLENDER_PLOT_BACKGROUND,                                                                  # Use a white figure background for readability
            tight_layout=True,                                                                     # Reduce unused margins around the velocity image
        )
        overlay_canvas = FigureCanvas(overlay_figure)                                              # Embed the velocity figure as a Qt canvas
        overlay_canvas.setParent(parent_widget)                                                    # Attach the overlay directly to the PyVista render widget
        overlay_canvas.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)                  # Let mouse events pass through to the PyVista interactor
        overlay_canvas.setGeometry(16, 16, 215, 155)                                               # Place the overlay in the upper-left corner of the single-view viewport
        overlay_canvas.setStyleSheet("background-color: #28282B; border: 1px solid #55555A;")        # Add a thin border so the overlay stays visible

        overlay_axis = overlay_figure.add_subplot(111)                                             # Create the 2D velocity-model axis
        overlay_axis.imshow(                                                                       # Draw the velocity model as a static 2D image
            self.viewer.velocity_model,                                                            # Use the single-view velocity model
            cmap=self.velocity_overlay_cmap,                                                       # Use viridis by default for the 2D velocity overlay
            origin="upper",                                                                       # Keep array orientation consistent with the stored model
            aspect="auto",                                                                        # Fill the compact overlay without changing its fixed size
        )
        overlay_axis.set_title("Velocity", fontsize=8, fontweight="bold", pad=1, color=_BLENDER_PLOT_TEXT)                  # Add a short label so the inset is self-explanatory
        overlay_axis.set_xticks([])                                                               # Hide x ticks to keep the overlay compact
        overlay_axis.set_yticks([])                                                               # Hide z ticks to keep the overlay compact
        for spine in overlay_axis.spines.values():                                                 # Keep a visible frame around the velocity image
            spine.set_linewidth(0.7)                                                              # Use a thin axis frame
            spine.set_color("#77777C")                                                           # Use a dark frame for contrast
        overlay_canvas.draw_idle()                                                                # Draw the static velocity overlay once
        overlay_canvas.raise_()                                                                   # Keep the overlay above the PyVista renderer
        return overlay_canvas                                                                     # Return the overlay canvas for later raise calls
        #--------------------------------------------------------

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------
    def _select_snapshot_ids(self, start_snapshot, end_snapshot, step):
        n_snapshots = self.viewer.wavefield_snapshots.shape[0]
        start_snapshot = int(start_snapshot)
        end_snapshot = n_snapshots - 1 if end_snapshot is None else int(end_snapshot)
        step = int(step)

        if start_snapshot < 0 or start_snapshot >= n_snapshots:
            raise IndexError(
                f"start_snapshot={start_snapshot} is out of range [0, {n_snapshots - 1}]."
            )

        if end_snapshot < 0 or end_snapshot >= n_snapshots:
            raise IndexError(
                f"end_snapshot={end_snapshot} is out of range [0, {n_snapshots - 1}]."
            )

        if end_snapshot < start_snapshot:
            raise ValueError("end_snapshot must be greater than or equal to start_snapshot.")

        if step <= 0:
            raise ValueError("step must be a positive integer.")

        return np.arange(start_snapshot, end_snapshot + 1, step, dtype=int)

    def _resolve_color_limits(self, clim):
        if clim is not None:
            return clim

        if self.normalize_colors:
            return (-1.0, 1.0)

        values = self.viewer.wavefield_snapshots[self.snapshot_ids].ravel()
        vmin = float(np.percentile(values, 2.0))
        vmax = float(np.percentile(values, 98.0))
        abs_max = max(abs(vmin), abs(vmax))

        if abs_max < 1e-20:
            return (-1.0, 1.0)

        return (-abs_max, abs_max)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _apply_application_style(self):
        self.setStyleSheet(
            """
            * { font-family: "Segoe UI"; font-size: 10px; }
            QMainWindow, QWidget#centralRoot { background: #1C1C1E; }
            QFrame#topBar { background: #262629; border-bottom: 1px solid #111113; min-height: 28px; }
            QToolButton#menuButton { color: #D5D5D7; background: transparent; border: 0; padding: 2px 7px; min-height: 22px; }
            QToolButton#menuButton:hover { background: #3B3B3E; }
            QToolButton#workspaceButton { color: #BDBDC0; background: transparent; border: 0; border-bottom: 2px solid transparent; padding: 2px 9px; min-height: 22px; }
            QToolButton#workspaceButton:checked { color: #F1F1F2; border-bottom-color: #E58A2B; background: #303033; }
            QFrame#toolbarSeparator { color: #4B4B4F; max-width: 1px; margin: 5px 5px; }
            QFrame#sidePanel { background: #28282B; border-left: 1px solid #161618; }
            QFrame#outliner { background: #2C2C2F; border: 1px solid #1A1A1C; }
            QFrame#viewportToolbar, QFrame#timelineToolbar { background: #303033; min-height: 26px; border-top: 1px solid #161618; border-bottom: 1px solid #151517; }
            QLabel#editorType { color: #E4E4E6; font-weight: 700; padding: 0 6px; }
            QLabel#editorMode { color: #BDBDC0; background: #3A3A3D; border: 1px solid #505055; padding: 2px 8px; }
            QToolButton#editorMenuButton { color: #D0D0D2; background: transparent; border: 0; padding: 2px 7px; min-height: 22px; }
            QToolButton#editorMenuButton:hover { background: #454549; }
            QLabel { color: #D5D5D5; } QLabel#applicationName { color: #E58A2B; font-weight: 700; } QLabel#crumb, QLabel#sceneItem { color: #E3E3E3; font-weight: 600; } QLabel#muted, QLabel#footerLabel { color: #858589; font-size: 9px; }
            QTabWidget::pane { border: 0; background: #28282B; } QTabBar::tab { color: #BDBDC0; background: #222224; border: 0; padding: 7px 13px; } QTabBar::tab:selected { color: #EEEEF0; background: #343437; border-bottom: 2px solid #E58A2B; }
            QGroupBox { color: #BDBDC0; font-weight: 600; border: 0; border-top: 1px solid #414145; margin-top: 15px; padding-top: 9px; } QGroupBox::title { subcontrol-origin: margin; left: 0; top: -2px; padding-right: 7px; background: #28282B; }
            QCheckBox { color: #CFCFD1; spacing: 7px; padding: 2px 0; } QCheckBox::indicator { width: 12px; height: 12px; border: 1px solid #68686C; background: #323235; } QCheckBox::indicator:checked { background: #E58A2B; border-color: #F0A15A; }
            QPushButton { color: #D5D5D7; background: #363639; border: 1px solid #4B4B4F; border-radius: 2px; min-height: 25px; padding: 2px 8px; } QPushButton:hover { background: #414145; border-color: #626267; }
            QComboBox, QDoubleSpinBox { color: #D5D5D7; background: #333336; border: 1px solid #4B4B4F; border-radius: 2px; min-height: 23px; padding: 1px 7px; } QComboBox QAbstractItemView { color: #D8D8DA; background: #303033; selection-background-color: #E58A2B; selection-color: #171719; }
            QSlider::groove:horizontal { height: 3px; background: #48484C; } QSlider::sub-page:horizontal { background: #D77C2B; } QSlider::handle:horizontal { background: #F0A15A; border: 1px solid #A95E1D; width: 10px; margin: -4px 0; }
            QStatusBar { background: #202022; color: #858589; border-top: 1px solid #111113; }
            QFrame#viewportEditor, QFrame#traceEditor { background: #242426; border: 1px solid #161618; }
            QLabel#editorHeader { background: #232325; color: #BDBDC0; min-height: 24px; padding-left: 9px; font-size: 9px; font-weight: 700; border-bottom: 1px solid #404044; }
            QGroupBox#seismogramEditor { background: #28282B; border: 1px solid #3E3E42; margin-top: 15px; padding: 9px 6px 6px 6px; }
            QWidget#render_container { background: #2B2B2D; border: 0; }
            QSplitter::handle { background: #1A1A1C; }
            QSplitter::handle:hover { background: #E58A2B; }
            QSplitter::handle:horizontal { width: 4px; }
            QSplitter::handle:vertical { height: 4px; }
            """
        )
    def _sync_workspace_buttons(self, index):
        """Keep the Blender-like workspace buttons aligned with the real tab."""
        for button_index, button in enumerate(getattr(self, "_workspace_buttons", [])):
            button.setChecked(button_index == int(index))

    def _build_top_bar(self):
        """Build a compact toolbar whose every visible command is functional."""
        bar = QtWidgets.QFrame()
        bar.setObjectName("topBar")
        layout = QtWidgets.QHBoxLayout(bar)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(3)

        brand = QtWidgets.QLabel("AI", bar)
        brand.setObjectName("applicationName")
        brand.setToolTip("AI Surface Seismogram")
        layout.addWidget(brand)

        file_button = QtWidgets.QToolButton(bar)
        file_button.setText("File")
        file_button.setObjectName("menuButton")
        file_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        file_menu = QtWidgets.QMenu(file_button)
        save_action = file_menu.addAction("Save SVG snapshot")
        save_action.setEnabled(self.presentation_widget.snapshot_export_dir is not None)
        save_action.triggered.connect(self.presentation_widget._save_svg_snapshot)
        file_menu.addSeparator()
        close_action = file_menu.addAction("Close viewer")
        close_action.triggered.connect(self.close)
        file_button.setMenu(file_menu)
        layout.addWidget(file_button)

        view_button = QtWidgets.QToolButton(bar)
        view_button.setText("View")
        view_button.setObjectName("menuButton")
        view_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        view_menu = QtWidgets.QMenu(view_button)
        for label, callback in (
            ("Oblique", self.presentation_widget._set_presentation_camera),
            ("Top", self.presentation_widget._set_top_camera),
            ("Left", self.presentation_widget._set_left_camera),
            ("Right", self.presentation_widget._set_right_camera),
            ("Frame all", lambda: self.presentation_widget.plotter.reset_camera()),
        ):
            action = view_menu.addAction(label)
            action.triggered.connect(callback)
        view_button.setMenu(view_menu)
        layout.addWidget(view_button)

        playback_button = QtWidgets.QToolButton(bar)
        playback_button.setText("Playback")
        playback_button.setObjectName("menuButton")
        playback_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        playback_menu = QtWidgets.QMenu(playback_button)
        toggle_action = playback_menu.addAction("Play / Pause")
        toggle_action.triggered.connect(self.presentation_widget._toggle_playback)
        fit_action = playback_menu.addAction("Frame current model")
        fit_action.triggered.connect(lambda: self.presentation_widget.plotter.reset_camera())
        playback_button.setMenu(playback_menu)
        layout.addWidget(playback_button)

        help_button = QtWidgets.QToolButton(bar)
        help_button.setText("Help")
        help_button.setObjectName("menuButton")
        help_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        help_menu = QtWidgets.QMenu(help_button)
        controls_action = help_menu.addAction("Viewport controls")
        controls_action.triggered.connect(
            lambda: QtWidgets.QMessageBox.information(
                self,
                "Viewport controls",
                "Use the PyVista mouse controls configured in the viewport.\n\nView provides named cameras; Playback controls frame animation.",
            )
        )
        help_button.setMenu(help_menu)
        layout.addWidget(help_button)

        separator = QtWidgets.QFrame(bar)
        separator.setFrameShape(QtWidgets.QFrame.VLine)
        separator.setObjectName("toolbarSeparator")
        layout.addWidget(separator)

        workspace_group = QtWidgets.QButtonGroup(bar)
        workspace_group.setExclusive(True)
        self._workspace_buttons = []
        for index, title in enumerate(("Layout", "Comparison") if self.comparison_viewers else ("Layout",)):
            workspace = QtWidgets.QToolButton(bar)
            workspace.setText(title)
            workspace.setCheckable(True)
            workspace.setObjectName("workspaceButton")
            workspace.setToolTip(f"Open {title} workspace")
            workspace.clicked.connect(lambda checked, target=index: self.tab_widget.setCurrentIndex(target))
            workspace_group.addButton(workspace, index)
            self._workspace_buttons.append(workspace)
            layout.addWidget(workspace)
        self._sync_workspace_buttons(self.tab_widget.currentIndex())
        self.tab_widget.currentChanged.connect(self._sync_workspace_buttons)

        layout.addStretch(1)
        scene = QtWidgets.QLabel("Scene  /  Wavefield", bar)
        scene.setObjectName("crumb")
        layout.addWidget(scene)
        close_button = QtWidgets.QToolButton(bar)
        close_button.setText("X")
        close_button.setToolTip("Close viewer")
        close_button.setObjectName("menuButton")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)
        return bar

    def _build_ui(self):
        central = QtWidgets.QWidget(self)
        central_layout = QtWidgets.QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        self.tab_widget = QtWidgets.QTabWidget(central)
        central_layout.addWidget(self.tab_widget)

        self.presentation_widget = _PresentationDomainWidget(
            viewer=self.viewer,
            snapshot_ids=self.snapshot_ids,
            vertical_scale=self.vertical_scale,
            normalize=self.normalize,
            normalize_colors=self.normalize_colors,
            cmap=self.cmap,
            clim=self.clim,
            show_source_receivers=self.show_source_receivers,
            point_size=self.point_size,
            frame_rate=self.frame_rate,
            orthographic=self.initial_orthographic,
            lod_factor=self.lod_factor,
            sample_keys=self.presentation_sample_keys,
            sample_loader=self.presentation_sample_loader,
            snapshot_export_dir=self.snapshot_export_dir,
            lighting_enabled=self.presentation_lighting_enabled,
            shadows_enabled=self.presentation_shadows_enabled,
            grid_visible=self.presentation_grid_visible,
            mesh_visible=self.presentation_mesh_visible,
            layer_outlines_visible=self.presentation_layer_outlines_visible,
            domain_thickness=self.presentation_domain_thickness,
            parent=self.tab_widget,
        )
        self.tab_widget.addTab(self.presentation_widget, "Single Simulation")

        if self.comparison_viewers:
            self.comparison_widget = _MultiSimulationComparisonWidget(
                comparison_viewers=self.comparison_viewers,
                start_snapshot=int(self.snapshot_ids[0]),
                end_snapshot=int(self.snapshot_ids[-1]),
                step=self.comparison_snapshot_step,
                vertical_scale=self.vertical_scale,
                normalize=self.normalize,
                deform_surface=self.deform_surface,
                normalize_colors=self.normalize_colors,
                cmap=self.cmap,
                clim=self.clim,
                show_source_receivers=self.show_source_receivers,
                show_labels=False,
                show_edges=self.show_edges,
                point_size=max(int(self.point_size * 0.55), 4),
                frame_rate=self.frame_rate,
                mesh_update_mode=self.mesh_update_mode,
                orthographic=self.initial_orthographic,
                lod_factor=self.lod_factor,
                grid_visible=self.presentation_grid_visible,
                mesh_visible=self.presentation_mesh_visible,
                lighting_enabled=self.presentation_lighting_enabled,
                shadows_enabled=self.presentation_shadows_enabled,
                layer_outlines_visible=self.presentation_layer_outlines_visible,
                domain_thickness=self.presentation_domain_thickness,
                parent=self.tab_widget,
            )
            self.tab_widget.addTab(self.comparison_widget, "2 Simulation Comparison")

        central_layout.insertWidget(0, self._build_top_bar())
        self.setCentralWidget(central)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(max(int(1000 / self.frame_rate), 1))

    def _build_info_group(self):
        group = QtWidgets.QGroupBox("Simulation")
        layout = QtWidgets.QFormLayout(group)

        self.info_model = QtWidgets.QLabel(str(self.viewer.model_type))
        self.info_frame = QtWidgets.QLabel("-")
        self.info_time = QtWidgets.QLabel("-")
        self.info_amp = QtWidgets.QLabel("-")
        self.info_shape = QtWidgets.QLabel(
            f"{self.viewer.nz} x {self.viewer.nx} | {len(self.snapshot_ids)} frames"
        )

        layout.addRow("Model", self.info_model)
        layout.addRow("Grid", self.info_shape)
        layout.addRow("Frame", self.info_frame)
        layout.addRow("Time", self.info_time)
        layout.addRow("Max |P|", self.info_amp)

        return group

    def _build_camera_group(self):
        group = QtWidgets.QGroupBox("Camera")
        layout = QtWidgets.QGridLayout(group)

        self.btn_oblique = QtWidgets.QPushButton("Oblique")
        self.btn_top = QtWidgets.QPushButton("Top")
        self.btn_front = QtWidgets.QPushButton("Front")
        self.btn_right = QtWidgets.QPushButton("Right")
        self.btn_fit = QtWidgets.QPushButton("Fit")
        self.chk_orthographic = QtWidgets.QCheckBox("Orthographic view")
        self.chk_orthographic.setChecked(self.initial_orthographic)

        layout.addWidget(self.btn_oblique, 0, 0)
        layout.addWidget(self.btn_top, 0, 1)
        layout.addWidget(self.btn_front, 1, 0)
        layout.addWidget(self.btn_right, 1, 1)
        layout.addWidget(self.btn_fit, 2, 0, 1, 2)
        layout.addWidget(self.chk_orthographic, 3, 0, 1, 2)

        return group

    def _build_data_group(self):
        group = QtWidgets.QGroupBox("Display")
        layout = QtWidgets.QFormLayout(group)

        self.chk_deform = QtWidgets.QCheckBox("Deform surface")
        self.chk_deform.setChecked(self.deform_surface)
        self.chk_bounds = QtWidgets.QCheckBox("Show grid and ticks")
        self.chk_bounds.setChecked(True)
        self.chk_axes = QtWidgets.QCheckBox("Show orientation axes")
        self.chk_axes.setChecked(True)

        self.scale_spin = QtWidgets.QDoubleSpinBox()
        self.scale_spin.setRange(0.0, 5000.0)
        self.scale_spin.setDecimals(1)
        self.scale_spin.setSingleStep(50.0)
        self.scale_spin.setValue(self.vertical_scale)

        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(["seismic", "gray", "RdBu_r", "viridis", "turbo", "coolwarm"])
        if self.cmap in [self.cmap_combo.itemText(i) for i in range(self.cmap_combo.count())]:
            self.cmap_combo.setCurrentText(self.cmap)

        self.color_min_spin = QtWidgets.QDoubleSpinBox()
        self.color_min_spin.setRange(-1.0e12, 1.0e12)
        self.color_min_spin.setDecimals(4)
        self.color_min_spin.setSingleStep(0.05)
        self.color_min_spin.setKeyboardTracking(False)
        self.color_min_spin.setValue(float(self.clim[0]))

        self.color_max_spin = QtWidgets.QDoubleSpinBox()
        self.color_max_spin.setRange(-1.0e12, 1.0e12)
        self.color_max_spin.setDecimals(4)
        self.color_max_spin.setSingleStep(0.05)
        self.color_max_spin.setKeyboardTracking(False)
        self.color_max_spin.setValue(float(self.clim[1]))

        color_limit_enabled = self.cmap_combo.currentText() == "RdBu_r"
        self.color_min_spin.setEnabled(color_limit_enabled)
        self.color_max_spin.setEnabled(color_limit_enabled)

        layout.addRow(self.chk_deform)
        layout.addRow(self.chk_bounds)
        layout.addRow(self.chk_axes)
        layout.addRow("Vertical scale", self.scale_spin)
        layout.addRow("Color map", self.cmap_combo)
        layout.addRow("Color min", self.color_min_spin)
        layout.addRow("Color max", self.color_max_spin)

        return group

    def _build_playback_group(self):
        group = QtWidgets.QGroupBox("Playback")
        layout = QtWidgets.QGridLayout(group)

        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setRange(0, len(self.snapshot_ids) - 1)
        self.frame_slider.setValue(0)

        self.btn_prev = QtWidgets.QPushButton("Prev")
        self.btn_play = QtWidgets.QPushButton("Play")
        self.btn_next = QtWidgets.QPushButton("Next")

        self.speed_spin = QtWidgets.QDoubleSpinBox()
        self.speed_spin.setRange(0.2, 60.0)
        self.speed_spin.setDecimals(1)
        self.speed_spin.setSingleStep(1.0)
        self.speed_spin.setValue(self.frame_rate)

        layout.addWidget(self.frame_slider, 0, 0, 1, 3)
        layout.addWidget(self.btn_prev, 1, 0)
        layout.addWidget(self.btn_play, 1, 1)
        layout.addWidget(self.btn_next, 1, 2)
        layout.addWidget(QtWidgets.QLabel("FPS"), 2, 0)
        layout.addWidget(self.speed_spin, 2, 1, 1, 2)

        return group

    def _build_plot_group(self):
        if self.viewer.seismogram_data is not None:
            return self._build_seismogram_group()

        return self._build_amplitude_group()

    def _build_amplitude_group(self):
        group = QtWidgets.QGroupBox("Wavefield Summary")
        layout = QtWidgets.QVBoxLayout(group)

        self.figure = Figure(figsize=(4.2, 2.8), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.plot(
            np.arange(len(self.amplitude_summary)),
            self.amplitude_summary,
            color="#1f77b4",
            linewidth=1.2,
        )
        self.time_line = self.ax.axvline(0, color="#aa1111", linewidth=1.2, alpha=0.8)
        self.ax.set_title(
            "Wavefield Amplitude Timeline",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Max |P|")
        self.ax.grid(True, alpha=0.25)

        layout.addWidget(self.canvas)
        return group

    def _build_seismogram_group(self):
        group = QtWidgets.QGroupBox("Surface Seismograms")
        layout = QtWidgets.QVBoxLayout(group)

        seismogram_data = self.viewer.seismogram_data
        seismograms = seismogram_data["surface_seismograms"]
        time_axis = seismogram_data["time_axis"]
        receiver_x = seismogram_data["receiver_x"]
        receiver_z = seismogram_data["receiver_z"]
        selected_receivers = seismogram_data["selected_receivers"]

        self.figure = Figure(figsize=(4.8, 4.1), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        axes = self.figure.subplots(len(selected_receivers), 1, sharex=True)
        self.seismogram_axes = np.atleast_1d(axes)
        self.seismogram_time_lines = []

        selected_traces = seismograms[selected_receivers]
        abs_limit = float(np.percentile(np.abs(selected_traces), 99.0))
        if abs_limit < 1e-20:
            abs_limit = 1.0
        abs_limit *= 1.15

        for local_id, (ax, receiver_id) in enumerate(zip(self.seismogram_axes, selected_receivers)):
            trace = seismograms[int(receiver_id)]
            _apply_blender_plot_theme(ax)
            ax.plot(
                time_axis,
                trace,
                color="#F0F0F2",
                linewidth=0.85,
                alpha=0.92,
            )
            time_line = ax.axvline(
                float(self.viewer.snapshot_times[int(self.snapshot_ids[0])]),
                color="#F0A15A",
                linewidth=1.0,
                linestyle="--",
                alpha=0.65,
            )
            self.seismogram_time_lines.append(time_line)

            if int(receiver_id) < len(receiver_x) and int(receiver_id) < len(receiver_z):
                receiver_label = (
                    f"ST{int(receiver_id) + 1}  "
                    f"ix={int(receiver_x[int(receiver_id)])}"
                )
            else:
                receiver_label = f"ST{int(receiver_id) + 1}"

            ax.text(
                0.01,
                0.82,
                receiver_label,
                transform=ax.transAxes,
                fontsize=8,
                fontweight="bold",
                color="#F0F0F2",
                ha="left",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "#38383C",
                    "edgecolor": "none",
                    "alpha": 0.72,
                },
            )
            ax.set_ylim(-abs_limit, abs_limit)
            ax.set_ylabel("Amp.", fontsize=8)
            ax.grid(True, axis="x", linestyle="--", alpha=0.28)
            ax.grid(True, axis="y", linestyle=":", alpha=0.16)
            ax.tick_params(axis="both", labelsize=8, width=0.7, length=3)
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)

        self.seismogram_axes[0].set_title(
            "Surface Seismograms",
            fontsize=11,
            fontweight="bold",
            color="#F0F0F2",
            pad=8,
        )
        self.seismogram_axes[-1].set_xlabel("Time [s]", fontsize=9)

        layout.addWidget(self.canvas)
        return group

    # ------------------------------------------------------------------
    # Scene construction and updates
    # ------------------------------------------------------------------
    def _build_scene(self):
        first_snapshot = int(self.snapshot_ids[0])
        self.surface = self.viewer._build_surface_mesh(
            field=self.viewer.wavefield_snapshots[first_snapshot],
            vertical_scale=self.vertical_scale,
            normalize=self.normalize,
            deform_surface=self.deform_surface,
            normalize_colors=self.normalize_colors,
            lod_factor=self.lod_factor,
        )

        self._add_wavefield_actor(show_scalar_bar=True)

        if self.show_source_receivers:
            self.viewer._add_source_and_receivers(
                plotter=self.plotter,
                marker_elevation=0.0,
                point_size=self.point_size,
                show_labels=self.show_labels,
            )

        self._set_orientation_axes_visible(True)
        self._set_bounds_visible(True)

        try:
            self.plotter.enable_anti_aliasing()
            self.plotter.enable_eye_dome_lighting()
        except Exception:
            pass

        self.plotter.background_color = (0.95, 0.95, 0.95)

        try:
            self.plotter.add_light(
                pv.Light(
                    position=(1.0, 1.0, 2.0),
                    focal_point=(0.0, 0.0, 0.0),
                    color=(1.0, 1.0, 1.0),
                    intensity=0.8,
                )
            )
            self.plotter.add_light(
                pv.Light(
                    position=(-1.0, -1.0, 1.5),
                    focal_point=(0.0, 0.0, 0.0),
                    color=(0.8, 0.8, 0.85),
                    intensity=0.4,
                )
            )
        except Exception:
            pass

        self._set_oblique_camera()
        self._set_orthographic_projection(self.initial_orthographic)

    def _set_orientation_axes_visible(self, visible):
        """
        Show or hide the small orientation axes in the render window.
        """
        if visible:
            if self.axes_actor is None:
                self.axes_actor = self.plotter.add_axes(
                    xlabel="X",
                    ylabel="Amplitude",
                    zlabel="Z",
                )
        else:
            try:
                self.plotter.hide_axes()
            except Exception:
                pass
            self.axes_actor = None

        self.plotter.render()

    def _set_bounds_visible(self, visible):
        """
        Show or hide the outer bounds, grid lines, axis labels, and tick labels.
        """
        try:
            self.plotter.remove_bounds_axes()
        except Exception:
            pass

        if visible:
            self.plotter.show_bounds(
                grid="front",
                location="outer",
                xtitle="X",
                ytitle="Amplitude",
                ztitle="Z",
                font_size=10,
            )

        self.plotter.render()

    def _add_wavefield_actor(self, show_scalar_bar=True):
        """
        Add the wavefield mesh actor using the current surface and color settings.
        """
        self.mesh_actor = self.plotter.add_mesh(
            self.surface,
            name="wavefield_surface",
            scalars="Amplitude",
            cmap=self.cmap,
            clim=self.clim,
            show_edges=self.show_edges,
            smooth_shading=True,
            show_scalar_bar=False,
            scalar_bar_args={
                "title": "Amplitude",
                "vertical": False,
                "position_x": 0.23,
                "position_y": 0.055,
                "width": 0.52,
                "height": 0.065,
                "title_font_size": 10,
                "label_font_size": 9,
                "color": _BLENDER_PLOT_TEXT,
                "fmt": "%.2f",
            },
        )

    def _replace_surface_data(self, new_surface):
        """
        Refresh the rendered wavefield surface.

        The Qt/VTK stack on Windows can keep showing an old GPU buffer when only
        point arrays are modified in place. The default replacement path is more
        reliable for live playback; the in-place path remains available for
        faster machines/drivers where it works correctly.
        """
        if self.mesh_update_mode == "replace":
            self.surface = new_surface
            self.plotter.remove_actor("wavefield_surface", reset_camera=False, render=False)
            self._add_wavefield_actor(show_scalar_bar=False)
            return

        self.surface.copy_from(new_surface)
        self.surface.Modified()

        points = self.surface.GetPoints()
        if points is not None:
            points.Modified()
            point_data = points.GetData()
            if point_data is not None:
                point_data.Modified()

        vtk_point_data = self.surface.GetPointData()
        if vtk_point_data is not None:
            vtk_point_data.Modified()
            for name in ("Amplitude", "RawAmplitude", "Velocity"):
                array = vtk_point_data.GetArray(name)
                if array is not None:
                    array.Modified()

        if self.mesh_actor is not None:
            self.mesh_actor.mapper.SetInputData(self.surface)
            self.mesh_actor.mapper.Update()
            self.mesh_actor.mapper.Modified()

    def _update_frame(self, frame_id):
        frame_id = int(np.clip(frame_id, 0, len(self.snapshot_ids) - 1))
        self.current_frame = frame_id
        snapshot_id = int(self.snapshot_ids[frame_id])
        field = self.viewer.wavefield_snapshots[snapshot_id]

        cache_key = _mesh_cache_key(
            snapshot_id=snapshot_id,
            vertical_scale=self.vertical_scale,
            normalize=self.normalize,
            deform_surface=self.deform_surface,
            normalize_colors=self.normalize_colors,
            lod_factor=self.lod_factor,
        )
        new_surface = self.viewer._mesh_cache.get(cache_key)
        if new_surface is None:
            new_surface = self.viewer._build_surface_mesh(
                field=field,
                vertical_scale=self.vertical_scale,
                normalize=self.normalize,
                deform_surface=self.deform_surface,
                normalize_colors=self.normalize_colors,
                lod_factor=self.lod_factor,
            )
            self.viewer._mesh_cache.put(cache_key, new_surface)

        self._replace_surface_data(new_surface)

        time_index = int(self.viewer.snapshot_time_indices[snapshot_id])
        time_value = float(self.viewer.snapshot_times[snapshot_id])
        max_amp = float(np.max(np.abs(field)))

        self.info_frame.setText(f"{frame_id + 1}/{len(self.snapshot_ids)} | snapshot {snapshot_id}")
        self.info_time.setText(f"it = {time_index} | t = {time_value:.5f} s")
        self.info_amp.setText(f"{max_amp:.6e}")

        if self.frame_slider.value() != frame_id:
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(frame_id)
            self.frame_slider.blockSignals(False)

        if self.viewer.seismogram_data is not None:
            for line in self.seismogram_time_lines:
                line.set_xdata([time_value, time_value])
        else:
            self.time_line.set_xdata([frame_id, frame_id])

        self.canvas.draw_idle()
        self.plotter.render()
        self.plotter.update()
        if self.velocity_overlay_canvas is not None:
            self.velocity_overlay_canvas.raise_()

    # ------------------------------------------------------------------
    # Events and commands
    # ------------------------------------------------------------------
    def _connect_events(self):
        self.frame_slider.valueChanged.connect(self._update_frame)
        self.btn_prev.clicked.connect(lambda: self._step_frame(-1))
        self.btn_play.clicked.connect(self._toggle_playback)
        self.btn_next.clicked.connect(lambda: self._step_frame(1))
        self.timer.timeout.connect(self._timer_tick)

        self.speed_spin.valueChanged.connect(self._set_frame_rate)
        self.chk_deform.toggled.connect(self._set_deformation)
        self.chk_bounds.toggled.connect(self._set_bounds_visible)
        self.chk_axes.toggled.connect(self._set_orientation_axes_visible)
        self.scale_spin.valueChanged.connect(self._set_vertical_scale)
        self.cmap_combo.currentTextChanged.connect(self._set_colormap)
        self.color_min_spin.valueChanged.connect(self._set_color_limits_from_controls)
        self.color_max_spin.valueChanged.connect(self._set_color_limits_from_controls)

        self.btn_oblique.clicked.connect(self._set_oblique_camera)
        self.btn_top.clicked.connect(self._set_top_camera)
        self.btn_front.clicked.connect(self._set_front_camera)
        self.btn_right.clicked.connect(self._set_right_camera)
        self.btn_fit.clicked.connect(self.plotter.reset_camera)
        self.chk_orthographic.toggled.connect(self._set_orthographic_projection)

    def _step_frame(self, delta):
        next_frame = (self.current_frame + int(delta)) % len(self.snapshot_ids)
        self._update_frame(next_frame)

    def _toggle_playback(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play.setText("Play")
        else:
            self.timer.start()
            self.btn_play.setText("Pause")

    def _timer_tick(self):
        self._step_frame(1)

    def _set_frame_rate(self, value):
        self.frame_rate = max(float(value), 0.2)
        self.timer.setInterval(max(int(1000 / self.frame_rate), 1))

    def _set_deformation(self, checked):
        self.deform_surface = bool(checked)
        self.viewer._mesh_cache.clear()
        self._update_frame(self.current_frame)

    def _set_vertical_scale(self, value):
        self.vertical_scale = float(value)
        self.viewer._mesh_cache.clear()
        self._update_frame(self.current_frame)

    def _refresh_wavefield_coloring(self):
        """
        Rebuild the wavefield actor when cmap or scalar limits change.
        """
        self.plotter.remove_actor("wavefield_surface", reset_camera=False, render=False)
        self._add_wavefield_actor(show_scalar_bar=True)
        self.plotter.render()

    def _update_scalar_range(self):
        """
        Update only the mapper and scalar bar range for interactive color limits.
        """
        if self.mesh_actor is not None:
            self.mesh_actor.mapper.SetScalarRange(float(self.clim[0]), float(self.clim[1]))
            self.mesh_actor.mapper.Modified()

        try:
            self.plotter.update_scalar_bar_range(self.clim, name="Amplitude")
        except Exception:
            try:
                self.plotter.update_scalar_bar_range(self.clim)
            except Exception:
                pass

        self.plotter.render()
        self.plotter.update()

    def _set_colormap(self, cmap):
        self.cmap = str(cmap)
        color_limit_enabled = self.cmap == "RdBu_r"
        self.color_min_spin.setEnabled(color_limit_enabled)
        self.color_max_spin.setEnabled(color_limit_enabled)

        if not color_limit_enabled:
            self.clim = self._resolve_color_limits(None)
            self.color_min_spin.blockSignals(True)
            self.color_max_spin.blockSignals(True)
            self.color_min_spin.setValue(float(self.clim[0]))
            self.color_max_spin.setValue(float(self.clim[1]))
            self.color_min_spin.blockSignals(False)
            self.color_max_spin.blockSignals(False)

        else:
            self.clim = (
                float(self.color_min_spin.value()),
                float(self.color_max_spin.value()),
            )

        self._refresh_wavefield_coloring()

    def _set_color_limits_from_controls(self):
        if self.cmap != "RdBu_r":
            return

        color_min = float(self.color_min_spin.value())
        color_max = float(self.color_max_spin.value())

        if color_min >= color_max:
            return

        self.clim = (color_min, color_max)
        self._update_scalar_range()

    def _set_orthographic_projection(self, checked):
        if checked:
            self.plotter.enable_parallel_projection()
        else:
            self.plotter.disable_parallel_projection()

        self.plotter.render()

    # ------------------------------------------------------------------
    # Camera presets
    # ------------------------------------------------------------------
    def _set_oblique_camera(self):
        x_mid = 0.5 * (self.viewer.nx - 1) * self.viewer.dx
        z_mid = 0.5 * (self.viewer.nz - 1) * self.viewer.dz

        x_span = (self.viewer.nx - 1) * self.viewer.dx
        z_span = (self.viewer.nz - 1) * self.viewer.dz
        span = max(x_span, z_span)

        self.plotter.camera_position = [
            (x_mid - 1.78 * x_span, 1.72 * span, z_mid - 0.22 * z_span),
            (x_mid, 0.0, z_mid),
            (0.0, 0.0, -1.0),
        ]
        self.plotter.camera.zoom(1.04)
        self.plotter.render()

    def _set_top_camera(self):
        x_mid = 0.5 * (self.viewer.nx - 1) * self.viewer.dx
        z_mid = 0.5 * (self.viewer.nz - 1) * self.viewer.dz
        span = max((self.viewer.nx - 1) * self.viewer.dx, (self.viewer.nz - 1) * self.viewer.dz)

        self.plotter.camera_position = [
            (x_mid, 2.0 * span, z_mid),
            (x_mid, 0.0, z_mid),
            (0.0, 0.0, -1.0),
        ]
        self.plotter.render()

    def _set_front_camera(self):
        x_mid = 0.5 * (self.viewer.nx - 1) * self.viewer.dx
        z_mid = 0.5 * (self.viewer.nz - 1) * self.viewer.dz
        span = max((self.viewer.nx - 1) * self.viewer.dx, (self.viewer.nz - 1) * self.viewer.dz)

        self.plotter.camera_position = [
            (x_mid, 1.3 * span, -0.8 * span),
            (x_mid, 0.0, z_mid),
            (0.0, 1.0, 0.0),
        ]
        self.plotter.render()

    def _set_right_camera(self):
        x_mid = 0.5 * (self.viewer.nx - 1) * self.viewer.dx
        z_mid = 0.5 * (self.viewer.nz - 1) * self.viewer.dz
        span = max((self.viewer.nx - 1) * self.viewer.dx, (self.viewer.nz - 1) * self.viewer.dz)

        self.plotter.camera_position = [
            (1.5 * span, 0.8 * span, z_mid),
            (x_mid, 0.0, z_mid),
            (0.0, 1.0, 0.0),
        ]
        self.plotter.render()
