"""
reservoir.py — Reservoir2D: VO₂ Feature Extraction Module
==========================================================

Wraps ``Circuit2D`` and the temporal pooling pipeline into a
``torch.nn.Module`` that accepts normalised pixel arrays and returns
high-dimensional spatiotemporal feature vectors suitable for downstream
linear classifiers.

Architecture
------------

    Normalised image  [0, 1]^784
            │
            │  Voltage mapping
            │  V = V_min + (V_max − V_min) × pixel
            ▼
    Circuit2D (28×28 VO₂ grid, FIXED — no gradient)
    10 µs simulation, dt = 10 ns → I(t) shape (batch, 784, 1000)
            │
            │  find_peaks → binary spike mask
            │  bin_traj   → temporal pooling (500 ns bins)
            ▼
    Feature vector: (batch, 784 × 20) = (batch, 15 680)
            │
            │  nn.Linear (trained)
            ▼
    Log-softmax output: (batch, N_out)

Note
----
In the multi-task paper, the ``self.out`` linear layer is *not* used.
Instead, three independent ``RidgeClassifier`` objects are trained on
the raw 15,680-dimensional feature vectors returned by ``reservoir_func``.
The linear layer is retained here for compatibility with the single-task
baseline and potential future use with gradient-based training.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

from model import Circuit2D
from utils import find_peaks, bin_traj


class Reservoir2D(nn.Module):
    """
    VO₂ reservoir computing module for 2-D image inputs.

    Implements the feature extraction pipeline from Zhang et al. (2024):
    maps each pixel of a 28×28 image to a VO₂ neuristor's input voltage,
    runs the physical simulation, and extracts a temporal pooling feature
    vector via spike detection and binning.

    The reservoir dynamics are **entirely fixed** — no parameters are updated
    during training.  Only the downstream readout (``self.out``) or external
    classifiers are trained.

    Parameters
    ----------
    batch : int
        Number of images processed simultaneously.  Limited by GPU memory;
        use 50 on a 15 GB T4.
    Nx : int
        Grid rows.  Must match image height (28 for MNIST).
    Ny : int
        Grid columns.  Must match image width (28 for MNIST).
    N_out : int
        Output dimensionality of the built-in linear readout.
        (Not used in multi-task Ridge Regression pipeline.)
    V_min : float
        Input voltage mapped to zero pixel intensity [V].
        Paper-validated value: 10.5 V.
    V_max : float
        Input voltage mapped to maximum pixel intensity [V].
        Paper-validated value: 12.2 V.
    Cth_factor : float
        Thermal capacitance scaling factor.
        Paper-validated value: 0.15  (Cth = 7.44 pJ/K).
    noise_strength : float
        Stochastic noise amplitude [µJ·s⁻¹/²].
        Paper-validated value: 2×10⁻⁴ (0.2×10⁻³).

    Attributes
    ----------
    t_max : int
        Simulation duration [ns].  Default 10,000 ns (10 µs).
    dt : int
        Integration time step [ns].  Default 10 ns.
    len_y : int
        Number of time steps per temporal bin.  Default 50 (= 500 ns).
    len_t : int
        Number of temporal bins per neuristor.  Default 20.
    peak_threshold : float
        Minimum current for spike detection [mA].  Default 1.5 mA.
    min_dist : int
        Minimum inter-spike distance [steps].  Default 101 steps (1.01 µs).
    """

    def __init__(
        self,
        batch: int,
        Nx: int,
        Ny: int,
        N_out: int,
        V_min: float = 11.0,
        V_max: float = 13.0,
        Cth_factor: float = 1.0,
        noise_strength: float = 0.001,
    ):
        super().__init__()

        # --- Grid / batch parameters ---
        self.batch = batch
        self.Nx = Nx
        self.Ny = Ny
        self.N = Nx * Ny              # Total number of neuristors (784 for MNIST)
        self.N_out = N_out

        # --- Voltage mapping parameters ---
        self.V = V_min                # Default constant input voltage [V]
        self.V_min = V_min            # Voltage for zero pixel intensity [V]
        self.V_max = V_max            # Voltage for max pixel intensity [V]

        # --- Physical circuit parameters ---
        self.R = 12                   # Load resistance [kΩ]
        self.noise_strength = noise_strength
        self.Cth_factor = Cth_factor
        self.couple_factor = 0.02     # Nearest-neighbour thermal coupling
        self.width_factor = 1.0       # VO₂ hysteresis width scaling
        self.T_base = 325             # Ambient temperature [K]

        # --- Simulation parameters ---
        self.t_max = 10_000           # Simulation duration [ns] = 10 µs
        self.dt = 10                  # Integration time step [ns]
        self.n_repeat = 1             # Number of simulation repeats (not used)

        # --- Spike detection parameters ---
        self.peak_threshold = 1.5     # Spike threshold [mA]
        self.min_dist = 101           # Minimum inter-spike distance [steps]

        # --- Temporal binning parameters ---
        self.len_x = 1                # Pool size along neuron axis (keep individual)
        self.len_y = 50               # Pool size along time axis: 50 × 10 ns = 500 ns
        self.n_step = int(np.ceil(self.t_max / self.dt))           # 1000 time steps
        self.len_t = int(np.ceil(self.n_step / self.len_y))        # 20 temporal bins

        # Human-readable identifier for caching / logging
        self.name_string = (
            f"{self.N}_{self.V:.1f}_{1000 * self.noise_strength:.4f}_"
            f"{self.Cth_factor:.4f}_{self.couple_factor:.4f}_{self.width_factor:.4f}"
        )

        # --- Sub-modules ---
        # Physical reservoir (fixed — gradients disabled)
        self.reservoir = Circuit2D(
            self.batch, self.Nx, self.Ny, self.V, self.R,
            self.noise_strength, self.Cth_factor,
            self.couple_factor, self.width_factor, self.T_base,
        )

        # Built-in linear readout (used only for single-task baseline)
        self.out = nn.Linear(self.N * self.len_t, self.N_out)

    def reset(
        self,
        V_min: float,
        V_max: float,
        Cth_factor: float,
        noise_strength: float,
    ):
        """
        Re-initialise the reservoir with new hyperparameters.

        Useful for hyperparameter search without recreating the module.

        Parameters
        ----------
        V_min, V_max : float
            New voltage range [V].
        Cth_factor : float
            New thermal capacitance factor.
        noise_strength : float
            New noise amplitude [µJ·s⁻¹/²].
        """
        self.V_min = V_min
        self.V_max = V_max
        self.Cth_factor = Cth_factor
        self.noise_strength = noise_strength
        self.reservoir.__init__(
            self.batch, self.Nx, self.Ny, self.V, self.R,
            self.noise_strength, self.Cth_factor,
            self.couple_factor, self.width_factor, self.T_base,
        )
        self.out.reset_parameters()

    def reservoir_func(self, V_input: torch.Tensor) -> torch.Tensor:
        """
        Run the full reservoir pipeline and return temporal pooling features.

        This is the main feature extraction method used in the multi-task
        paper.  It:

        1. Injects the voltage-mapped pixel values into the neuristor grid.
        2. Integrates the V–T ODEs for ``t_max`` ns.
        3. Detects spikes in the current trajectory using ``find_peaks``.
        4. Bins spikes into ``len_t`` temporal windows using ``bin_traj``.
        5. Flattens the result to a 1-D feature vector per image.

        Parameters
        ----------
        V_input : torch.Tensor
            Input voltages, shape (batch, N) [V].
            Should already be in the range [V_min, V_max].

        Returns
        -------
        torch.Tensor
            Temporal pooling feature vectors, shape (batch, N × len_t).
            For MNIST (28×28, 20 bins): shape (batch, 15 680).
        """
        # Inject pixel voltages into the reservoir
        self.reservoir.set_input(V=V_input)

        # Initial conditions: V₁ = 0, T = T_base
        y0 = torch.stack(
            [
                torch.zeros(self.batch, self.N),
                torch.ones(self.batch, self.N) * self.T_base,
            ],
            dim=1,
        )

        # Integrate ODEs — I_traj: (batch, N, n_steps)
        _, I_traj = self.reservoir.solve(y0, self.t_max, self.dt)

        # Detect spikes in flattened (batch × N, n_steps) array
        peaks = find_peaks(
            I_traj.reshape(self.batch * self.N, self.n_step),
            self.peak_threshold,
            self.min_dist,
        )
        peaks = peaks.reshape(self.batch, self.N, self.n_step)  # (batch, N, 1000)

        # Temporal binning: (batch, N, 1000) → (batch, N, 20)
        binned = bin_traj(peaks, self.len_x, self.len_y)

        # Flatten: (batch, N, 20) → (batch, 15 680)
        return binned.reshape(self.batch, self.N * self.len_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: pixel values → log-softmax class scores.

        Used for the single-task baseline with gradient-based training.
        For multi-task Ridge Regression, use ``reservoir_func`` directly.

        Parameters
        ----------
        x : torch.Tensor
            Normalised pixel values in [0, 1], shape (batch, N).

        Returns
        -------
        torch.Tensor
            Log-softmax class probabilities, shape (batch, N_out).
        """
        # Map [0, 1] pixels to [V_min, V_max] voltages
        V_input = self.V_min + (self.V_max - self.V_min) * x

        # Extract reservoir features
        reservoir_output = self.reservoir_func(V_input)  # (batch, N × len_t)

        # Linear readout + log-softmax
        return nn.functional.log_softmax(self.out(reservoir_output), dim=-1)
