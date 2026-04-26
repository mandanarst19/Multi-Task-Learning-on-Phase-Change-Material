"""
tests/test_reservoir.py — Unit Tests for Core Components
=========================================================

Tests run on CPU with a minimal 3×3 grid to verify correctness
without requiring a GPU or long simulation times.

Run with:
    python -m pytest tests/test_reservoir.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import pytest

from model import VO2, Circuit, Circuit2D, P
from utils import find_peaks, bin_traj, bin_peaks


# ---------------------------------------------------------------------------
# VO2 model tests
# ---------------------------------------------------------------------------

class TestVO2:
    """Tests for the VO₂ hysteresis resistance model."""

    def test_initialization(self):
        """VO2 initialises without error and produces finite values."""
        model = VO2(N=4)
        model.initialize(T0=325.0)
        assert model.Tr is not None
        assert model.gr is not None
        assert torch.isfinite(model.gr).all()

    def test_resistance_insulating(self):
        """At low temperature (325 K), resistance should be high (insulating)."""
        model = VO2(N=1)
        model.initialize(T0=324.9)
        T = torch.tensor([325.0])
        R = model.R(T)
        assert R.item() > 100.0, f"Expected R >> 1 kΩ in insulating state, got {R.item():.1f} kΩ"

    def test_resistance_metallic(self):
        """At high temperature (360 K), resistance should be low (metallic)."""
        model = VO2(N=1)
        model.initialize(T0=359.9)
        model.delta = -torch.ones(1)   # Cooling branch
        T = torch.tensor([360.0])
        R = model.R(T)
        assert R.item() < 5.0, f"Expected R < 5 kΩ in metallic state, got {R.item():.1f} kΩ"

    def test_resistance_shape(self):
        """R output shape matches input batch size."""
        N = 16
        model = VO2(N=N)
        model.initialize(T0=325.0)
        T = torch.full((N,), 325.0)
        R = model.R(T)
        assert R.shape == (N,), f"Expected shape ({N},), got {R.shape}"

    def test_temperature_clamping(self):
        """Temperatures outside [305, 370] K are clamped safely."""
        model = VO2(N=2)
        model.initialize(T0=325.0)
        T = torch.tensor([200.0, 500.0])   # Far out of range
        R = model.R(T)
        assert torch.isfinite(R).all(), "R should be finite even for extreme T"

    def test_P_function_bounds(self):
        """P(x, gamma) should be in [0, 1] for x in [-10, 10]."""
        x = torch.linspace(-10, 10, 100)
        result = P(x, gamma=0.956)
        assert (result >= 0).all() and (result <= 1).all(), "P must be in [0, 1]"


# ---------------------------------------------------------------------------
# Spike detection tests
# ---------------------------------------------------------------------------

class TestFindPeaks:
    """Tests for the GPU-compatible spike detection function."""

    def test_detects_clear_spike(self):
        """A single isolated spike above threshold is detected."""
        y = torch.zeros(1, 200)
        y[0, 100] = 2.0   # Clear spike at step 100
        y[0, 99] = 1.0    # Rising
        y[0, 101] = 1.0   # Falling
        peaks = find_peaks(y, threshold=1.5, min_dist=11)
        assert peaks[0, 100].item(), "Spike at step 100 should be detected"

    def test_ignores_subthreshold(self):
        """Values below threshold are not flagged as spikes."""
        y = torch.zeros(1, 100)
        y[0, 50] = 1.0    # Local max but below threshold=1.5
        peaks = find_peaks(y, threshold=1.5, min_dist=11)
        assert not peaks.any(), "Subthreshold peak should not be detected"

    def test_min_dist_enforced(self):
        """Two spikes closer than min_dist are merged into one."""
        y = torch.zeros(1, 200)
        y[0, 50] = 2.0
        y[0, 49] = y[0, 51] = 1.6
        y[0, 60] = 2.5    # Second spike only 10 steps away (min_dist=101)
        y[0, 59] = y[0, 61] = 1.8
        peaks = find_peaks(y, threshold=1.5, min_dist=101)
        n_peaks = peaks.sum().item()
        assert n_peaks <= 1, f"Expected ≤1 peak (min_dist enforced), got {n_peaks}"

    def test_output_shape(self):
        """Output shape matches input shape."""
        batch, length = 8, 500
        y = torch.rand(batch, length)
        peaks = find_peaks(y, threshold=0.5, min_dist=11)
        assert peaks.shape == (batch, length)

    def test_odd_min_dist_warning(self):
        """Even min_dist triggers a UserWarning."""
        y = torch.zeros(1, 100)
        with pytest.warns(UserWarning, match="min_dist must be odd"):
            find_peaks(y, threshold=1.5, min_dist=10)


# ---------------------------------------------------------------------------
# Temporal binning tests
# ---------------------------------------------------------------------------

class TestBinTraj:
    """Tests for the temporal pooling (bin_traj) function."""

    def test_output_shape(self):
        """bin_traj reduces time axis by factor len_y."""
        batch, N, n_steps = 4, 16, 100
        peaks = torch.zeros(batch, N, n_steps)
        binned = bin_traj(peaks, len_x=1, len_y=5)
        assert binned.shape == (batch, N, n_steps // 5)

    def test_spike_count_preserved(self):
        """Total spike count is preserved across binning."""
        batch, N, n_steps = 2, 9, 100
        peaks = torch.zeros(batch, N, n_steps)
        peaks[0, 0, 10] = 1   # Spike at step 10
        peaks[0, 0, 60] = 1   # Spike at step 60
        binned = bin_traj(peaks, len_x=1, len_y=10)
        # Total count across bins should equal 2
        assert binned[0, 0].sum().item() == 2.0

    def test_spatial_pooling(self):
        """len_x > 1 reduces neuron axis."""
        batch, N, n_steps = 2, 8, 40
        peaks = torch.zeros(batch, N, n_steps)
        binned = bin_traj(peaks, len_x=2, len_y=5)
        assert binned.shape == (batch, N // 2, n_steps // 5)


# ---------------------------------------------------------------------------
# Circuit2D integration test (tiny grid, short simulation)
# ---------------------------------------------------------------------------

class TestCircuit2D:
    """Smoke tests for the full 2-D circuit ODE integrator."""

    def test_forward_pass_shape(self):
        """Circuit2D.solve() returns tensors of expected shape."""
        batch, Nx, Ny = 2, 3, 3
        N = Nx * Ny
        circuit = Circuit2D(
            batch=batch, Nx=Nx, Ny=Ny,
            V=11.0, R=12, noise_strength=0.0002,
            Cth_factor=0.15, couple_factor=0.02, width_factor=1.0
        )
        y0 = torch.stack(
            [torch.zeros(batch, N), torch.full((batch, N), 325.0)], dim=1
        )
        # Short simulation: 100 ns with dt=10 ns
        y_final, I_traj = circuit.solve(y0, t_max=100, dt=10)
        assert y_final.shape == (batch, 2, N), f"Unexpected y_final shape: {y_final.shape}"
        assert I_traj.shape == (batch, N, 10), f"Unexpected I_traj shape: {I_traj.shape}"

    def test_temperature_positive(self):
        """Temperature should remain above 300 K throughout."""
        batch, Nx, Ny = 1, 3, 3
        N = Nx * Ny
        circuit = Circuit2D(
            batch=batch, Nx=Nx, Ny=Ny,
            V=11.0, R=12, noise_strength=0.0,   # No noise for determinism
            Cth_factor=0.15, couple_factor=0.02, width_factor=1.0
        )
        y0 = torch.stack(
            [torch.zeros(batch, N), torch.full((batch, N), 325.0)], dim=1
        )
        y_final, _ = circuit.solve(y0, t_max=50, dt=10)
        T_final = y_final[:, 1, :]
        assert (T_final > 300).all(), "Temperature dropped below physical range"

    def test_current_finite(self):
        """Current trajectory should contain only finite values."""
        batch, Nx, Ny = 1, 3, 3
        N = Nx * Ny
        circuit = Circuit2D(
            batch=batch, Nx=Nx, Ny=Ny,
            V=11.0, R=12, noise_strength=0.0002,
            Cth_factor=0.15, couple_factor=0.02, width_factor=1.0
        )
        y0 = torch.stack(
            [torch.zeros(batch, N), torch.full((batch, N), 325.0)], dim=1
        )
        _, I_traj = circuit.solve(y0, t_max=100, dt=10)
        assert torch.isfinite(I_traj).all(), "I_traj contains NaN or Inf values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
