"""
utils.py — Spike Detection and Temporal Binning Utilities
==========================================================

Provides GPU-accelerated functions to extract temporal features from the
continuous current trajectories produced by the VO₂ reservoir simulation.

The feature extraction pipeline is:

    1. ``find_peaks``  — detect spike events in each neuristor's I(t) trace.
    2. ``bin_traj``    — bin the binary spike mask into coarser time windows,
                         yielding a compact spatiotemporal feature matrix.

This "temporal pooling" approach is the key contribution of this work: instead
of counting total spikes (which discards timing information), we preserve the
*temporal structure* of the spiking pattern by recording how many spikes fell
within each 500 ns window.  For the 10 µs simulation with 500 ns windows,
this yields 20 bins per neuristor, giving a 784 × 20 = 15,680-dimensional
feature vector per image.

All functions operate on PyTorch tensors and are compatible with CUDA.
"""

import warnings
import torch


def find_peaks(
    y: torch.Tensor,
    threshold: float,
    min_dist: int,
) -> torch.Tensor:
    """
    Detect local maxima (spikes) in a batch of 1-D time series.

    A time step ``t`` is classified as a spike if ALL of the following hold:

    * It is a local maximum: ``y[t-1] < y[t] > y[t+1]``
    * It exceeds the absolute threshold: ``y[t] > threshold``
    * It is the unique maximum within a ``min_dist``-wide neighbourhood
      (prevents double-counting of broad peaks).

    This GPU-friendly implementation uses ``max_pool1d`` with ``return_indices``
    to efficiently enforce the minimum-distance constraint across all traces
    simultaneously.

    Parameters
    ----------
    y : torch.Tensor
        Batch of time series, shape (batch, length).
        Typically the current trajectory I(t) for N neuristors, reshaped to
        (batch × N, n_steps).
    threshold : float
        Minimum current value for a valid spike [mA].
        Paper value: 1.5 mA.
    min_dist : int
        Minimum number of time steps between consecutive spikes.
        Must be odd (enforced internally).
        Paper value: 101 steps (≈ 1 µs at dt = 10 ns).

    Returns
    -------
    torch.Tensor
        Boolean spike mask, shape (batch, length).
        ``True`` at each detected spike location.

    Raises
    ------
    UserWarning
        If ``min_dist`` is even (automatically incremented to the next odd value).
    """
    if min_dist % 2 == 0:
        warnings.warn(
            f"min_dist must be odd, but got {min_dist}. Incrementing to {min_dist + 1}."
        )
        min_dist += 1

    batch = y.shape[0]

    # --- Condition 1: strict local maximum ---
    local_maxima_mask = torch.cat(
        [
            torch.zeros(batch, 1, dtype=torch.bool, device=y.device),
            (y[:, :-2] < y[:, 1:-1]) & (y[:, 2:] < y[:, 1:-1]),
            torch.zeros(batch, 1, dtype=torch.bool, device=y.device),
        ],
        dim=1,
    )

    # --- Condition 2: above absolute threshold ---
    threshold_mask = y > threshold

    # --- Condition 3: maximum within min_dist neighbourhood (max_pool trick) ---
    # max_pool1d returns the index of the maximum in each window;
    # only keep positions that are their own window maximum.
    _, indices = torch.nn.functional.max_pool1d(
        y,
        kernel_size=min_dist,
        stride=1,
        padding=min_dist // 2,
        return_indices=True,
    )
    maxpool_mask = torch.zeros_like(y, dtype=torch.bool)
    maxpool_mask.scatter_(1, indices, True)

    return local_maxima_mask & threshold_mask & maxpool_mask


def traj2peak(
    y: torch.Tensor,
    peak_threshold: float,
    min_dist: int,
) -> torch.Tensor:
    """
    Count the total number of spikes at each time step across a batch.

    Convenience wrapper around ``find_peaks`` that sums the spike mask along
    the batch dimension.  Useful for plotting mean firing rates.

    Parameters
    ----------
    y : torch.Tensor
        Current trajectories, shape (batch, length).
    peak_threshold : float
        Spike detection threshold [mA].
    min_dist : int
        Minimum inter-spike distance [steps].

    Returns
    -------
    torch.Tensor
        Spike count at each time step, shape (length,).
    """
    peak_mask = find_peaks(y, peak_threshold, min_dist)
    return peak_mask.sum(dim=0)


def bin_peaks(peaks: torch.Tensor, bin_size: int) -> torch.Tensor:
    """
    Bin a 1-D spike train into coarser time windows (count-per-bin).

    Parameters
    ----------
    peaks : torch.Tensor
        Binary spike train, shape (length,).
    bin_size : int
        Number of time steps per bin.

    Returns
    -------
    torch.Tensor
        Spike counts per bin, shape (length // bin_size,).
    """
    return bin_size * torch.nn.functional.avg_pool1d(
        peaks.float().unsqueeze(0), bin_size, stride=bin_size
    ).squeeze(0)


def bin_traj(
    peak_mask: torch.Tensor,
    len_x: int,
    len_y: int,
) -> torch.Tensor:
    """
    Apply 2-D average pooling to a spatiotemporal spike mask.

    Reduces the full spike mask of shape (batch, N, n_steps) into a compact
    feature matrix by pooling over ``len_x`` neurons and ``len_y`` time steps.

    For the MNIST reservoir (len_x=1, len_y=50):
        Input:  (batch, 784, 1000)  — 784 neurons, 1000 time steps
        Output: (batch, 784, 20)    — 784 neurons, 20 temporal bins

    The divisor is overridden to 1 so that the output is a *count* (not a
    fraction), consistent with the temporal pooling interpretation.

    Parameters
    ----------
    peak_mask : torch.Tensor
        Binary spike mask, shape (batch, N, n_steps).
    len_x : int
        Pooling window size along the neuron axis.
        Set to 1 to keep individual neurons (paper's approach).
    len_y : int
        Pooling window size along the time axis.
        Paper: 50 steps × 10 ns/step = 500 ns per bin.

    Returns
    -------
    torch.Tensor
        Binned spike counts, shape (batch, N // len_x, n_steps // len_y).
        For paper parameters: (batch, 784, 20).
    """
    return torch.nn.functional.avg_pool2d(
        peak_mask.float().unsqueeze(0),  # add channel dim for avg_pool2d
        kernel_size=(len_x, len_y),
        stride=(len_x, len_y),
        divisor_override=1,              # return sum (count), not mean
    ).squeeze(0)
