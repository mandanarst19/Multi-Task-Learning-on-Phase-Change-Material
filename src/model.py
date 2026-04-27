"""
model.py — VO₂ Thermal Neuristor Physics Model
===============================================

Implements the coupled electrical and thermal dynamics of a vanadium dioxide
(VO₂) neuristor network, following:

    Zhang et al., "Collective dynamics and long-range order in thermal
    

Physical picture
----------------
Each VO₂ element undergoes a sharp insulator-to-metal transition (IMT) near
~340 K.  When an input voltage drives sufficient Joule heating, the temperature
rises above the IMT, resistance drops abruptly (~MΩ → ~kΩ), and a current
spike is produced.  After spiking, reduced dissipation lets the element cool
and return to the insulating state.  Nearest-neighbour thermal coupling through
``S_couple`` synchronises adjacent elements, producing rich collective dynamics.

The simulation integrates the following coupled ODEs using the Euler–Maruyama
scheme (stochastic, forward-Euler):

    Electrical:
        C dV/dt = (V_in − V) / R_load − V / R(T)

    Thermal:
        C_th dT/dt = I²R(T) − S_env(T − T₀) + S_couple ∇²T + σ ξ(t)

    VO₂ resistance (hysteresis model):
        R(T, δ) = R_m + [R₀ exp(Eₐ/T) − R_m] / [1 + exp(−(T − Tᵣ − w(1−δ)/2)/(w/β))]^γ

where δ = +1 (heating) or −1 (cooling) tracks the hysteresis branch.

Classes
-------
VO2         : Hysteresis resistance model for a batch of VO₂ elements.
Circuit     : 1-D chain of thermally coupled neuristors + ODE solver.
Circuit2D   : 2-D grid extension (used for MNIST pixel mapping).
"""

import time
import numpy as np
import torch
import torch.nn as nn

pi = np.pi


# ---------------------------------------------------------------------------
# Hysteresis helper
# ---------------------------------------------------------------------------

def P(x: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Minor-loop interpolation function for the VO₂ hysteresis model.

    Produces a smooth transition between the major hysteresis loop and the
    reversal point, ensuring continuity of resistance during partial cycles.

    Parameters
    ----------
    x : torch.Tensor
        Normalised temperature offset from the reversal point,
        (T − T_reversal) / T_pr.
    gamma : float
        Shape parameter of the minor-loop function (≈ 0.956 from fit).

    Returns
    -------
    torch.Tensor
        Values in [0, 1] — 0 at the reversal point, 1 far from it.
    """
    return 0.5 * (1 - torch.sin(gamma * x)) * (1 + torch.tanh(pi**2 - 2 * pi * x))


# ---------------------------------------------------------------------------
# VO₂ resistance model
# ---------------------------------------------------------------------------

class VO2:
    """
    Temperature-dependent resistance model for a batch of VO₂ elements,
    including first-order phase-transition hysteresis.

    The model captures the sharp insulator-to-metal transition of VO₂ near
    T_c ≈ 333 K with a hysteresis width w ≈ 7.19 K.  Minor-loop behaviour
    (partial reversals before completing the transition) is tracked via the
    reversal state variables ``Tr``, ``gr``, and ``Tpr``.

    All parameters are taken from the experimental fit in Zhang et al. (2024).

    Parameters
    ----------
    N : int
        Number of VO₂ elements in the batch.
    width_factor : float, optional
        Multiplicative scaling of the hysteresis width ``w``.
        Default 1.0 (no scaling).

    Attributes
    ----------
    w : float
        Hysteresis full-width [K].
    Tc : float
        IMT centre temperature [K].
    beta : float
        Transition sharpness [K⁻¹].
    R0 : float
        Insulating-state pre-exponential resistance [kΩ·exp(−Eₐ/T)].
    Ea : float
        Activation energy for insulating resistance [K].
    Rm : float
        Metallic-state resistance floor [Ω].
    delta : torch.Tensor
        Current heating/cooling direction (+1 or −1) for each element.
    """

    def __init__(self, N: int, width_factor: float = 1.0):
        self.N = N
        self.width_factor = width_factor

        # --- Fitted VO₂ parameters (Zhang et al., 2024, Table I) ---
        self.w = 7.19357064e+00 * width_factor   # Hysteresis width [K]
        self.Tc = 3.32805839e+02                  # IMT centre temperature [K]
        self.beta = 2.52796285e-01                # Transition sharpness [K⁻¹]
        self.R0 = 5.35882879e-03                  # Pre-exponential [kΩ]
        self.Ea = 5.22047417e+03                  # Activation energy [K]
        self.gamma = 9.56269682e-01               # Minor-loop shape parameter

        # Metallic resistance: R_m = R_m0 × R_m_factor [Ω]
        self.Rm0 = 262.5
        self.Rm_factor = 4.90025335
        self.Rm = self.Rm0 * self.Rm_factor       # ≈ 1286 Ω

        # --- Hysteresis state variables (initialised in self.initialize) ---
        self.delta = torch.ones(N)     # +1 = heating, −1 = cooling
        self.reversed = torch.zeros(N) # 1 if currently on a minor loop
        self.Tr = None                 # Temperature at last reversal [K]
        self.gr = None                 # g(T) value at last reversal
        self.Tpr = None                # Minor-loop temperature scale [K]
        self.T_last = None             # Temperature at previous time step [K]

    def initialize(self, T0: float):
        """
        Set all elements to the equilibrium state at temperature T0.

        Must be called before the first simulation step.

        Parameters
        ----------
        T0 : float
            Initial temperature [K].  Typically T_base − 0.1 K to ensure the
            system starts in the insulating state.
        """
        T = T0 * torch.ones(self.N)
        self.gr = self.g_major(T)
        self.Tr = T
        self.Tpr = self.Tpr_func()
        self.T_last = T

    def reversal(self, T: torch.Tensor):
        """
        Detect temperature reversals and update the minor-loop state.

        Called at every integration step *before* evaluating R(T).  Elements
        that reverse direction (dT changes sign) are placed on a minor loop
        originating from the current temperature and g value.

        Parameters
        ----------
        T : torch.Tensor
            Current temperatures for all N elements [K], shape (N,).
        """
        T = T.clamp(305, 370)        # Physical temperature bounds [K]
        dT = T - self.T_last
        if dT.abs().max() > 0.01:   # Only update if temperature has changed
            delta = torch.sign(dT)
            reversal_mask = (delta != self.delta) & (delta != 0)
            if reversal_mask.any():
                self.gr[reversal_mask] = self.g(T)[reversal_mask]
                self.delta[reversal_mask] = delta[reversal_mask]
                self.reversed[reversal_mask] = 1
                self.Tr[reversal_mask] = T[reversal_mask]
                self.Tpr[reversal_mask] = self.Tpr_func()[reversal_mask]
            self.T_last = T

    def Tpr_func(self) -> torch.Tensor:
        """
        Compute the minor-loop temperature scale T_pr from current state.

        Returns
        -------
        torch.Tensor
            Temperature scale for current minor loops [K], shape (N,).
        """
        return (self.delta * self.w / 2 + self.Tc
                - torch.arctanh(2 * self.gr - 1) / self.beta - self.Tr)

    def g_major(self, T: torch.Tensor) -> torch.Tensor:
        """
        Metallic fraction on the major (outer) hysteresis loop.

        Returns
        -------
        torch.Tensor
            Values in [0, 1]; 0 = fully insulating, 1 = fully metallic.
        """
        return 0.5 + 0.5 * torch.tanh(
            self.beta * (self.delta * self.w / 2 + self.Tc - T))

    def g(self, T: torch.Tensor) -> torch.Tensor:
        """
        Effective metallic fraction including minor-loop correction.

        Returns
        -------
        torch.Tensor
            Values in [0, 1], shape (N,).
        """
        Tp = (self.Tpr * P((T - self.Tr) / (self.Tpr + 1e-6), self.gamma)
              * self.reversed)
        return 0.5 + 0.5 * torch.tanh(
            self.beta * (self.delta * self.w / 2 + self.Tc - (T + Tp)))

    def R(self, T: torch.Tensor) -> torch.Tensor:
        """
        Temperature-dependent VO₂ resistance including hysteresis.

        Parameters
        ----------
        T : torch.Tensor
            Temperature of each element [K], shape (N,).

        Returns
        -------
        torch.Tensor
            Resistance [kΩ], shape (N,).
        """
        T = T.clamp(305, 370)
        return (self.R0 * torch.exp(self.Ea / T) * self.g(T) + self.Rm) / 1000


# ---------------------------------------------------------------------------
# 1-D Circuit
# ---------------------------------------------------------------------------

class Circuit:
    """
    1-D chain of VO₂ thermal neuristors with nearest-neighbour thermal
    coupling and stochastic noise.

    Integrates the coupled V–T ODEs using the explicit Euler–Maruyama scheme.
    Physical parameters are taken from Zhang et al. (2024).

    Parameters
    ----------
    batch : int
        Number of independent simulations run in parallel (mini-batch size).
    N : int
        Number of neuristors in the chain.
    V : float
        Default input voltage applied to all elements [V].
    R : float
        Load resistance [kΩ].
    noise_strength : float
        Standard deviation of the thermal noise term σ [µJ·s⁻¹/²].
        Converted to mW·ns⁻¹/² units internally.
    Cth_factor : float
        Multiplicative scaling of the baseline thermal capacitance
        (C_th = Cth_factor × 49.63 pJ/K).
    couple_factor : float
        Nearest-neighbour thermal coupling strength, relative to S_th.
    width_factor : float
        VO₂ hysteresis width scaling (passed to VO2).
    T_base : float, optional
        Ambient temperature [K].  Default 325 K.

    Notes
    -----
    Fixed physical constants (from experiment):
        C = 145.35 pF  (electrical capacitance)
        C_th = 49.63 pJ/K  (thermal capacitance, before Cth_factor)
        S_th = 0.2056 mW/K  (total thermal conductance)
    """

    def __init__(
        self,
        batch: int,
        N: int,
        V: float,
        R: float,
        noise_strength: float,
        Cth_factor: float,
        couple_factor: float,
        width_factor: float,
        T_base: float = 325,
    ):
        self.batch = batch
        self.N = N
        self.d = 1                    # Dimensionality (1 for chain, 2 for grid)

        # --- Circuit parameters ---
        self.V0 = V * torch.ones(self.batch, self.N)  # Input voltage [V]
        self.R0 = R                                   # Load resistance [kΩ]
        self.C0 = 145.34619293                        # Electrical capacitance [pF]
        self.R0C0 = self.R0 * self.C0                 # RC time constant [kΩ·pF = ns]

        # --- Thermal parameters ---
        self.Cth_factor = Cth_factor
        self.Cth = 49.62776831                        # Base thermal cap. [pJ/K]
        self.Sth = 0.20558726                         # Total thermal cond. [mW/K]
        self.couple_factor = couple_factor

        # Environmental and coupling conductances [mW/K]
        self.S_env = self.Sth * (1 - 2 * self.d * self.couple_factor)
        self.S_couple = self.couple_factor * self.Sth

        self.noise_strength = noise_strength          # [µJ·s⁻¹/²]
        self.width_factor = width_factor
        self.T_base = T_base                          # Ambient temperature [K]

        # --- VO₂ model (one per neuristor per batch element) ---
        self.VO2 = VO2(batch * N, width_factor)
        self.VO2.initialize(self.T_base - 0.1)       # Start just below IMT

        # Runtime state (filled during integration)
        self.IR = None            # Current [mA] through each element
        self.T = None
        self.R = None
        self.compiled_step = None

    def set_input(self, V: torch.Tensor = None, Cth_factor: torch.Tensor = None):
        """
        Update input voltage and/or thermal capacitance factor mid-simulation.

        Parameters
        ----------
        V : torch.Tensor, optional
            New input voltage array, shape (batch, N) [V].
        Cth_factor : torch.Tensor, optional
            New thermal capacitance factor, shape (batch, N).
        """
        if V is not None:
            self.V0 = V
        if Cth_factor is not None:
            self.Cth_factor = Cth_factor

    def dydt(self, t: float, y: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the right-hand side of the coupled V–T ODEs.

        Parameters
        ----------
        t : float
            Current simulation time [ns]  (unused — autonomous system).
        y : torch.Tensor
            State tensor, shape (batch, 2, N).
            y[:, 0, :] = nodal voltage V₁ [V]
            y[:, 1, :] = temperature T [K]

        Returns
        -------
        torch.Tensor
            Time derivatives dV/dt and dT/dt, shape (batch, 2, N).
        """
        V1 = y[:, 0, :]   # Nodal voltage  [V],   shape (batch, N)
        T  = y[:, 1, :]   # Temperature    [K],   shape (batch, N)

        # 1-D discrete Laplacian (Neumann BCs via edge padding)
        T_padded = torch.cat([T[:, :1], T, T[:, -1:]], dim=1)  # replicate edges
        laplacian = T_padded[:, :-2] - 2 * T + T_padded[:, 2:]

        # VO₂ resistance and Joule heating
        R   = self.VO2.R(T.reshape(-1)).reshape(self.batch, self.N)  # [kΩ]
        IR  = V1 / R                                                   # [mA]
        QR  = IR ** 2 * R                                              # [mW]
        self.IR = IR

        # Electrical ODE:  C dV/dt = (V_in − V)/R_load − V/R(T)
        dV1 = self.V0 / self.R0C0 - V1 / self.R0C0 - V1 / (R * self.C0)

        # Thermal ODE:  C_th dT/dt = QR − S_env(T−T₀) + S_couple ∇²T + σ ξ
        dT = (
            (QR - self.S_env * (T - self.T_base) + self.S_couple * laplacian)
            / self.Cth
            + self.noise_strength * torch.randn_like(T)
        ) / self.Cth_factor

        return torch.stack([dV1, dT], dim=1)

    def step(self, t: float, y: torch.Tensor) -> torch.Tensor:
        """
        Single Euler–Maruyama integration step.

        Checks for VO₂ hysteresis reversals *before* evaluating dydt, which
        is necessary for correct minor-loop tracking.

        Parameters
        ----------
        t : float
            Current time [ns].
        y : torch.Tensor
            Current state, shape (batch, 2, N).

        Returns
        -------
        torch.Tensor
            dy/dt evaluated at (t, y), shape (batch, 2, N).
        """
        self.VO2.reversal(y[:, 1].reshape(-1))
        return self.dydt(t, y)

    @torch.no_grad()
    def solve(
        self, y0: torch.Tensor, t_max: float, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate the ODE system from t=0 to t=t_max using Euler–Maruyama.

        Parameters
        ----------
        y0 : torch.Tensor
            Initial state, shape (batch, 2, N).
            Typically: V=0, T=T_base.
        t_max : float
            Simulation duration [ns].  Paper uses 10,000 ns (10 µs).
        dt : float
            Integration time step [ns].  Paper uses 10 ns.

        Returns
        -------
        y_final : torch.Tensor
            Final state at t=t_max, shape (batch, 2, N).
        I_traj : torch.Tensor
            Current trajectory for all neurons over all time steps,
            shape (batch, N, n_steps).  Used for temporal feature extraction.
        """
        t = 0.0
        y = y0
        n_max = int(t_max / dt)
        I_traj = []

        self.compiled_step = self.step  # torch.compile can be enabled here

        for _ in range(n_max):
            dy = self.compiled_step(t, y)
            t += dt
            y += dy * dt
            I_traj.append(self.IR.detach().clone())

        return y, torch.stack(I_traj, dim=-1)  # I_traj: (batch, N, n_steps)


# ---------------------------------------------------------------------------
# 2-D Circuit (used for MNIST)
# ---------------------------------------------------------------------------

class Circuit2D(Circuit):
    """
    2-D grid of VO₂ thermal neuristors with 4-connected nearest-neighbour
    thermal coupling.

    Extends ``Circuit`` by replacing the 1-D Laplacian with a 2-D discrete
    Laplacian on a rectangular Nx × Ny grid.  Each neuristor is electrically
    isolated (independent V–T dynamics) but thermally coupled to its four
    grid neighbours.

    This is the architecture used for MNIST classification: a 28×28 grid
    (Nx=Ny=28) maps directly onto the pixel array, with each pixel's
    intensity controlling its neuristor's input voltage.

    Parameters
    ----------
    batch : int
        Number of images processed in parallel.
    Nx : int
        Grid size along x (rows).  For MNIST: 28.
    Ny : int
        Grid size along y (columns).  For MNIST: 28.
    V, R, noise_strength, Cth_factor, couple_factor, width_factor, T_base :
        See ``Circuit`` documentation.
    """

    def __init__(
        self,
        batch: int,
        Nx: int,
        Ny: int,
        V: float,
        R: float,
        noise_strength: float,
        Cth_factor: float,
        couple_factor: float,
        width_factor: float,
        T_base: float = 325,
    ):
        N = Nx * Ny
        super().__init__(batch, N, V, R, noise_strength, Cth_factor,
                         couple_factor, width_factor, T_base)
        self.Nx = Nx
        self.Ny = Ny
        self.d = 2  # Override dimensionality
        # Recompute S_env with d=2 (4 neighbours vs. 2 for 1-D chain)
        self.S_env = self.Sth * (1 - 2 * self.d * self.couple_factor)

    def dydt(self, t: float, y: torch.Tensor) -> torch.Tensor:
        """
        ODE right-hand side for the 2-D grid.

        Computes the 2-D discrete Laplacian with replicate (Neumann) boundary
        conditions via ``torch.nn.functional.pad``.

        Parameters
        ----------
        t : float
            Current time [ns]  (unused).
        y : torch.Tensor
            State, shape (batch, 2, N) where N = Nx × Ny.

        Returns
        -------
        torch.Tensor
            dy/dt, shape (batch, 2, N).
        """
        V1 = y[:, 0, :]  # (batch, N)
        T  = y[:, 1, :]  # (batch, N)

        # Reshape to 2-D for Laplacian computation
        T_2D = T.view(self.batch, self.Nx, self.Ny)   # (batch, Nx, Ny)

        # Replicate-pad to enforce Neumann (zero-flux) BCs
        T_padded = nn.functional.pad(T_2D, (1, 1, 1, 1), mode='replicate')  # (batch, Nx+2, Ny+2)

        # 2-D discrete Laplacian: ∇²T ≈ T_{i-1,j} + T_{i+1,j} + T_{i,j-1} + T_{i,j+1} − 4T_{i,j}
        laplacian = (
            T_padded[:, :-2, 1:-1]   # up
            + T_padded[:, 2:,  1:-1] # down
            + T_padded[:, 1:-1, :-2] # left
            + T_padded[:, 1:-1, 2:]  # right
            - 4 * T_2D
        ).view(self.batch, self.N)    # Back to flat (batch, N)

        # Shared electrical + thermal ODE (same as 1-D, new laplacian)
        R  = self.VO2.R(T.reshape(-1)).reshape(self.batch, self.N)
        IR = V1 / R
        QR = IR ** 2 * R
        self.IR = IR

        dV1 = self.V0 / self.R0C0 - V1 / self.R0C0 - V1 / (R * self.C0)
        dT = (
            (QR - self.S_env * (T - self.T_base) + self.S_couple * laplacian)
            / self.Cth
            + self.noise_strength * torch.randn_like(T)
        ) / self.Cth_factor

        return torch.stack([dV1, dT], dim=1)


# ---------------------------------------------------------------------------
# Quick self-test (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Running VO₂ R(T) hysteresis self-test...")
    model = VO2(1)
    T_waypoints = torch.tensor([325, 360, 325, 360, 330, 360, 335, 360], dtype=torch.float)
    T_last = 324.9
    model.initialize(T_last)

    Rs, Ts = [], []
    for Ti in T_waypoints[1:]:
        T_sweep = torch.linspace(T_last, Ti, 100)
        for T_i in T_sweep:
            T_i = T_i.unsqueeze(0)
            model.reversal(T_i)
            Rs.append(model.R(T_i))
        T_last = Ti
        Ts.append(T_sweep)

    Rs = torch.cat(Rs).detach().numpy()
    Ts = torch.cat(Ts).detach().numpy()

    plt.figure(figsize=(5, 4))
    plt.plot(Ts, Rs)
    plt.yscale('log')
    plt.xlim([320, 360])
    plt.xlabel('Temperature (K)')
    plt.ylabel('Resistance (kΩ)')
    plt.title('VO₂ R(T) Hysteresis')
    plt.tight_layout()
    plt.savefig('results/VO2_R.png', dpi=300, bbox_inches='tight')
    print("Saved: results/VO2_R.png")
