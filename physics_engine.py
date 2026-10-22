"""
physics_engine.py
=================
Theoretical and Computational Engine for Sequential Stern-Gerlach Experiments.

Designed for Advanced Undergraduate Quantum Mechanics Courses & Viva Evaluation.

Authoritative Physics Content:
- SU(2) Spin-1/2 Formalism (Pauli Spin Matrices, Spinors)
- Arbitrary Measurement Axes in Spherical Coordinates (theta, phi)
- Projective Measurement (Von Neumann Collapse & Born Rule)
- Quantum Transition Probabilities & Spin-1/2 Malus's Law: P = cos^2(Delta_theta / 2)
- Bloch Sphere Vector Mapping and Density Matrix Support for Unpolarized Ensembles
- Monte Carlo Stochastic Atom Sampling with Statistical Convergence Verification

References:
1. J.J. Sakurai, "Modern Quantum Mechanics", Chapter 1 (Fundamental Concepts).
2. D.J. Griffiths, "Introduction to Quantum Mechanics", Chapter 4 (Quantum Mechanics in 3D).
3. R.P. Feynman, "The Feynman Lectures on Physics", Vol. III, Chapter 5 (Spin One-Half).
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np


# ==============================================================================
# SECTION 1: FUNDAMENTAL QUANTUM OPERATORS (PAULI MATRICES)
# ==============================================================================

# 2x2 Identity Matrix
SIGMA_0 = np.array([[1.0, 0.0],
                    [0.0, 1.0]], dtype=complex)

# Pauli X: Inverts spin along z, eigenstates are spin up/down along x
SIGMA_X = np.array([[0.0, 1.0],
                    [1.0, 0.0]], dtype=complex)

# Pauli Y: Associated with imaginary phase, eigenstates along y
SIGMA_Y = np.array([[0.0, -1.0j],
                    [1.0j, 0.0]], dtype=complex)

# Pauli Z: Diagonal in standard z-basis {| +z >, | -z >}
SIGMA_Z = np.array([[1.0, 0.0],
                    [0.0, -1.0]], dtype=complex)

# Natural units: hbar / 2 is factored out when working in dimensionless spin units.


# ==============================================================================
# SECTION 2: HELPER FUNCTIONS FOR SPHERICAL COORDINATES & UNIT VECTORS
# ==============================================================================

def spherical_to_cartesian(theta_rad: float, phi_rad: float) -> np.ndarray:
    """
    Computes unit orientation vector n_hat = (nx, ny, nz) from polar angles:
        nx = sin(theta) * cos(phi)
        ny = sin(theta) * sin(phi)
        nz = cos(theta)

    Parameters:
        theta_rad (float): Polar angle from +z axis [0, pi].
        phi_rad (float): Azimuthal angle in xy plane from +x axis [0, 2*pi].

    Returns:
        np.ndarray: 3-element unit vector [nx, ny, nz].
    """
    nx = np.sin(theta_rad) * np.cos(phi_rad)
    ny = np.sin(theta_rad) * np.sin(phi_rad)
    nz = np.cos(theta_rad)
    return np.array([nx, ny, nz], dtype=float)


def compute_spin_operator(theta_rad: float, phi_rad: float) -> np.ndarray:
    """
    Constructs the dimensionless spin operator sigma_n = sigma . n_hat:
        sigma_n = nx * sigma_x + ny * sigma_y + nz * sigma_z
                = [[ cos(theta),              e^(-i*phi) * sin(theta) ],
                   [ e^(i*phi) * sin(theta), -cos(theta)              ]]

    Parameters:
        theta_rad (float): Polar orientation angle [rad].
        phi_rad (float): Azimuthal orientation angle [rad].

    Returns:
        np.ndarray: 2x2 Hermitian matrix representing the measurement observable.
    """
    n = spherical_to_cartesian(theta_rad, phi_rad)
    sigma_n = n[0] * SIGMA_X + n[1] * SIGMA_Y + n[2] * SIGMA_Z
    return sigma_n


def compute_eigenstates(theta_rad: float, phi_rad: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the exact theoretical orthonormal eigenspinors for sigma . n_hat:
    
    1. Spin UP state along n_hat (| +n >):
       | +n > = cos(theta/2) | +z > + e^(i*phi) * sin(theta/2) | -z >
              = [ cos(theta/2),
                  e^(i*phi) * sin(theta/2) ]^T
                  
    2. Spin DOWN state along n_hat (| -n >):
       | -n > = sin(theta/2) | +z > - e^(i*phi) * cos(theta/2) | -z >
              = [ sin(theta/2),
                  -e^(i*phi) * cos(theta/2) ]^T

    Orthogonality condition check:
        < +n | -n > = cos(theta/2)*sin(theta/2) - (e^(-i*phi))*(e^(i*phi))*sin(theta/2)*cos(theta/2)
                    = 0 (Identically orthogonal for all theta, phi).

    Parameters:
        theta_rad (float): Polar angle in radians.
        phi_rad (float): Azimuthal angle in radians.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (| +n >, | -n >) as 2D column vectors (2, 1).
    """
    half_th = theta_rad / 2.0
    phase = np.exp(1.0j * phi_rad)

    plus_state = np.array([
        [np.cos(half_th)],
        [phase * np.sin(half_th)]
    ], dtype=complex)

    minus_state = np.array([
        [np.sin(half_th)],
        [-phase * np.cos(half_th)]
    ], dtype=complex)

    # Ensure rigorous normalization
    plus_state /= np.linalg.norm(plus_state)
    minus_state /= np.linalg.norm(minus_state)

    return plus_state, minus_state


# ==============================================================================
# SECTION 3: QUANTUM STATE REPRESENTATION (PURE & DENSITY MATRIX)
# ==============================================================================

class QuantumSpinState:
    """
    Represents a Spin-1/2 quantum state.
    Supports:
    - Pure states represented by a complex 2-component spinor |psi> = [alpha, beta]^T.
    - Statistical ensembles / mixed states represented by a 2x2 density matrix rho
      (such as the unpolarized thermal beam from an oven: rho = 0.5 * I).
    """

    def __init__(self, spinor: Optional[np.ndarray] = None,
                 density_matrix: Optional[np.ndarray] = None,
                 label: str = "State"):
        self.label = label

        if spinor is not None:
            # Pure State
            vec = np.array(spinor, dtype=complex).reshape((2, 1))
            norm = np.linalg.norm(vec)
            if norm == 0:
                raise ValueError("Spinor norm cannot be zero.")
            self.spinor: Optional[np.ndarray] = vec / norm
            self.rho: np.ndarray = self.spinor @ self.spinor.conj().T
            self.is_pure: bool = True
        elif density_matrix is not None:
            # Mixed State
            rho = np.array(density_matrix, dtype=complex).reshape((2, 2))
            tr = np.trace(rho)
            if abs(tr) > 1e-12:
                rho = rho / tr
            self.rho = rho
            # Test if pure state: Tr(rho^2) == 1
            purity = np.real(np.trace(self.rho @ self.rho))
            self.is_pure = bool(np.isclose(purity, 1.0, atol=1e-5))
            if self.is_pure:
                # Find dominant eigenvector
                eigvals, eigvecs = np.linalg.eigh(self.rho)
                idx = np.argmax(eigvals)
                self.spinor = eigvecs[:, idx].reshape((2, 1))
            else:
                self.spinor = None
        else:
            # Default to unpolarized state: rho = 0.5 * I
            self.rho = 0.5 * SIGMA_0
            self.spinor = None
            self.is_pure = False

    @classmethod
    def unpolarized(cls, label: str = "Unpolarized Beam") -> 'QuantumSpinState':
        """Creates an incoherent unpolarized beam: rho = 0.5 * I."""
        return cls(density_matrix=0.5 * SIGMA_0, label=label)

    @classmethod
    def from_polar_angles(cls, theta_rad: float, phi_rad: float,
                           spin_sign: str = '+', label: str = "Spin State") -> 'QuantumSpinState':
        """
        Creates an eigenspinor oriented along (theta, phi) for eigenvalue +1 or -1.
        """
        plus_state, minus_state = compute_eigenstates(theta_rad, phi_rad)
        chosen = plus_state if spin_sign == '+' else minus_state
        return cls(spinor=chosen, label=label)

    def bloch_vector(self) -> np.ndarray:
        """
        Computes the Cartesian coordinates on or inside the Bloch Sphere:
            r = ( <sigma_x>, <sigma_y>, <sigma_z> )
            where <sigma_k> = Tr(rho * sigma_k)

        For pure states, |r| = 1 (on surface).
        For totally unpolarized states, r = (0, 0, 0) (sphere origin).
        """
        rx = float(np.real(np.trace(self.rho @ SIGMA_X)))
        ry = float(np.real(np.trace(self.rho @ SIGMA_Y)))
        rz = float(np.real(np.trace(self.rho @ SIGMA_Z)))
        return np.array([rx, ry, rz], dtype=float)

    def transition_probability_to(self, target_state: 'QuantumSpinState') -> float:
        """
        Calculates Born rule transition probability P(this -> target):
            P = Tr(rho_source * Pi_target)
            where Pi_target = |target><target| (projection operator).
            
        For pure states, reduces to standard inner product squared:
            P = |< target | source >|^2
        """
        if target_state.spinor is not None:
            pi_target = target_state.spinor @ target_state.spinor.conj().T
        else:
            pi_target = target_state.rho

        prob = float(np.real(np.trace(self.rho @ pi_target)))
        return float(np.clip(prob, 0.0, 1.0))

    def spinor_components_display(self) -> Dict[str, complex]:
        """Returns alpha and beta components if pure state, else None."""
        if not self.is_pure or self.spinor is None:
            return {"alpha": complex(0.5, 0.0), "beta": complex(0.5, 0.0)}
        return {
            "alpha": self.spinor[0, 0],
            "beta": self.spinor[1, 0]
        }


# ==============================================================================
# SECTION 4: STERN-GERLACH APPARATUS MODEL
# ==============================================================================

class SternGerlachStage:
    """
    Represents a single Stern-Gerlach apparatus stage.
    
    Physics parameters:
    - Orientation: polar angle theta [rad] and azimuthal angle phi [rad].
    - Filter choice:
        '+'    : Passes only | +n > beam, blocks | -n > beam.
        '-'    : Passes only | -n > beam, blocks | +n > beam.
        'both' : Both beams pass unblocked (no filter).
    """

    def __init__(self, stage_id: int, name: str, theta_deg: float, phi_deg: float,
                 transmitted_beam: str = '+'):
        """
        Parameters:
            stage_id: Identifier index (1, 2, 3...)
            name: Human-friendly name (e.g. 'SG_1 (z-axis)')
            theta_deg: Polar angle in degrees [0, 180]
            phi_deg: Azimuthal angle in degrees [0, 360]
            transmitted_beam: '+' for spin up, '-' for spin down, 'both' for split without blocking
        """
        self.stage_id = stage_id
        self.name = name
        self.theta_deg = float(theta_deg)
        self.phi_deg = float(phi_deg)
        self.theta_rad = float(np.radians(theta_deg))
        self.phi_rad = float(np.radians(phi_deg))
        self.transmitted_beam = transmitted_beam

        # Compute eigenspinors along this orientation
        self.plus_spinor, self.minus_spinor = compute_eigenstates(self.theta_rad, self.phi_rad)
        self.plus_state = QuantumSpinState(spinor=self.plus_spinor, label=f"SG{stage_id} |+n>")
        self.minus_state = QuantumSpinState(spinor=self.minus_spinor, label=f"SG{stage_id} |-n>")

        # Projection operators: Pi_+ = |+n><+n|, Pi_- = |-n><-n|
        self.proj_plus = self.plus_spinor @ self.plus_spinor.conj().T
        self.proj_minus = self.minus_spinor @ self.minus_spinor.conj().T

        # Direction unit vector
        self.n_hat = spherical_to_cartesian(self.theta_rad, self.phi_rad)

    def analyze_incident_state(self, incident_state: QuantumSpinState) -> Dict[str, Any]:
        """
        Calculates measurement outcomes when an incident beam hits this SG magnet:
        - Born Rule Probability for Up: P(+) = Tr(rho * Pi_+)
        - Born Rule Probability for Down: P(-) = Tr(rho * Pi_-)
        - Collapse: Output state depends on transmitted_beam selection.
        """
        p_plus = float(np.real(np.trace(incident_state.rho @ self.proj_plus)))
        p_minus = float(np.real(np.trace(incident_state.rho @ self.proj_minus)))

        # Numerical cleanup
        p_plus = float(np.clip(p_plus, 0.0, 1.0))
        p_minus = float(np.clip(p_minus, 0.0, 1.0))

        # Check normalization
        total_p = p_plus + p_minus
        if abs(total_p - 1.0) > 1e-5 and total_p > 0:
            p_plus /= total_p
            p_minus /= total_p

        # Determine collapsed output state based on beam mask / blocker
        if self.transmitted_beam == '+':
            selected_prob = p_plus
            output_state = self.plus_state
        elif self.transmitted_beam == '-':
            selected_prob = p_minus
            output_state = self.minus_state
        else:  # 'both'
            selected_prob = 1.0
            # Both beams exit; for sequential cascade where both propagate incoherently
            output_state = incident_state

        return {
            "p_plus": p_plus,
            "p_minus": p_minus,
            "selected_prob": selected_prob,
            "output_state": output_state,
            "plus_state": self.plus_state,
            "minus_state": self.minus_state
        }


# ==============================================================================
# SECTION 5: SEQUENTIAL SIMULATION EXPERIMENT PIPELINE
# ==============================================================================

class SequentialSGExperiment:
    """
    Manages an arbitrary cascade of sequential Stern-Gerlach apparatuses
    (typically SG1 -> SG2 -> SG3).
    
    Provides:
    - Complete quantum state evolution through each filter.
    - Exact theoretical cumulative probabilities for every stage.
    - Verification of Malus's Law equivalent for spin-1/2 fermions:
        P(+ -> +) = cos^2(Delta_theta / 2)
    - Comparison between Classical Hidden-Variable expectation and Quantum Measurement.
    - Monte Carlo simulation of N particles traversing the apparatus cascade.
    """

    def __init__(self, initial_state: QuantumSpinState, stages: List[SternGerlachStage]):
        self.initial_state = initial_state
        self.stages = stages

    def run_analytical_simulation(self) -> Dict[str, Any]:
        """
        Propagates the quantum beam through all sequential stages.
        Returns detailed analytics for every intermediate stage.
        """
        current_state = self.initial_state
        cumulative_prob = 1.0
        stage_results = []

        for stage in self.stages:
            stage_eval = stage.analyze_incident_state(current_state)
            p_plus = stage_eval["p_plus"]
            p_minus = stage_eval["p_minus"]
            branch_prob = stage_eval["selected_prob"]
            next_state = stage_eval["output_state"]

            # Cumulative probability from the original incident beam
            cumulative_prob *= branch_prob

            stage_info = {
                "stage_id": stage.stage_id,
                "name": stage.name,
                "theta_deg": stage.theta_deg,
                "phi_deg": stage.phi_deg,
                "n_hat": stage.n_hat,
                "transmitted_beam": stage.transmitted_beam,
                "incident_state": current_state,
                "p_plus_conditional": p_plus,
                "p_minus_conditional": p_minus,
                "branch_selected_prob": branch_prob,
                "cumulative_prob": cumulative_prob,
                "output_state": next_state,
                "plus_state": stage_eval["plus_state"],
                "minus_state": stage_eval["minus_state"],
            }
            stage_results.append(stage_info)
            current_state = next_state

        return {
            "initial_state": self.initial_state,
            "stage_results": stage_results,
            "final_cumulative_prob": cumulative_prob,
            "final_state": current_state
        }

    def run_monte_carlo(self, num_atoms: int = 10000, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Performs a stochastic Monte Carlo simulation by sending 'num_atoms' individual
        silver atoms through the cascade.
        
        At each stage:
        1. A random number r in [0, 1] is drawn.
        2. If r < P(+), the atom collapses to | +n >; otherwise to | -n >.
        3. If the atom collapses to a blocked beam, it is filtered out (absorbed/lost).
        4. Atoms surviving all stages reach the final detector.
        
        Demonstrates the Law of Large Numbers (Monte Carlo ratio -> Theoretical Born probability).
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        analytics = self.run_analytical_simulation()
        stage_survivors: List[int] = []
        plus_counts_per_stage: List[int] = []
        minus_counts_per_stage: List[int] = []

        current_atoms = num_atoms

        for s_idx, stage_info in enumerate(analytics["stage_results"]):
            p_plus = stage_info["p_plus_conditional"]
            transmitted_beam = stage_info["transmitted_beam"]

            # Simulate each surviving atom entering this stage
            if current_atoms > 0:
                draws = rng.random(current_atoms)
                plus_count = int(np.sum(draws < p_plus))
                minus_count = current_atoms - plus_count
            else:
                plus_count = 0
                minus_count = 0

            plus_counts_per_stage.append(plus_count)
            minus_counts_per_stage.append(minus_count)

            # Filter surviving atoms
            if transmitted_beam == '+':
                current_atoms = plus_count
            elif transmitted_beam == '-':
                current_atoms = minus_count
            else:
                current_atoms = plus_count + minus_count

            stage_survivors.append(current_atoms)

        simulated_final_ratio = current_atoms / float(num_atoms) if num_atoms > 0 else 0.0
        theoretical_prob = analytics["final_cumulative_prob"]
        stat_error = np.abs(simulated_final_ratio - theoretical_prob)

        return {
            "num_atoms": num_atoms,
            "stage_survivors": stage_survivors,
            "plus_counts": plus_counts_per_stage,
            "minus_counts": minus_counts_per_stage,
            "final_detected_atoms": current_atoms,
            "simulated_ratio": simulated_final_ratio,
            "theoretical_prob": theoretical_prob,
            "absolute_error": stat_error,
            "expected_standard_error": np.sqrt(theoretical_prob * (1 - theoretical_prob) / num_atoms) if (num_atoms > 0 and 0 < theoretical_prob < 1) else 0.0
        }


# ==============================================================================
# SECTION 6: THEORETICAL ANALYSIS & PROFESSOR VIVA UTILITIES
# ==============================================================================

def explain_malus_law_spin_half(theta_1_deg: float, phi_1_deg: float,
                                theta_2_deg: float, phi_2_deg: float) -> Dict[str, Any]:
    """
    Computes and mathematically details the spin-1/2 analogue of Malus's Law.
    
    Formula:
        cos(Delta_theta) = n1 . n2
        P(+ -> +) = cos^2(Delta_theta / 2) = (1 + n1 . n2) / 2
        P(+ -> -) = sin^2(Delta_theta / 2) = (1 - n1 . n2) / 2
    
    Pedagogical Note for Viva:
        In classical optics (spin-1 photons), Malus's law is I = I0 * cos^2(theta).
        In quantum spin-1/2, the rotation angle appears as theta/2 because spinors
        transform under SU(2) with a 4*pi rotational symmetry (a 2*pi rotation produces -|psi>).
    """
    th1 = np.radians(theta_1_deg)
    ph1 = np.radians(phi_1_deg)
    th2 = np.radians(theta_2_deg)
    ph2 = np.radians(phi_2_deg)

    n1 = spherical_to_cartesian(th1, ph1)
    n2 = spherical_to_cartesian(th2, ph2)

    dot_product = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
    delta_angle_rad = float(np.arccos(dot_product))
    delta_angle_deg = float(np.degrees(delta_angle_rad))

    p_plus_plus = float(np.cos(delta_angle_rad / 2.0)**2)
    p_plus_minus = float(np.sin(delta_angle_rad / 2.0)**2)

    return {
        "n1": n1,
        "n2": n2,
        "dot_product": dot_product,
        "delta_angle_deg": delta_angle_deg,
        "delta_angle_rad": delta_angle_rad,
        "p_plus_plus": p_plus_plus,
        "p_plus_minus": p_plus_minus,
        "formula_latex": r"P(+ \rightarrow +) = \cos^2\left(\frac{\Delta\theta}{2}\right) = \frac{1 + \hat{n}_1 \cdot \hat{n}_2}{2}"
    }


def explain_canonical_experiment() -> Dict[str, str]:
    """
    Returns the foundational explanation for the canonical 3-stage experiment:
        SG_z (Pass +z) -> SG_x (Pass +x) -> SG_z (Test z-spin again)
        
    Addresses the core viva question:
    "Why does the spin-down state |-z> reappear at stage 3 if it was already
     completely blocked at stage 1?"
    """
    return {
        "title": "The Canonical 3-Stage Paradox: SG_z -> SG_x -> SG_z",
        "stage1_explanation": (
            "An unpolarized thermal beam enters SG_z. The beam splits 50/50 into |+z> and |-z>. "
            "We block the |-z> beam. At this instant, 100% of the surviving atoms are in the pure state |+z>."
        ),
        "stage2_explanation": (
            "The pure |+z> beam enters SG_x. Because [S_z, S_x] = i*hbar*S_y != 0, S_z and S_x do not commute. "
            "By basis transformation: |+z> = (1/sqrt(2)) |+x> + (1/sqrt(2)) |-x>. "
            "Measurement along x collapses each atom to |+x> (50%) or |-x> (50%). "
            "We transmit |+x>. This measurement completely erases and scrambles the prior S_z information!"
        ),
        "stage3_explanation": (
            "The beam entering SG_3 is now pure |+x>. In the z-basis: "
            "|+x> = (1/sqrt(2)) |+z> + (1/sqrt(2)) |-z>. "
            "When this beam enters SG_z for the second time, we measure S_z again. "
            "The probabilities are: P(+z) = |1/sqrt(2)|^2 = 50%, and P(-z) = |1/sqrt(2)|^2 = 50%!"
        ),
        "classical_vs_quantum": (
            "Classical Expectation: Zero atoms should be detected in |-z>, because stage 1 already filtered them out! "
            "Quantum Reality: 25% of the original oven atoms exit in |-z>! The intermediate non-commuting observation "
            "acted as a creative quantum filter, repopulating the orthogonal state through wave function collapse."
        )
    }
