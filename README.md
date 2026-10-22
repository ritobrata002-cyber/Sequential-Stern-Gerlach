# Sequential Stern-Gerlach Experiment Simulation

An interactive, production-ready simulation of the **Sequential Stern-Gerlach Experiment** designed for advanced undergraduate quantum mechanics courses, laboratory demonstrations, and oral examination (viva voce) preparation.

Built with **Python 3**, **NumPy**, **Matplotlib**, and **Streamlit**.

---

## 🌟 Key Features

1. **Rigorous $SU(2)$ Quantum Formalism:**
   - 2-component complex spinors $\begin{pmatrix} \alpha \\ \beta \end{pmatrix}$ and Pauli spin matrices ($\sigma_x, \sigma_y, \sigma_z$).
   - General spin operator along an arbitrary orientation vector $\hat{n}(\theta, \phi)$:
     $$\vec{S} \cdot \hat{n} = \frac{\hbar}{2} \begin{pmatrix} \cos\theta & e^{-i\phi}\sin\theta \\ e^{i\phi}\sin\theta & -\cos\theta \end{pmatrix}$$
   - Exact orthonormal eigenspinors $|+\hat{n}\rangle$ and $|-\hat{n}\rangle$ with correct complex phase relations.
   - Density matrix formalism ($\rho$) supporting both pure states ($\text{Tr}(\rho^2) = 1$) and unpolarized thermal ensembles ($\rho = \frac{1}{2} I$).

2. **Sequential Multi-Stage Cascade:**
   - Configurable 2-stage or 3-stage apparatuses with independent polar ($\theta$) and azimuthal ($\phi$) angles.
   - Selectable beam filtering: transmit $|+\hat{n}\rangle$ (spin up), transmit $|-\hat{n}\rangle$ (spin down), or pass both branches.
   - Exact analytical calculation of conditional and cumulative branch probabilities using projection operators $\hat{\Pi}_\pm = |\pm\hat{n}\rangle\langle\pm\hat{n}|$.
   - Explicit proof of the **Spin-1/2 Malus's Law**:
     $$P(+ \rightarrow +) = \cos^2\left(\frac{\Delta\theta}{2}\right) = \frac{1 + \hat{n}_1 \cdot \hat{n}_2}{2}$$

3. **Rich Visualizations:**
   - **2D Beam-Splitting Flowchart:** Publication-quality schematic displaying the atomic oven, magnetic pole pieces ($N/S$ knife edges), beam trajectories, absorption blocks, and live intensity percentages.
   - **3D Bloch Sphere Visualizer:** 3D sphere showing the evolution of the state vector $\vec{r} = (\langle\sigma_x\rangle, \langle\sigma_y\rangle, \langle\sigma_z\rangle)$ and the measurement axis $\hat{n}$ across all stages.
   - **Real-time Probability Charts:** Bar graphs comparing conditional outcomes across stages.

4. **Stochastic Monte Carlo Particle Engine:**
   - Simulates $N$ individual silver atoms ($N = 500$ to $50,000$) traversing the apparatus cascade.
   - Live measurement collapse demonstrating the **Law of Large Numbers** and statistical convergence to the Born rule ($1/\sqrt{N}$ error bounds).

5. **Oral Exam & Viva Defense Guide:**
   - In-depth theoretical explanations addressing non-commuting observables $[S_z, S_x] = i\hbar S_y$, spatial quantization, magnetic gradients, and mixed vs. pure states.

---

## 📚 Theoretical Background

### 1. Physical Principle of the Stern-Gerlach Experiment (1922)
When a neutral silver atom ($^{107}\text{Ag}$ or $^{109}\text{Ag}$) travels through a magnetic field $\vec{B}$, its magnetic dipole moment $\vec{\mu}$ experiences a potential energy:
$$U = -\vec{\mu} \cdot \vec{B}$$

The net force acting on the atom is the spatial gradient of this potential:
$$\vec{F} = -\nabla U = \nabla(\vec{\mu} \cdot \vec{B})$$

In an **inhomogeneous** magnetic field with a strong vertical gradient $\frac{\partial B_z}{\partial z}$:
$$F_z \approx \mu_z \frac{\partial B_z}{\partial z} = -g_s \mu_B m_s \frac{\partial B_z}{\partial z}$$

Classically, dipoles are randomly oriented, predicting a continuous vertical smear on the detector screen. Instead, Stern and Gerlach observed **two discrete spots**, proving the quantization of space and the existence of an intrinsic electron spin of $S = 1/2$ ($m_s = \pm 1/2$).

---

### 2. The Sequential Paradox: $SG_z \rightarrow SG_x \rightarrow SG_z$

Consider the canonical 3-stage setup:
1. **Stage 1 ($SG_z$):** An unpolarized beam from the oven splits into $|+z\rangle$ ($50\%$) and $|-z\rangle$ ($50\%$). The $|-z\rangle$ beam is blocked. Surviving atoms are in the pure state $|+z\rangle$.
2. **Stage 2 ($SG_x$):** The $|+z\rangle$ atoms enter a magnet oriented along $\hat{x}$. In the $S_x$ eigenbasis:
   $$|+z\rangle = \frac{1}{\sqrt{2}} |+x\rangle + \frac{1}{\sqrt{2}} |-x\rangle$$
   Measurement collapses each atom into $|+x\rangle$ ($50\%$) or $|-x\rangle$ ($50\%$). We transmit only $|+x\rangle$.
3. **Stage 3 ($SG_z$):** The $|+x\rangle$ atoms enter a third magnet oriented along $\hat{z}$ again. Expanding in the $S_z$ basis:
   $$|+x\rangle = \frac{1}{\sqrt{2}} |+z\rangle + \frac{1}{\sqrt{2}} |-z\rangle$$
   Measuring $S_z$ produces:
   $$P(+z) = \left|\frac{1}{\sqrt{2}}\right|^2 = 50\%, \quad P(-z) = \left|\frac{1}{\sqrt{2}}\right|^2 = 50\%$$

**The Quantum Surprise:** Even though all spin-down atoms were eliminated in Stage 1, $25\%$ of the initial atoms emerge from Stage 3 with spin-down! Because $[S_z, S_x] \ne 0$, the measurement of $S_x$ destroys the prior $S_z$ information, resetting the wave function.

---

### 3. Spin-1/2 Malus's Law
For two consecutive Stern-Gerlach apparatuses oriented at unit vectors $\hat{n}_1$ and $\hat{n}_2$ separated by angle $\Delta\theta = \arccos(\hat{n}_1 \cdot \hat{n}_2)$, the probability of an atom transmitting through both filters is:
$$P(+ \rightarrow +) = |\langle +\hat{n}_2 | +\hat{n}_1 \rangle|^2 = \cos^2\left(\frac{\Delta\theta}{2}\right)$$

*Note on Half-Angles:* Unlike classical polarized light (spin-1 photons) which follows $\cos^2(\Delta\theta)$, spin-1/2 particles transform under the fundamental representation of $SU(2)$, requiring a $4\pi$ spatial rotation to return to their initial state.

---

## 📁 Repository Structure

```
Sequential Stern Gerlach/
│
├── physics_engine.py      # Core quantum mechanics engine:
│                          # - Pauli matrices & arbitrary spin operators
│                          # - Analytic eigenspinor solvers
│                          # - QuantumSpinState (pure spinors & density matrices)
│                          # - SternGerlachStage & SequentialSGExperiment
│                          # - Monte Carlo stochastic sampling engine
│
├── visualizer.py          # Visualization routines:
│                          # - 2D publication-grade experiment schematic
│                          # - 3D Bloch sphere with state & measurement vectors
│                          # - Stage probability bar charts
│                          # - Monte Carlo statistical histograms
│
├── app.py                 # Interactive Streamlit web application:
│                          # - Preset scenarios & custom slider controls
│                          # - Live probability readouts & LaTeX matrix derivations
│                          # - 3D Bloch sphere gallery & Monte Carlo statistics
│
├── requirements.txt       # Project dependencies (numpy, matplotlib, streamlit)
└── README.md              # Complete theoretical & practical documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
Ensure you have **Python 3.8+** installed.

### 2. Install Dependencies
Open your terminal in the project directory and install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Launch the Application
Run the Streamlit application:

```bash
streamlit run app.py
```

The application will launch automatically in your default browser at `http://localhost:8501`.

---

## 🧪 Experiments to Try in the Dashboard

1. **The Canonical Paradox:**
   - Select preset: `Canonical Paradox: SG(z) -> SG(x) -> SG(z)`
   - Observe how $25\%$ of the total initial beam survives at the final detector in state $|+z\rangle$, and note that an identical $25\%$ would emerge in $|-z\rangle$ if the filter were inverted.
2. **Quantum Repeatability (Non-Demolition):**
   - Select preset: `Quantum Repeatability: SG(z) -> SG(z) -> SG(z)`
   - Verify that once a pure state $|+z\rangle$ is prepared, subsequent measurements along $\hat{z}$ yield $100\%$ transmission with zero collapse disturbance.
3. **Continuous Angle Verification (Malus's Law):**
   - Select preset: `Malus's Law Continuous Sweep: SG(z) -> SG(θ) -> SG(z)`
   - Vary the $\theta_2$ slider from $0^\circ$ to $180^\circ$ and observe the smooth $\cos^2(\theta/2)$ curve.
4. **Law of Large Numbers (Monte Carlo):**
   - Switch to the **Monte Carlo Particle Statistics** tab.
   - Adjust $N$ from $500$ to $50,000$ and observe the statistical fluctuation $|P_{sim} - P_{theo}|$ decrease as $\mathcal{O}(1/\sqrt{N})$.

---

## 🎓 Viva Voce (Oral Examination) Questions & Answers

| # | Question | Key Physics Concept |
|---|----------|---------------------|
| 1 | Why is a magnetic field gradient $\frac{\partial B_z}{\partial z}$ required? | Uniform field yields only torque ($\vec{\mu}\times\vec{B}$); net translational force requires gradient: $\vec{F} = \nabla(\vec{\mu}\cdot\vec{B})$. |
| 2 | Why did Stern and Gerlach use neutral silver atoms? | Outer shell has a single $5s^1$ electron ($l=0$). Entire magnetic dipole moment is purely due to electron spin $s=1/2$. |
| 3 | Why does the intermediate magnet in $SG_z \to SG_x \to SG_z$ revive $|-z\rangle$? | Non-commuting observables $[S_z, S_x] = i\hbar S_y$. The intermediate measurement collapses the state into $|+x\rangle$, which has equal $50\%$ projections onto both $|+z\rangle$ and $|-z\rangle$. |
| 4 | What is the distinction between an unpolarized beam and a superposition? | Unpolarized beam is a mixed state ($\rho = \frac{1}{2}I$, $\text{Tr}(\rho^2) = 1/2$); pure superposition is coherent ($\text{Tr}(\rho^2) = 1$) with deterministic outcome along its eigenaxis. |
| 5 | Why is Malus's Law $\cos^2(\theta/2)$ instead of $\cos^2\theta$? | Spinors live in $SU(2)$, requiring a $4\pi$ spatial rotation to return to their initial state ($e^{-i \theta \sigma / 2}$). |

---

## 📜 References
- J.J. Sakurai & J. Napolitano, *Modern Quantum Mechanics* (2nd/3rd Ed.), Chapter 1.
- D.J. Griffiths & D.F. Schroeter, *Introduction to Quantum Mechanics* (3rd Ed.), Chapter 4.
- R.P. Feynman, R.B. Leighton, & M. Sands, *The Feynman Lectures on Physics*, Vol. III, Chapter 5.
