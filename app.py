"""
app.py
======
Interactive Sequential Stern-Gerlach Experiment Simulation.
Production-ready Web Dashboard for Advanced Undergraduate Physics Evaluation & Viva.

Tech Stack: Streamlit, NumPy, Matplotlib.
Run Command: streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Import custom theoretical physics and visualization engines
from physics_engine import (
    QuantumSpinState,
    SternGerlachStage,
    SequentialSGExperiment,
    explain_malus_law_spin_half,
    SIGMA_X, SIGMA_Y, SIGMA_Z, SIGMA_0
)
from visualizer import (
    plot_experiment_schematic,
    plot_bloch_sphere,
    plot_probability_bars,
    plot_monte_carlo_histogram
)

# ==============================================================================
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# ==============================================================================

st.set_page_config(
    page_title="Sequential Stern-Gerlach Simulation | Quantum Mechanics",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern scientific dark theme CSS
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0b1120;
        color: #f1f5f9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 50%, #082f49 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 0px;
    }

    /* Metric Card */
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #38bdf8;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-caption {
        color: #64748b;
        font-size: 0.75rem;
        margin-top: 2px;
    }

    /* Concept & Formula Boxes */
    .concept-box {
        background-color: #0f172a;
        border-left: 4px solid #38bdf8;
        border-radius: 6px;
        padding: 14px 18px;
        margin: 12px 0;
    }
    
    /* Code/Math styling */
    .math-inline {
        background-color: #1e293b;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: monospace;
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. SIDEBAR EXPERIMENT CONTROLS
# ==============================================================================

with st.sidebar:
    st.markdown("## ⚛️ Experiment Controls")
    st.markdown("Configure the magnetic stages and beam parameters.")

    # 1. Preset Experiments Selector
    preset_choice = st.selectbox(
        "📚 Standard Configurations:",
        [
            "Canonical Paradox: SG(z) -> SG(x) -> SG(z)",
            "Quantum Repeatability: SG(z) -> SG(z) -> SG(z)",
            "Orthogonal Sequence: SG(z) -> SG(x) -> SG(y)",
            "Malus's Law Continuous Sweep: SG(z) -> SG(θ) -> SG(z)",
            "Custom Configuration"
        ],
        index=0
    )

    st.markdown("---")

    # 2. Initial State Selection
    st.markdown("### 1. Initial Atomic Beam")
    init_state_type = st.radio(
        "Source Type:",
        ["Unpolarized Thermal Beam (Oven)", "Prepared Pure State |+z⟩", "Custom Spinor (θ₀, ϕ₀)"],
        index=0
    )

    init_theta, init_phi = 0.0, 0.0
    if init_state_type == "Custom Spinor (θ₀, ϕ₀)":
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            init_theta = st.slider("θ₀ (°)", 0.0, 180.0, 0.0, 5.0)
        with c_col2:
            init_phi = st.slider("ϕ₀ (°)", 0.0, 360.0, 0.0, 5.0)

    st.markdown("---")

    # 3. Stages Configuration
    st.markdown("### 2. Sequential Magnetic Stages")
    num_stages = st.radio("Number of Stages:", [2, 3], index=1, horizontal=True)

    # Preset logic to populate defaults
    default_th1, default_ph1, default_flt1 = 0.0, 0.0, '+'
    default_th2, default_ph2, default_flt2 = 90.0, 0.0, '+'
    default_th3, default_ph3, default_flt3 = 0.0, 0.0, '+'

    if preset_choice == "Canonical Paradox: SG(z) -> SG(x) -> SG(z)":
        default_th1, default_ph1, default_flt1 = 0.0, 0.0, '+'
        default_th2, default_ph2, default_flt2 = 90.0, 0.0, '+'
        default_th3, default_ph3, default_flt3 = 0.0, 0.0, '+'
    elif preset_choice == "Quantum Repeatability: SG(z) -> SG(z) -> SG(z)":
        default_th1, default_ph1, default_flt1 = 0.0, 0.0, '+'
        default_th2, default_ph2, default_flt2 = 0.0, 0.0, '+'
        default_th3, default_ph3, default_flt3 = 0.0, 0.0, '+'
    elif preset_choice == "Orthogonal Sequence: SG(z) -> SG(x) -> SG(y)":
        default_th1, default_ph1, default_flt1 = 0.0, 0.0, '+'
        default_th2, default_ph2, default_flt2 = 90.0, 0.0, '+'
        default_th3, default_ph3, default_flt3 = 90.0, 90.0, '+'
    elif preset_choice == "Malus's Law Continuous Sweep: SG(z) -> SG(θ) -> SG(z)":
        default_th1, default_ph1, default_flt1 = 0.0, 0.0, '+'
        default_th2, default_ph2, default_flt2 = 45.0, 0.0, '+'
        default_th3, default_ph3, default_flt3 = 0.0, 0.0, '+'

    # STAGE 1
    with st.expander("📍 Stage 1: SG-1 Parameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            th1 = st.slider("θ₁ (Polar, °)", 0.0, 180.0, default_th1, 5.0, key="th1")
        with col2:
            ph1 = st.slider("ϕ₁ (Azimuth, °)", 0.0, 360.0, default_ph1, 5.0, key="ph1")
        flt1 = st.selectbox("Beam Transmitted:", ["|+n₁⟩ (Pass Up)", "|-n₁⟩ (Pass Down)", "Both (Unfiltered)"],
                            index=0 if default_flt1 == '+' else 1, key="flt1")
        flt1_code = '+' if "Pass Up" in flt1 else ('-' if "Pass Down" in flt1 else 'both')

    # STAGE 2
    with st.expander("📍 Stage 2: SG-2 Parameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            th2 = st.slider("θ₂ (Polar, °)", 0.0, 180.0, default_th2, 5.0, key="th2")
        with col2:
            ph2 = st.slider("ϕ₂ (Azimuth, °)", 0.0, 360.0, default_ph2, 5.0, key="ph2")
        flt2 = st.selectbox("Beam Transmitted:", ["|+n₂⟩ (Pass Up)", "|-n₂⟩ (Pass Down)", "Both (Unfiltered)"],
                            index=0 if default_flt2 == '+' else 1, key="flt2")
        flt2_code = '+' if "Pass Up" in flt2 else ('-' if "Pass Down" in flt2 else 'both')

    # STAGE 3 (If enabled)
    th3, ph3, flt3_code = default_th3, default_ph3, '+'
    if num_stages == 3:
        with st.expander("📍 Stage 3: SG-3 Parameters", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                th3 = st.slider("θ₃ (Polar, °)", 0.0, 180.0, default_th3, 5.0, key="th3")
            with col2:
                ph3 = st.slider("ϕ₃ (Azimuth, °)", 0.0, 360.0, default_ph3, 5.0, key="ph3")
            flt3 = st.selectbox("Beam Transmitted:", ["|+n₃⟩ (Pass Up)", "|-n₃⟩ (Pass Down)", "Both (Unfiltered)"],
                                index=0 if default_flt3 == '+' else 1, key="flt3")
            flt3_code = '+' if "Pass Up" in flt3 else ('-' if "Pass Down" in flt3 else 'both')

    st.markdown("---")

    # 4. Monte Carlo Settings
    st.markdown("### 3. Stochastic Sampling")
    num_atoms = st.slider("Silver Atoms (N):", 500, 50000, 10000, 500)
    resample_btn = st.button("🎲 Resample Monte Carlo Run", use_container_width=True)


# ==============================================================================
# 3. BUILD QUANTUM SYSTEM & RUN SIMULATION
# ==============================================================================

# Construct initial quantum state
if init_state_type == "Unpolarized Thermal Beam (Oven)":
    initial_quantum_state = QuantumSpinState.unpolarized(label="Unpolarized Oven Beam")
elif init_state_type == "Prepared Pure State |+z⟩":
    initial_quantum_state = QuantumSpinState(spinor=np.array([[1.0], [0.0]]), label="Prepared |+z>")
else:
    initial_quantum_state = QuantumSpinState.from_polar_angles(
        np.radians(init_theta), np.radians(init_phi), spin_sign='+', label="Custom Initial State"
    )

# Construct stages
stages = [
    SternGerlachStage(1, "SG-1", th1, ph1, flt1_code),
    SternGerlachStage(2, "SG-2", th2, ph2, flt2_code)
]
if num_stages == 3:
    stages.append(SternGerlachStage(3, "SG-3", th3, ph3, flt3_code))

# Run analytical quantum propagation
experiment = SequentialSGExperiment(initial_quantum_state, stages)
sim_results = experiment.run_analytical_simulation()

# Run Monte Carlo particle simulation
mc_results = experiment.run_monte_carlo(num_atoms=num_atoms, seed=None if resample_btn else 42)


# ==============================================================================
# 4. MAIN DASHBOARD CONTENT
# ==============================================================================

# Hero Header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Sequential Stern-Gerlach Quantum Simulation</div>
    <div class="hero-subtitle">
        Rigorous SU(2) Spinor Analysis &bull; Projective Von Neumann Collapse &bull; Spin-1/2 Malus's Law
    </div>
</div>
""", unsafe_allow_html=True)

# Top Metrics Row
final_prob = sim_results["final_cumulative_prob"]
malus_info = explain_malus_law_spin_half(th1, ph1, th2, ph2)
delta_th = malus_info["delta_angle_deg"]

mcol1, mcol2, mcol3, mcol4 = st.columns(4)

with mcol1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Final Beam Transmission</div>
        <div class="metric-value">{final_prob * 100:.2f}%</div>
        <div class="metric-caption">Cumulative intensity reaching detector</div>
    </div>
    """, unsafe_allow_html=True)

with mcol2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Inter-Stage Angle (Δθ₁₂)</div>
        <div class="metric-value">{delta_th:.1f}°</div>
        <div class="metric-caption">Angle between n̂₁ and n̂₂ vectors</div>
    </div>
    """, unsafe_allow_html=True)

with mcol3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Malus P(+ → +) Factor</div>
        <div class="metric-value">{malus_info['p_plus_plus'] * 100:.1f}%</div>
        <div class="metric-caption">cos²(Δθ₁₂ / 2) overlap</div>
    </div>
    """, unsafe_allow_html=True)

with mcol4:
    detected_count = mc_results["final_detected_atoms"]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Detected Atoms (N)</div>
        <div class="metric-value">{detected_count:,}</div>
        <div class="metric-caption">Out of {num_atoms:,} sampled atoms</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# Main Tabs Navigation
tab_schematic, tab_math, tab_bloch, tab_mc = st.tabs([
    "🔬 1. Experiment Schematic & Beams",
    "📐 2. Quantum Formalism & Matrices",
    "🌐 3. 3D Bloch Sphere Visualizer",
    "🎲 4. Monte Carlo Particle Statistics"
])


# ==============================================================================
# TAB 1: EXPERIMENT SCHEMATIC & BEAMS
# ==============================================================================
with tab_schematic:
    st.markdown("### Sequential Beam Splitting & Filter Flowchart")
    st.markdown("""
    The diagram below models the physical trajectory of the atomic silver beam. 
    Each magnet produces an inhomogeneous magnetic gradient $\\nabla B$, splitting the incident beam into 
    spatial components corresponding to eigenvalues of $\\vec{S} \\cdot \\hat{n}$.
    """)

    # Render schematic figure
    fig_schematic = plot_experiment_schematic(sim_results)
    st.pyplot(fig_schematic, use_container_width=True)
    plt.close(fig_schematic)

    st.markdown("#### Stage-by-Stage Detailed Branching")
    
    col_table, col_bar = st.columns([1.1, 0.9])
    
    with col_table:
        table_rows = []
        for s in sim_results["stage_results"]:
            flt_label = "Pass |+n⟩" if s['transmitted_beam'] == '+' else ("Pass |-n⟩" if s['transmitted_beam'] == '-' else "Both")
            table_rows.append({
                "Stage": f"SG-{s['stage_id']} ({s['name']})",
                "Orientation (θ, ϕ)": f"({s['theta_deg']:.0f}°, {s['phi_deg']:.0f}°)",
                "P(+|in)": f"{s['p_plus_conditional'] * 100:.2f}%",
                "P(-|in)": f"{s['p_minus_conditional'] * 100:.2f}%",
                "Filter Setting": flt_label,
                "Cumulative Output": f"{s['cumulative_prob'] * 100:.2f}%"
            })
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

        st.info("""
        💡 **Key Quantum Observation:** In a sequential filter setup like $SG_z \\to SG_x \\to SG_z$, 
        the classical intuition suggests zero atoms should exit $SG_3$ in state $|-z\\rangle$. 
        However, the intermediate measurement along $\\hat{x}$ projects the state onto $|+x\\rangle = \\frac{1}{\\sqrt{2}}(|+z\\rangle + |-z\\rangle)$, 
        re-introducing both $z$-components with equal 50% amplitude!
        """)

    with col_bar:
        fig_bars = plot_probability_bars(sim_results["stage_results"])
        st.pyplot(fig_bars, use_container_width=True)
        plt.close(fig_bars)


# ==============================================================================
# TAB 2: QUANTUM FORMALISM & MATRICES
# ==============================================================================
with tab_math:
    st.markdown("### Mathematical Formalism & Exact Matrix Algebra")
    st.markdown("""
    The Stern-Gerlach apparatus measures the spin observable along a unit vector $\\hat{n}(\\theta, \\phi)$:
    $$\\hat{n} = (\\sin\\theta \\cos\\phi, \\sin\\theta \\sin\\phi, \\cos\\theta)$$
    The corresponding spin operator is given by the matrix:
    $$\\vec{S} \\cdot \\hat{n} = \\frac{\\hbar}{2} \\vec{\\sigma} \\cdot \\hat{n} = \\frac{\\hbar}{2} \\begin{pmatrix} \\cos\\theta & e^{-i\\phi}\\sin\\theta \\\\ e^{i\\phi}\\sin\\theta & -\\cos\\theta \\end{pmatrix}$$
    """)

    st.markdown("---")

    # Step-by-step breakdown of each stage
    for idx, s in enumerate(sim_results["stage_results"]):
        st.markdown(f"#### 📍 Stage {s['stage_id']}: Orientation $\\hat{{n}}_{s['stage_id']}$ at $(\\theta={s['theta_deg']:.1f}^\\circ, \\phi={s['phi_deg']:.1f}^\\circ)$")
        
        m_col1, m_col2 = st.columns(2)
        
        with m_col1:
            st.markdown("**Eigenspinors in Standard $z$-basis:**")
            th_r = np.radians(s['theta_deg'])
            ph_r = np.radians(s['phi_deg'])
            plus_spinor = s['plus_state'].spinor
            minus_spinor = s['minus_state'].spinor
            
            p_a, p_b = plus_spinor[0, 0], plus_spinor[1, 0]
            m_a, m_b = minus_spinor[0, 0], minus_spinor[1, 0]
            
            st.latex(rf"""
            | +n \rangle = \begin{{pmatrix}} \cos(\theta/2) \\ e^{{i\phi}} \sin(\theta/2) \end{{pmatrix}} 
            = \begin{{pmatrix}} {p_a.real:.4f} + {p_a.imag:.4f}i \\ {p_b.real:.4f} + {p_b.imag:.4f}i \end{{pmatrix}}
            """)
            st.latex(rf"""
            | -n \rangle = \begin{{pmatrix}} \sin(\theta/2) \\ -e^{{i\phi}} \cos(\theta/2) \end{{pmatrix}}
            = \begin{{pmatrix}} {m_a.real:.4f} + {m_a.imag:.4f}i \\ {m_b.real:.4f} + {m_b.imag:.4f}i \end{{pmatrix}}
            """)

        with m_col2:
            st.markdown("**Projection Operators & Measurement Probabilities:**")
            st.markdown(f"- Projection Operator for Up: $\\hat{{\\Pi}}_+ = |+n\\rangle \\langle +n|$")
            st.markdown(f"- Projection Operator for Down: $\\hat{{\\Pi}}_- = |-n\\rangle \\langle -n|$")
            
            st.latex(rf"""
            P(+) = \text{{Tr}}(\rho_{{\text{{in}}}} \hat{{\Pi}}_+) = {s['p_plus_conditional'] * 100:.2f}\%
            """)
            st.latex(rf"""
            P(-) = \text{{Tr}}(\rho_{{\text{{in}}}} \hat{{\Pi}}_-) = {s['p_minus_conditional'] * 100:.2f}\%
            """)
            st.markdown(f"**State after filter:** Transmitted `{s['transmitted_beam']}` $\\implies$ Collapsed into $|\\psi_{{out}}\\rangle$.")

        st.markdown("---")

    # Spin-1/2 Malus's Law Deep-Dive
    st.markdown("### 🔬 Malus's Law Equivalent for Spin-1/2 Fermions")
    st.markdown("""
    When an atom prepared in an eigenspinor $|+\\hat{n}_1\\rangle$ enters a second magnet oriented along $\\hat{n}_2$, 
    the transition probability is given by the square of the inner product:
    $$P(+ \\to +) = |\\langle +\\hat{n}_2 | +\\hat{n}_1 \\rangle|^2 = \\cos^2\\left(\\frac{\\Delta\\theta}{2}\\right) = \\frac{1 + \\hat{n}_1 \\cdot \\hat{n}_2}{2}$$
    $$P(+ \\to -) = |\\langle -\\hat{n}_2 | +\\hat{n}_1 \\rangle|^2 = \\sin^2\\left(\\frac{\\Delta\\theta}{2}\\right) = \\frac{1 - \\hat{n}_1 \\cdot \\hat{n}_2}{2}$$
    """)

    m_eval1, m_eval2 = st.columns(2)
    with m_eval1:
        st.write(f"**Current $\\hat{{n}}_1$ (Stage 1):** `[{malus_info['n1'][0]:.3f}, {malus_info['n1'][1]:.3f}, {malus_info['n1'][2]:.3f}]`")
        st.write(f"**Current $\\hat{{n}}_2$ (Stage 2):** `[{malus_info['n2'][0]:.3f}, {malus_info['n2'][1]:.3f}, {malus_info['n2'][2]:.3f}]`")
        st.write(f"**Dot Product ($\\hat{{n}}_1 \\cdot \\hat{{n}}_2$):** `{malus_info['dot_product']:.4f}`")
        st.write(f"**Relative Angle ($\\Delta\\theta$):** `{malus_info['delta_angle_deg']:.2f}^\\circ`")
    with m_eval2:
        st.latex(rf"P(+ \to +) = \cos^2\left(\frac{{{malus_info['delta_angle_deg']:.1f}^\circ}}{{2}}\right) = {malus_info['p_plus_plus']:.4f} \implies {malus_info['p_plus_plus']*100:.2f}\%")
        st.latex(rf"P(+ \to -) = \sin^2\left(\frac{{{malus_info['delta_angle_deg']:.1f}^\circ}}{{2}}\right) = {malus_info['p_plus_minus']:.4f} \implies {malus_info['p_plus_minus']*100:.2f}\%")


# ==============================================================================
# TAB 3: 3D BLOCH SPHERE VISUALIZER
# ==============================================================================
with tab_bloch:
    st.markdown("### 3D Bloch Sphere Evolution")
    st.markdown("""
    The Bloch Sphere maps the two-dimensional complex Hilbert space $\\mathbb{C}^2$ of a spin-1/2 particle 
    onto the surface of a three-dimensional unit ball $S^2$.
    The coordinates of the state vector are given by expectation values:
    $$\\vec{r} = (\\langle \\sigma_x \\rangle, \\langle \\sigma_y \\rangle, \\langle \\sigma_z \\rangle)$$
    """)

    b_cols = st.columns(len(sim_results["stage_results"]) + 1)

    # 1. Initial State Bloch Sphere
    with b_cols[0]:
        init_r = initial_quantum_state.bloch_vector()
        fig_b_init = plot_bloch_sphere(init_r, title="Initial State")
        st.pyplot(fig_b_init, use_container_width=True)
        plt.close(fig_b_init)
        st.caption(f"**Initial:** $\\vec{{r}} = [{init_r[0]:.2f}, {init_r[1]:.2f}, {init_r[2]:.2f}]$")

    # 2. Stage-by-Stage Bloch Spheres
    for idx, s in enumerate(sim_results["stage_results"]):
        with b_cols[idx + 1]:
            out_r = s["output_state"].bloch_vector()
            n_axis = s["n_hat"]
            fig_b_stage = plot_bloch_sphere(out_r, orientation_axis=n_axis, title=f"After SG-{s['stage_id']}")
            st.pyplot(fig_b_stage, use_container_width=True)
            plt.close(fig_b_stage)
            st.caption(f"**SG-{s['stage_id']}:** $\\vec{{r}} = [{out_r[0]:.2f}, {out_r[1]:.2f}, {out_r[2]:.2f}]$, $\\hat{{n}} = [{n_axis[0]:.2f}, {n_axis[1]:.2f}, {n_axis[2]:.2f}]$")

    st.markdown("""
    > [!NOTE]
    > **Physical Interpretation of the 3D Bloch Sphere:**
    > - The **green arrow** represents the quantum state vector $|\\psi\\rangle$.
    > - The **dashed purple arrow** represents the magnetic field axis $\\hat{n}$.
    > - **Von Neumann Collapse:** Upon measurement, the state vector is projected directly onto either $+\\hat{n}$ or $-\\hat{n}$.
    > - For an unpolarized beam from the oven, $\\vec{r} = (0, 0, 0)$, situated at the exact center of the sphere, reflecting maximum entropy (a statistical mixture with no preferred orientation).
    """)


# ==============================================================================
# TAB 4: MONTE CARLO PARTICLE STATISTICS
# ==============================================================================
with tab_mc:
    st.markdown("### Stochastic Monte Carlo Simulation (Law of Large Numbers)")
    st.markdown(f"""
    In laboratory experiments, atoms pass through one-by-one. 
    Here we simulate **{num_atoms:,} individual silver atoms** traversing the apparatus cascade. 
    At each magnet, a random quantum measurement collapse occurs according to Born's probability rule.
    """)

    mc_col1, mc_col2 = st.columns([1.1, 0.9])

    with mc_col1:
        fig_mc = plot_monte_carlo_histogram(mc_results)
        st.pyplot(fig_mc, use_container_width=True)
        plt.close(fig_mc)

    with mc_col2:
        st.markdown("#### Statistical Convergence Analysis")
        st.write(f"- **Simulated Transmission Ratio:** `{mc_results['simulated_ratio'] * 100:.2f}%`")
        st.write(f"- **Exact Quantum Theoretical Probability:** `{mc_results['theoretical_prob'] * 100:.2f}%`")
        st.write(f"- **Absolute Error ($|P_{{sim}} - P_{{theo}}|$):** `{mc_results['absolute_error'] * 100:.3f}%`")
        st.write(f"- **Expected Standard Error ($\\sigma = \\sqrt{{P(1-P)/N}}$):** `{mc_results['expected_standard_error'] * 100:.3f}%`")
        
        is_consistent = mc_results['absolute_error'] <= 2.5 * mc_results['expected_standard_error']
        if is_consistent:
            st.success("✅ **Experimentally Verified:** The empirical count is well within the $2.5\\sigma$ statistical confidence interval!")
        else:
            st.warning("⚠️ Fluctuation detected (typical statistical anomaly for finite N). Try resampling or increasing N.")

        st.markdown("#### Stage-by-Stage Atom Breakdown")
        for s_idx, (plus_c, minus_c, surv) in enumerate(zip(mc_results['plus_counts'], mc_results['minus_counts'], mc_results['stage_survivors'])):
            st.write(f"**Stage {s_idx+1}:** `|+n⟩` = {plus_c:,} atoms, `|-n⟩` = {minus_c:,} atoms $\\implies$ **Surviving to next stage:** {surv:,} atoms.")


