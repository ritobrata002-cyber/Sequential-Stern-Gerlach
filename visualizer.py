"""
visualizer.py
=============
Interactive and Publication-Quality Visualizations for Sequential Stern-Gerlach Simulation.

Contains:
1. 2D Sequential Experiment Schematic / Beam-Splitting Flowchart
2. 3D Bloch Sphere Visualizer with Vector Projections
3. Stage Probability Comparison Charts
4. Monte Carlo Statistical Analysis & Convergence Plots

Designed for clear display in Streamlit dashboards.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Arrow, Rectangle
from mpl_toolkits.mplot3d import Axes3D

# Configure aesthetic typography and palette
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 0.8


def plot_experiment_schematic(simulation_results: Dict[str, Any]) -> plt.Figure:
    """
    Renders an accurate, physical 2D schematic of the sequential Stern-Gerlach apparatus.
    
    Physics Design:
    - In free space (drift regions between apparatuses and before detector), atoms travel
      as straight, horizontal lines representing ballistic/inertial motion without forces.
    - Inside each magnetic chamber, the inhomogeneous gradient splits the beam.
    - Each subsequent apparatus is aligned with the incoming beam height, so every beam
      falls squarely and dead-center onto the next apparatus entrance.
    - Blocked beams terminate at an absorber stop plate.
    - The final beam travels straight and horizontally into the detector.
    """
    stage_results = simulation_results["stage_results"]
    num_stages = len(stage_results)

    # Geometry parameters
    mag_w = 1.6
    mag_h = 2.0
    filter_w = 2.0
    stage_stride = mag_w + filter_w
    x_stage_start = 2.0
    y_split = 0.45  # Split deflection within each magnet

    x_det = x_stage_start + (num_stages - 1) * stage_stride + mag_w + filter_w
    x_max = x_det + 2.5

    # 1. Pre-calculate stage heights so each stage is centered on its incoming beam
    stage_y_centers = []
    current_y = 0.0
    for s_info in stage_results:
        stage_y_centers.append(current_y)
        flt = s_info['transmitted_beam']
        if flt == '+':
            current_y += y_split
        elif flt == '-':
            current_y -= y_split
        # if 'both', keep current_y

    final_beam_y = current_y

    # Calculate optimal vertical bounds for centering
    all_y = [0.0] + stage_y_centers + [final_beam_y]
    y_center = (min(all_y) + max(all_y)) / 2.0
    y_span = max(max(all_y) - min(all_y) + 2.6, 3.8)
    y_min = y_center - y_span / 2.0
    y_max = y_center + y_span / 2.0

    fig, ax = plt.subplots(figsize=(11.8, 4.6), dpi=120)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    ax.set_xlim(-0.6, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axis('off')

    # Colors
    color_up = '#10b981'    # Emerald for spin up |+n>
    color_down = '#f43f5e'  # Rose/Red for spin down |-n>
    color_neut = '#38bdf8'  # Cyan for initial / incident beam

    # 2. Oven / Source (Left)
    oven_w, oven_h = 1.3, 1.4
    oven_x, oven_y = -0.3, -oven_h / 2.0
    oven_box = FancyBboxPatch((oven_x, oven_y), oven_w, oven_h,
                              boxstyle="round,pad=0.08,rounding_size=0.15",
                              facecolor='#1e293b', edgecolor='#38bdf8', linewidth=2.0)
    ax.add_patch(oven_box)
    ax.text(oven_x + oven_w / 2, 0.2, "ATOMIC\nOVEN", color='#f8fafc', fontsize=10,
            weight='bold', ha='center', va='center')
    ax.text(oven_x + oven_w / 2, -0.38, "Ag (Spin-1/2)", color='#94a3b8', fontsize=8,
            ha='center', va='center')

    # Initial beam exiting oven (straight horizontal line to Stage 1 entrance)
    oven_exit_x = oven_x + oven_w
    ax.plot([oven_exit_x, x_stage_start], [0.0, 0.0], color=color_neut, linewidth=3.0, solid_capstyle='round')
    ax.annotate('', xy=(x_stage_start, 0.0), xytext=(x_stage_start - 0.4, 0.0),
                arrowprops=dict(arrowstyle="-|>", color=color_neut, lw=2.5, mutation_scale=14))
    ax.text((oven_exit_x + x_stage_start) / 2, 0.26, "100%", color=color_neut, fontsize=9, weight='bold', ha='center')

    # 3. Draw Stages
    for idx, s_info in enumerate(stage_results):
        x_in = x_stage_start + idx * stage_stride
        x_out = x_in + mag_w
        x_next = x_in + stage_stride
        y_in = stage_y_centers[idx]

        stage_name = f"SG-{idx+1}"
        theta_val = s_info['theta_deg']
        phi_val = s_info['phi_deg']
        p_plus = s_info['p_plus_conditional']
        p_minus = s_info['p_minus_conditional']
        filter_mode = s_info['transmitted_beam']
        is_last = (idx == num_stages - 1)

        # Draw Magnet Body centered on the incoming beam height y_in
        mag_box = FancyBboxPatch((x_in, y_in - mag_h / 2.0), mag_w, mag_h,
                                 boxstyle="round,pad=0.06,rounding_size=0.12",
                                 facecolor='#1e293b', edgecolor='#a855f7', linewidth=2.0)
        ax.add_patch(mag_box)

        # Entrance aperture slit marker (centered at incoming beam)
        ax.plot([x_in, x_in], [y_in - 0.22, y_in + 0.22], color='#a855f7', linewidth=3.5)
        ax.scatter([x_in], [y_in], color=color_neut, s=25, zorder=5)

        # Stage Header Badge (Positioned ABOVE magnet)
        ax.text(x_in + mag_w / 2, y_in + mag_h / 2 + 0.36, stage_name, color='#e2e8f0',
                fontsize=10.5, weight='bold', ha='center')
        angle_str = f"θ={theta_val:.0f}°" + (f", ϕ={phi_val:.0f}°" if phi_val != 0 else "")
        ax.text(x_in + mag_w / 2, y_in + mag_h / 2 + 0.14, angle_str, color='#c084fc',
                fontsize=9, ha='center', weight='bold')

        # Inhomogeneous Magnetic Pole Pieces
        ax.text(x_in + mag_w / 2, y_in + mag_h / 2 - 0.22, "N (Knife-edge)", color='#f43f5e',
                fontsize=8.5, weight='bold', ha='center', va='center')
        ax.text(x_in + mag_w / 2, y_in - mag_h / 2 + 0.22, "S (Grooved)", color='#38bdf8',
                fontsize=8.5, weight='bold', ha='center', va='center')

        # Beams traversing INSIDE the magnet (splitting under magnetic gradient)
        x_pts = np.linspace(x_in, x_out, 30)
        t_mag = (x_pts - x_in) / mag_w
        y_up_mag = y_in + y_split * (t_mag ** 1.3)
        y_down_mag = y_in - y_split * (t_mag ** 1.3)

        # Draw internal beam splitting
        ax.plot(x_pts, y_up_mag, color=color_up, linewidth=2.6, zorder=4)
        ax.plot(x_pts, y_down_mag, color=color_down, linewidth=2.6, zorder=4)

        # Exit markers at the magnet exit face
        ax.plot([x_out, x_out], [y_in - 0.22, y_in + 0.22], color='#a855f7', linewidth=2.5)

        # Free space drift region beyond magnet exit (BALLISTIC STRAIGHT HORIZONTAL LINES)
        x_stop = x_out + 0.75
        x_target = x_det if is_last else x_next

        y_exit_up = y_in + y_split
        y_exit_down = y_in - y_split

        if filter_mode == '+':
            # Lower beam (| -n >) travels straight to blocker
            ax.plot([x_out, x_stop], [y_exit_down, y_exit_down], color=color_down, linewidth=2.2, linestyle=':')
            stop_box = Rectangle((x_stop - 0.05, y_exit_down - 0.22), 0.10, 0.44,
                                 facecolor='#ef4444', edgecolor='#fca5a5', linewidth=1.5, zorder=6)
            ax.add_patch(stop_box)
            ax.text(x_stop, y_exit_down - 0.40, "BLOCKED", color='#ef4444',
                    fontsize=7.5, weight='bold', ha='center')
            ax.text((x_out + x_stop)/2, y_exit_down + 0.16, f"|-n⟩\n{p_minus*100:.1f}%",
                    color=color_down, fontsize=8, weight='bold', ha='center')

            # Upper beam (| +n >) travels STRAIGHT AND HORIZONTAL all the way to target
            ax.plot([x_out, x_target], [y_exit_up, y_exit_up], color=color_up, linewidth=2.8, zorder=4)
            # Arrowhead entering next stage / detector
            ax.annotate('', xy=(x_target, y_exit_up), xytext=(x_target - 0.35, y_exit_up),
                        arrowprops=dict(arrowstyle="-|>", color=color_up, lw=2.2, mutation_scale=12))
            ax.text((x_out + x_target)/2, y_exit_up + 0.22, f"|+n⟩  {p_plus*100:.1f}%",
                    color=color_up, fontsize=8.5, weight='bold', ha='center')

        elif filter_mode == '-':
            # Upper beam (| +n >) travels straight to blocker
            ax.plot([x_out, x_stop], [y_exit_up, y_exit_up], color=color_up, linewidth=2.2, linestyle=':')
            stop_box = Rectangle((x_stop - 0.05, y_exit_up - 0.22), 0.10, 0.44,
                                 facecolor='#ef4444', edgecolor='#fca5a5', linewidth=1.5, zorder=6)
            ax.add_patch(stop_box)
            ax.text(x_stop, y_exit_up + 0.32, "BLOCKED", color='#ef4444',
                    fontsize=7.5, weight='bold', ha='center')
            ax.text((x_out + x_stop)/2, y_exit_up - 0.34, f"|+n⟩\n{p_plus*100:.1f}%",
                    color=color_up, fontsize=8, weight='bold', ha='center')

            # Lower beam (| -n >) travels STRAIGHT AND HORIZONTAL all the way to target
            ax.plot([x_out, x_target], [y_exit_down, y_exit_down], color=color_down, linewidth=2.8, zorder=4)
            # Arrowhead entering next stage / detector
            ax.annotate('', xy=(x_target, y_exit_down), xytext=(x_target - 0.35, y_exit_down),
                        arrowprops=dict(arrowstyle="-|>", color=color_down, lw=2.2, mutation_scale=12))
            ax.text((x_out + x_target)/2, y_exit_down - 0.34, f"|-n⟩  {p_minus*100:.1f}%",
                    color=color_down, fontsize=8.5, weight='bold', ha='center')

        else:  # 'both'
            ax.plot([x_out, x_target], [y_exit_up, y_exit_up], color=color_up, linewidth=2.4)
            ax.plot([x_out, x_target], [y_exit_down, y_exit_down], color=color_down, linewidth=2.4)
            ax.text((x_out + x_target)/2, y_exit_up + 0.20, f"|+n⟩ {p_plus*100:.1f}%",
                    color=color_up, fontsize=8, weight='bold', ha='center')
            ax.text((x_out + x_target)/2, y_exit_down - 0.34, f"|-n⟩ {p_minus*100:.1f}%",
                    color=color_down, fontsize=8, weight='bold', ha='center')

    # 4. Final Detector Screen (Right)
    det_w = 0.45
    det_h = max(2.6, abs(final_beam_y) * 2.0 + 1.6)
    det_box = FancyBboxPatch((x_det, final_beam_y - det_h / 2.0), det_w, det_h,
                             boxstyle="round,pad=0.05,rounding_size=0.1",
                             facecolor='#0284c7', edgecolor='#38bdf8', linewidth=2.2, zorder=7)
    ax.add_patch(det_box)
    ax.text(x_det + det_w / 2, final_beam_y, "DETECTOR", color='#ffffff',
            fontsize=9.5, weight='bold', rotation=90, ha='center', va='center', zorder=8)

    # Glowing detection spot where the straight beam hits the detector
    ax.scatter([x_det], [final_beam_y], color='#22d3ee', s=100, edgecolors='#ffffff', linewidths=1.6, zorder=9)

    # Final cumulative intensity badge
    final_prob = simulation_results["final_cumulative_prob"]
    badge_x = x_det + det_w + 0.25
    badge = FancyBboxPatch((badge_x, final_beam_y - 0.65), 1.55, 1.3,
                           boxstyle="round,pad=0.08,rounding_size=0.15",
                           facecolor='#0f2b46', edgecolor='#22d3ee', linewidth=2.0)
    ax.add_patch(badge)
    ax.text(badge_x + 0.77, final_beam_y + 0.22, "Final Beam", color='#94a3b8', fontsize=8.5, ha='center')
    ax.text(badge_x + 0.77, final_beam_y - 0.22, f"{final_prob*100:.2f}%", color='#22d3ee',
            fontsize=13, weight='bold', ha='center')

    plt.tight_layout()
    return fig


def plot_bloch_sphere(bloch_vec: np.ndarray, orientation_axis: Optional[np.ndarray] = None,
                      title: str = "Bloch Sphere Representation") -> plt.Figure:
    """
    Generates an educational, publication-ready 3D Bloch Sphere representation.
    
    Shows:
    - Semi-transparent unit sphere surface |r| = 1.
    - Coordinate axes (x, y, z) with basis labels (| +z >, | -z >, | +x >, | +y >).
    - Current State Vector (bold red/emerald vector).
    - Measurement Axis n_hat (dashed violet vector) showing the projection orientation.
    """
    fig = plt.figure(figsize=(4.5, 4.5), dpi=120)
    fig.patch.set_facecolor('#0f172a')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0f172a')

    # 1. Wireframe / Semi-transparent Sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(xs, ys, zs, color='#334155', alpha=0.12, edgecolor='#475569',
                    linewidth=0.3, rstride=2, cstride=2)

    # 2. Equator and Meridian Guide Circles
    theta_circ = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta_circ), np.sin(theta_circ), np.zeros_like(theta_circ),
            color='#64748b', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.plot(np.cos(theta_circ), np.zeros_like(theta_circ), np.sin(theta_circ),
            color='#64748b', linestyle=':', linewidth=0.6, alpha=0.5)

    # 3. Coordinate Axes lines
    axis_lim = 1.3
    ax.plot([-axis_lim, axis_lim], [0, 0], [0, 0], color='#475569', linewidth=1)
    ax.plot([0, 0], [-axis_lim, axis_lim], [0, 0], color='#475569', linewidth=1)
    ax.plot([0, 0], [0, 0], [-axis_lim, axis_lim], color='#475569', linewidth=1)

    # Basis State Annotations
    ax.text(0, 0, 1.35, r"$|+z\rangle$", color='#38bdf8', fontsize=10, weight='bold', ha='center')
    ax.text(0, 0, -1.45, r"$|-z\rangle$", color='#f43f5e', fontsize=10, weight='bold', ha='center')
    ax.text(1.35, 0, 0, r"$|+x\rangle$", color='#a3e635', fontsize=9, ha='center')
    ax.text(0, 1.35, 0, r"$|+y\rangle$", color='#facc15', fontsize=9, ha='center')

    # 4. Measurement Orientation Axis (if provided)
    if orientation_axis is not None:
        nx, ny, nz = orientation_axis
        ax.quiver(0, 0, 0, nx * 1.15, ny * 1.15, nz * 1.15,
                  color='#c084fc', arrow_length_ratio=0.12, linewidth=2.2, linestyle='--')
        ax.text(nx * 1.25, ny * 1.25, nz * 1.25, r"$\hat{n}$", color='#c084fc',
                fontsize=11, weight='bold', ha='center')

    # 5. Quantum State Vector
    rx, ry, rz = bloch_vec
    norm_r = np.linalg.norm(bloch_vec)

    if norm_r > 1e-4:
        ax.quiver(0, 0, 0, rx, ry, rz,
                  color='#10b981', arrow_length_ratio=0.15, linewidth=3.2)
        ax.scatter([rx], [ry], [rz], color='#34d399', s=45, edgecolors='#ffffff', linewidths=1.2)
        ax.text(rx * 1.15, ry * 1.15, rz * 1.15, r"$|\psi\rangle$", color='#34d399',
                fontsize=11, weight='bold', ha='center')
    else:
        # Origin point for unpolarized state
        ax.scatter([0], [0], [0], color='#94a3b8', s=60, edgecolors='#ffffff', linewidths=1.5)
        ax.text(0.1, 0.1, 0.1, "Unpolarized (r=0)", color='#94a3b8', fontsize=9, weight='bold')

    # Visual camera tuning
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_axis_off()
    ax.view_init(elev=24, azim=45)
    plt.title(title, color='#f8fafc', fontsize=11, pad=10, weight='bold')
    plt.tight_layout()
    return fig


def plot_probability_bars(stage_results: List[Dict[str, Any]]) -> plt.Figure:
    """
    Renders comparative bar charts showing conditional branch probabilities P(+) and P(-)
    at each sequential stage.
    """
    fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=120)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    stage_names = [f"Stage {s['stage_id']}\n(θ={s['theta_deg']:.0f}°)" for s in stage_results]
    p_plus_vals = [s['p_plus_conditional'] * 100 for s in stage_results]
    p_minus_vals = [s['p_minus_conditional'] * 100 for s in stage_results]

    x = np.arange(len(stage_names))
    width = 0.35

    rects1 = ax.bar(x - width/2, p_plus_vals, width, label=r'$P(|+n\rangle)$',
                    color='#10b981', edgecolor='#059669', linewidth=1.2)
    rects2 = ax.bar(x + width/2, p_minus_vals, width, label=r'$P(|-n\rangle)$',
                    color='#f43f5e', edgecolor='#e11d48', linewidth=1.2)

    # Bar value labels
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', color='#34d399', fontsize=8, weight='bold')

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', color='#fb7185', fontsize=8, weight='bold')

    ax.set_ylabel('Probability (%)', color='#e2e8f0', fontsize=9)
    ax.set_title('Conditional Branch Probabilities per Stage', color='#f8fafc', fontsize=11, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stage_names, color='#cbd5e1', fontsize=9)
    ax.tick_params(colors='#94a3b8')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle='--', alpha=0.2, color='#64748b')

    legend = ax.legend(facecolor='#0f172a', edgecolor='#334155', fontsize=9)
    for text in legend.get_texts():
        text.set_color('#e2e8f0')

    plt.tight_layout()
    return fig


def plot_monte_carlo_histogram(mc_results: Dict[str, Any]) -> plt.Figure:
    """
    Renders a statistical bar chart showing simulated particle counts vs theoretical Born rule.
    """
    fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=120)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    total_atoms = mc_results["num_atoms"]
    detected_atoms = mc_results["final_detected_atoms"]
    sim_ratio = mc_results["simulated_ratio"] * 100
    theo_ratio = mc_results["theoretical_prob"] * 100

    categories = ['Final Detected', 'Blocked / Filtered']
    sim_values = [detected_atoms, total_atoms - detected_atoms]
    colors = ['#06b6d4', '#64748b']

    bars = ax.bar(categories, sim_values, color=colors, edgecolor='#ffffff', linewidth=0.8, width=0.45)

    for bar, val in zip(bars, sim_values):
        pct = (val / total_atoms) * 100 if total_atoms > 0 else 0
        ax.annotate(f'{val:,}\n({pct:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', color='#f8fafc', fontsize=9, weight='bold')

    # Draw theoretical line for detected
    theo_count = total_atoms * (theo_ratio / 100.0)
    ax.axhline(theo_count, color='#facc15', linestyle='--', linewidth=1.8,
               label=f'Theoretical Born Rule: {theo_ratio:.2f}% ({int(theo_count):,} atoms)')

    ax.set_ylabel('Silver Atom Counts (N)', color='#e2e8f0', fontsize=9)
    ax.set_title(f'Monte Carlo Simulation (N = {total_atoms:,} Atoms)', color='#f8fafc', fontsize=11, weight='bold')
    ax.tick_params(colors='#94a3b8')
    ax.set_ylim(0, max(sim_values) * 1.28 if max(sim_values) > 0 else 100)
    ax.grid(axis='y', linestyle='--', alpha=0.2, color='#64748b')

    legend = ax.legend(facecolor='#0f172a', edgecolor='#334155', fontsize=8.5, loc='upper right')
    for text in legend.get_texts():
        text.set_color('#facc15')

    plt.tight_layout()
    return fig
