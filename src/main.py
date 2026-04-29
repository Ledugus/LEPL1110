# main_kpp_fisher.py
"""
KPP-Fisher species diffusion in a shifting climate (LEPL1110 - Groupe 75)

Solves:
    ∂u/∂t − D·Δu = f(u, x, t)

with:
    f(u, x, t) = r·u·(1 − u/K)   if m(x,t) > 0  (favourable habitat)
               = −r̃·u             otherwise

    m(x, t) = m(x + c·t)          travelling habitat wave

Boundary conditions: homogeneous Neumann ∂u/∂n = 0 (no-flux)
Initial condition:   u(x,0) = u0·exp(−‖x − x0‖²/(2σ²))
"""

import argparse
import numpy as np
from mesh import *

from gmsh_utils import (
    gmsh_init,
    gmsh_finalize,
    prepare_quadrature_and_basis,
    get_jacobians,
    build_2d_rectangle_mesh,
)
from stiffness import assemble_stiffness_and_rhs
from mass import assemble_mass
from dirichlet import theta_step
from plot_utils import (
    setup_interactive_figure,
    plot_mesh_2d,
    plot_fe_solution_2d,
    VideoDisplay,
    InteractiveDisplay,
)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ---------------------------------------------------------------------------
# Habitat viability m(x, t): travelling wave at speed c along x-axis.
# Returns True in the favourable zone.
# The favourable band is centred at x = c*t, with half-width L_hab.
# ---------------------------------------------------------------------------
def habitat(x, t, c, L_hab):
    yi = x[1] - c * t          # <-- axe y au lieu de x[0]
    return 1.0 if abs(yi - 0) < L_hab else -1.0


# ---------------------------------------------------------------------------
# Nonlinear source term  f(u, x, t)
# ---------------------------------------------------------------------------
def f_source(u, x, t, c, L_hab, r, r_tilde, K):
    m = habitat(x, t, c, L_hab)
    if m > 0:
        return r * u * (1.0 - u / K)
    else:
        return -r_tilde * u


# ---------------------------------------------------------------------------
# Gaussian initial condition
# ---------------------------------------------------------------------------
def u_init(x, x0, u0_max, sigma):
    """u(x, 0) = u0_max · exp(−‖x − x0‖² / (2σ²))"""
    dist2 = np.sum((x[:2] - np.array(x0)) ** 2)  # use only x,y coords
    return u0_max * np.exp(-dist2 / (2.0 * sigma**2))


# ---------------------------------------------------------------------------
# Linearised (explicit-in-u) source for the theta-scheme:
# The stiffness assembler expects f(x, t) → scalar, so we freeze u from the
# previous time step and pass f(u_n, x, t) as an explicit contribution.
# ---------------------------------------------------------------------------
def make_explicit_source(U, dof_coords, tag_to_dof, t, c, L_hab, r, r_tilde, K):
    """
    Build the F vector by evaluating f(u_n, x_i, t) at every DOF.
    Returns a callable x → scalar for use with assemble_stiffness_and_rhs.
    """

    def _f(x):
        # Find closest DOF (cheap for moderate mesh sizes)
        idx = np.argmin(np.sum((dof_coords[:, :2] - x[:2]) ** 2, axis=1))
        u_n = U[idx]
        u_n = max(u_n, 0.0)  # enforce positivity
        return f_source(u_n, x, t, c, L_hab, r, r_tilde, K)

    return _f


def plot_favourable_band(ax, t, c, L_hab, x_min, x_max):
    y_center = c * t
    y0 = y_center - L_hab
    y1 = y_center + L_hab
    ax.axhspan(y0, y1, alpha=0.14, color='limegreen', zorder=6)
    ax.axhline(y0, linestyle='--', linewidth=1.0, color='forestgreen', zorder=7)
    ax.axhline(y1, linestyle='--', linewidth=1.0, color='forestgreen', zorder=7)


def main(h, order, dt, nstep, theta, country):   # <-- L et H disparaissent, déduits du mesh
    D = 0.5
    r = 1.0
    r_tilde = 0.5
    K = 10
    c = 5.0        # km/an, déplacement vers le nord (axe y ici)
    
    gmsh_init("belgium_kpp")
    
    # --- Mesh Belgique ---
    (elemType, nodeTags, nodeCoords, elemTags, elemNodeTags,
     bnds, bndsTags, bounds) = build_country_mesh(
        country, mesh_size=h, order=order
    )
    
    x_min, x_max, y_min, y_max = bounds
    L = x_max - x_min
    H = y_max - y_min
    L_hab = H / 6.0   # bande favorable : 1/3 de la hauteur du pays
    
    # Centre initial de la population 
    x0 = [0.0, -H]
    sigma = L / 8.0
    u0_max = 10.0

    plot_mesh_2d(elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bndsTags)
    # ------------------------------------------------------------------
    # DOF bookkeeping
    # ------------------------------------------------------------------
    unique_dofs_tags = np.unique(elemNodeTags)
    num_dofs = len(unique_dofs_tags)
    max_tag = int(np.max(nodeTags))

    dof_coords = np.zeros((num_dofs, 3))
    all_coords = nodeCoords.reshape(-1, 3)
    tag_to_dof = np.full(max_tag + 1, -1, dtype=int)

    for i, tag in enumerate(unique_dofs_tags):
        tag_to_dof[int(tag)] = i
        dof_coords[i] = all_coords[i]

    # ------------------------------------------------------------------
    # Quadrature, basis functions, Jacobians
    # ------------------------------------------------------------------
    xi, w, N, gN = prepare_quadrature_and_basis(elemType, order)
    jac, det, coords = get_jacobians(elemType, xi)

    # ------------------------------------------------------------------
    # Diffusion coefficient κ = D (isotropic)
    # ------------------------------------------------------------------
    def kappa(x):
        return D

    # ------------------------------------------------------------------
    # Initial condition (Gaussian)
    # ------------------------------------------------------------------
    U = np.array([u_init(x, x0, u0_max, sigma) for x in dof_coords], dtype=float)

    # ------------------------------------------------------------------
    # Mass matrix  M  (time-independent)
    # ------------------------------------------------------------------
    M_lil = assemble_mass(elemTags, elemNodeTags, det, w, N, tag_to_dof)
    M = M_lil.tocsr()

    # ------------------------------------------------------------------
    # Time loop
    # ------------------------------------------------------------------
    # display = InteractiveDisplay()
    display = VideoDisplay()

    fig, ax = display.get_figure()

    for step in range(nstep):
        if (step % 10) == 0:
            print(f"Calculating frame {step+1}/{nstep} ({100 * step/nstep:.2f}%)")
        t = step * dt

        # --- Source term at t_n (explicit, frozen at U_n) ---
        f_n = make_explicit_source(
            U, dof_coords, tag_to_dof, t, c, L_hab, r, r_tilde, K
        )

        # --- Source term at t_{n+1} (explicit, still frozen at U_n
        #     for a semi-implicit treatment; replace with U_{n+1} for
        #     a fully implicit nonlinear Newton loop if needed) ---
        f_np1 = make_explicit_source(
            U, dof_coords, tag_to_dof, t + dt, c, L_hab, r, r_tilde, K
        )

        # --- Assemble stiffness + RHS at t_n and t_{n+1} ---
        K_lil_n, F_n = assemble_stiffness_and_rhs(
            elemTags, elemNodeTags, jac, det, coords, w, N, gN, kappa, f_n, tag_to_dof
        )

        K_lil_np1, F_np1 = assemble_stiffness_and_rhs(
            elemTags, elemNodeTags, jac, det, coords, w, N, gN, kappa, f_np1, tag_to_dof
        )

        K_n = K_lil_n.tocsr()
        K_np1 = K_lil_np1.tocsr()

        # --- θ-scheme step with NO Dirichlet DOFs (pure Neumann) ---
        #     Pass empty arrays so theta_step applies no Dirichlet correction
        U = theta_step(
            M,
            K_np1,
            F_n,
            F_np1,
            U,
            dt=dt,
            theta=theta,
            dirichlet_dofs=np.array([], dtype=int),
            dir_vals_np1=np.array([], dtype=float),
        )

        # Enforce positivity (population cannot be negative)
        U = np.maximum(U, 0.0)

        # --- Plot ---
        ax.clear()
        plot_fe_solution_2d(
            elemNodeTags=elemNodeTags,
            nodeTags=nodeTags,
            nodeCoords=nodeCoords,
            U=U,
            tag_to_dof=tag_to_dof,
            show_mesh=False,
            ax=ax,
        )
        plot_favourable_band(ax, t + dt, c, L_hab, L, H)
        ax.set_title(
            f"t = {t + dt:.3f} s  |  "
            f"c = {c:.2f} m/s  |  "
            f"θ = {theta}  |  "
            f"max(u) = {U.max():.3f}"
        )
        ax.set_xlabel("x  [m]")
        ax.set_ylabel("y  [m]")
        ax.axis("equal")

        # --- Display ---
        success = display.add_frame(fig, ax)
        if not success:
            print("Interrupted by display (error or closing)")
            break

    display.end()
    gmsh_finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-order", type=int, default=1)
    parser.add_argument("-hc", type=float, default=100)   # en km maintenant !
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--nsteps", type=int, default=100)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--country", type=str, default="Australia")
    args = parser.parse_args()
    main(args.hc, args.order, args.dt, args.nsteps, args.theta, args.country)
