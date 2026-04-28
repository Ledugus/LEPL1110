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
    """
    m(x, t) = m(x + c·t) in the sense of a band moving northward.
    Returns +1 (favourable) or -1 (unfavourable).
    """
    xi = x[0] + c * t  # habitat coordinate (translated)
    return 1.0 if abs(xi) < L_hab else -1.0


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


def plot_favourable_band(ax, t, c, L_hab, L, H):
    """
    Overlay the moving favourable habitat band on the current axes.
    """
    x_left = -L_hab - c * t
    x_right = L_hab - c * t

    # Clip the band to the computational domain [0, L].
    x0 = max(0.0, x_left)
    x1 = min(L, x_right)
    if x1 <= x0:
        return

    width = x1 - x0
    band = Rectangle(
        (x0, 0.0),
        width,
        H,
        facecolor="limegreen",
        alpha=0.14,
        edgecolor="none",
        zorder=6,
    )
    ax.add_patch(band)

    ax.plot(
        [x0, x0],
        [0.0, H],
        linestyle="--",
        linewidth=1.0,
        color="forestgreen",
        alpha=0.9,
        zorder=7,
    )
    ax.plot(
        [x1, x1],
        [0.0, H],
        linestyle="--",
        linewidth=1.0,
        color="forestgreen",
        alpha=0.9,
        zorder=7,
    )


def main(L, H, h, order, dt, nstep, theta):
    # ------------------------------------------------------------------
    # Physical parameters (SI units: m, s, individuals/m²)
    # ------------------------------------------------------------------
    D = 1.0  # diffusion coefficient  [m²/s]
    r = 1.0  # growth rate            [1/s]
    r_tilde = 1.0  # decay rate outside habitat  [1/s]
    K = 100.0  # carrying capacity      [ind/m²]
    c = 0.3  # climate shift speed    [m/s]
    L_hab = L / 3.0  # half-width of favourable band  [m]

    # Initial Gaussian
    u0_max = 10  # peak density
    x0 = [L / 2, L / 2]  # centre of initial population
    sigma = L / 10.0  # width of initial Gaussian

    def size_field(x, y):
        return h

    gmsh_init("kpp_fisher_2d")

    # ------------------------------------------------------------------
    # Build mesh
    # ------------------------------------------------------------------
    (elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bndsTags) = (
        build_2d_rectangle_mesh(L=L, H=H, size_field=size_field, order=order)
    )

    # plot_mesh_2d(elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bndsTags)

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
    parser = argparse.ArgumentParser(description="KPP-Fisher 2D species diffusion")
    parser.add_argument("-order", type=int, default=1)
    parser.add_argument("-L", type=float, default=5.0)
    parser.add_argument("-H", type=float, default=5.0)
    parser.add_argument("-hc", type=float, default=0.2)
    parser.add_argument("--dt", type=float, default=0.1, help="time step")
    parser.add_argument("--nsteps", type=int, default=300, help="number of steps")
    parser.add_argument(
        "--theta", type=float, default=1.0, help="θ-scheme (1=implicit Euler)"
    )
    args = parser.parse_args()
    main(args.L, args.H, args.hc, args.order, args.dt, args.nsteps, args.theta)
