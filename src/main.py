# main_kpp_fisher.py
"""
KPP-Fisher species diffusion in a shifting climate (LEPL1110 - Groupe 75)

Solves:
    ∂u/∂t − D·Δu = f(u, x, t)

with:
    f(u, x, t) = r(x)·u·(1 − u/K)   if m(x,t) > 0  (favourable habitat)
               = −r̃·u                otherwise

    r(x)   = r · exp(−(elev(x) − elev_opt)² / (2·elev_width²))
    D(x)   = D / (1 + alpha_slope · slope(x)/slope_max)

    m(x, t) = m(x + c·t)          travelling habitat wave (along y-axis)

Boundary conditions: homogeneous Neumann ∂u/∂n = 0 (no-flux)
Initial condition:   u(x,0) = u0·exp(−‖x − x0‖²/(2σ²))
"""

import argparse
import numpy as np
from scipy.spatial import KDTree
from mesh import *
from altitude import *

from gmsh_utils import (
    gmsh_init,
    gmsh_finalize,
    prepare_quadrature_and_basis,
    get_jacobians,
)
from stiffness import assemble_stiffness_and_rhs
from mass import assemble_mass
from dirichlet import theta_step
from plot_utils import (
    plot_mesh_2d,
    plot_fe_solution_2d,
    VideoDisplay,
    InteractiveDisplay,
)
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Habitat viability m(x, t): travelling wave along y-axis at speed c.
# ---------------------------------------------------------------------------
def habitat(x, t, c, L_hab, band_y0=0.0):
    yi = x[1] - (band_y0 + c * t)
    exp = 6
    return (1-(yi/L_hab)**exp)/(1+(yi/L_hab)**exp)


# ---------------------------------------------------------------------------
# Nonlinear source term  f(u, x, t)
# ---------------------------------------------------------------------------
def f_source_binaire(u, x, t, c, L_hab, r_tilde, K, r_fn, band_y0=0.0):
    m = habitat(x, t, c, L_hab, band_y0=band_y0)
    if m > 0:
        r_x = r_fn(x)
        return r_x * u * (1.0 - u / K)
    else:
        return -r_tilde * u
    
def f_source_non_binaire(u, x, t, c, L_hab, r_tilde, K, r_fn, band_y0=0.0):
    m = habitat(x, t, c, L_hab, band_y0=band_y0)
    r_x = r_fn(x)
    return ((m+1) * r_x * u * (1.0 - u / K) + (1-m) * (-r_tilde * u))/2

f_source = f_source_non_binaire
    
# ---------------------------------------------------------------------------
# Gaussian initial condition
# ---------------------------------------------------------------------------
def u_init(x, x0, u0_max, sigma):
    """u(x, 0) = u0_max · exp(−‖x − x0‖² / (2σ²))"""
    dist2 = np.sum((x[:2] - np.array(x0)) ** 2)
    return u0_max * np.exp(-dist2 / (2.0 * sigma**2))


# ---------------------------------------------------------------------------
# Semi-implicit explicit source term (u frozen at U_n)
# ---------------------------------------------------------------------------
def make_explicit_source(U, dof_tree, dof_coords, t, c, L_hab, r_tilde, K, r_fn, band_y0=0.0):
    def _f(x):
        _, idx = dof_tree.query(x[:2])
        u_n = max(U[idx], 0.0)
        return f_source(u_n, x, t, c, L_hab, r_tilde, K, r_fn, band_y0=band_y0)
    return _f


# ---------------------------------------------------------------------------
# Favourable band overlay on plot
# ---------------------------------------------------------------------------
def plot_favourable_band(ax, t, c, L_hab, band_y0=0.0):
    y_center = band_y0 + c * t
    y0 = y_center - L_hab
    y1 = y_center + L_hab
    ax.axhspan(y0, y1, alpha=0.14, color='limegreen', zorder=6)
    ax.axhline(y0, linestyle='--', linewidth=1.0, color='forestgreen', zorder=7)
    ax.axhline(y1, linestyle='--', linewidth=1.0, color='forestgreen', zorder=7)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(h, order, dt, nstep, theta, country, band_y0=0.0):
    # --- Physical parameters ---
    D          = 0.5    # base diffusion coefficient  [km²/an]
    r          = 1.0    # base growth rate             [1/an]
    r_tilde    = 0.1    # mortality rate outside band  [1/an]
    K          = 3  # carrying capacity
    c          = 5.0    # climate shift speed (northward, y-axis) [km/an]

    # --- Terrain effect parameters ---
    alpha_slope = 3  # slope penalty: D(x) = D / (1 + alpha_slope * s/s_max) (s is slope in m/m, s_max is 95th percentile of slope across the domain)
    elev_opt    = 20 # altitude optimale de l'espèce [m]
    elev_width  = 30 # demi-largeur de la niche altitudinale [m]

    # ------------------------------------------------------------------
    # Mesh
    # ------------------------------------------------------------------
    gmsh_init("kpp_fisher")

    (elemType, nodeTags, nodeCoords, elemTags, elemNodeTags,
     bnds, bndsTags, bounds, center) = build_country_mesh(
        country, mesh_size=h, order=order
    )

    # --- changing climate parameters ---
    x_min, x_max, y_min, y_max = bounds
    L = x_max - x_min
    H = y_max - y_min
    L_hab = H / 6.0   # Habitat width (demi-largeur du band de climat favorable)

    # Gaussian IC centred at the lower-middle of the domain
    x0    = [(x_min + x_max) / 2.0 + 500, y_min + H * 0.1]
    sigma = min(L, H) / 8.0
    u0_max = 5.0


    # ------------------------------------------------------------------
    # Elevation model
    # ------------------------------------------------------------------
    proj        = pyproj.Proj("EPSG:3857")
    elev_model  = ElevationModel("src/italy.tif", proj, center_km=center)

    # Optional: visualise slope on mesh before simulation
    values = compute_node_field(nodeCoords, elev_model, field="elevation")
    #plot_mesh_2d(elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bndsTags, node_values=values, colorbar_label="Elevation (m)")

    # ------------------------------------------------------------------
    # DOF bookkeeping  (BUG FIX: use tag_to_index to map tags → rows)
    # ------------------------------------------------------------------
    max_tag     = int(np.max(nodeTags))
    tag_to_index = np.zeros(max_tag + 1, dtype=int)
    for i, tag in enumerate(nodeTags):
        tag_to_index[int(tag)] = i

    unique_dof_tags = np.unique(elemNodeTags)
    num_dofs        = len(unique_dof_tags)
    tag_to_dof      = np.full(max_tag + 1, -1, dtype=int)
    dof_coords      = np.zeros((num_dofs, 3))
    all_coords      = nodeCoords.reshape(-1, 3)

    for i, tag in enumerate(unique_dof_tags):
        tag_to_dof[int(tag)] = i
        dof_coords[i]        = all_coords[tag_to_index[int(tag)]]

    # KDTree for fast nearest-DOF lookup in kappa / r_fn
    dof_tree = KDTree(dof_coords[:, :2])

    # ------------------------------------------------------------------
    # Terrain fields at every DOF (computed once)
    # ------------------------------------------------------------------
    print("Computing elevation and slope at DOFs...")
    elev_at_dof  = np.array([elev_model.get_elevation(x[0], x[1]) for x in dof_coords])
    slope_at_dof = np.array([elev_model.get_slope   (x[0], x[1]) for x in dof_coords])
    slope_max    = np.percentile(slope_at_dof, 95)   # robust normalisation

    # ------------------------------------------------------------------
    # Spatially variable diffusion  D(x) — reduced on steep slopes
    # ------------------------------------------------------------------
    def kappa(x):
        _, idx = dof_tree.query(x[:2])
        s = slope_at_dof[idx] / slope_max        # ∈ [0, 1] roughly
        return D / (1.0 + alpha_slope * s)

    # ------------------------------------------------------------------
    # Spatially variable growth rate  r(x) — Gaussian niche in altitude
    # ------------------------------------------------------------------
    def r_fn(x):
        _, idx = dof_tree.query(x[:2])
        e = elev_at_dof[idx]
        return r * np.exp(-((e - elev_opt) ** 2) / (2.0 * elev_width ** 2))

    # ------------------------------------------------------------------
    # Quadrature, basis functions, Jacobians
    # ------------------------------------------------------------------
    xi, w, N, gN = prepare_quadrature_and_basis(elemType, order)
    jac, det, coords = get_jacobians(elemType, xi)

    # ------------------------------------------------------------------
    # Initial condition
    # ------------------------------------------------------------------
    U = np.array([u_init(x, x0, u0_max, sigma) for x in dof_coords], dtype=float)
    print(f"IC: max(U) = {U.max():.4f}  (should be ~{u0_max})")

    # ------------------------------------------------------------------
    # Mass matrix  M  (time-independent)
    # ------------------------------------------------------------------
    M_lil = assemble_mass(elemTags, elemNodeTags, det, w, N, tag_to_dof)
    M     = M_lil.tocsr()

    # ------------------------------------------------------------------
    # Time loop
    # ------------------------------------------------------------------
    display = VideoDisplay()
    fig, ax = display.get_figure()

    for step in range(nstep):
        if step % 10 == 0:
            print(f"  step {step+1}/{nstep}  ({100*step/nstep:.1f}%)  max(u)={U.max():.4f}")
        t = step * dt

        # Source terms frozen at U_n (semi-implicit)
        f_n = make_explicit_source(
            U, dof_tree, dof_coords, t,      c, L_hab, r_tilde, K, r_fn, band_y0
        )
        f_np1 = make_explicit_source(
            U, dof_tree, dof_coords, t + dt, c, L_hab, r_tilde, K, r_fn, band_y0
        )

        # Stiffness + RHS with spatially variable kappa
        K_lil_n,   F_n   = assemble_stiffness_and_rhs(
            elemTags, elemNodeTags, jac, det, coords, w, N, gN, kappa, f_n,   tag_to_dof
        )
        K_lil_np1, F_np1 = assemble_stiffness_and_rhs(
            elemTags, elemNodeTags, jac, det, coords, w, N, gN, kappa, f_np1, tag_to_dof
        )

        K_n   = K_lil_n.tocsr()
        K_np1 = K_lil_np1.tocsr()

        # θ-scheme — pure Neumann, no Dirichlet DOFs
        U = theta_step(
            M, K_np1, F_n, F_np1, U,
            dt=dt, theta=theta,
            dirichlet_dofs  = np.array([], dtype=int),
            dir_vals_np1    = np.array([], dtype=float),
        )

        # Enforce positivity
        U = np.maximum(U, 0.0)

        # --- Plot ---
        ax.clear()
        contour = plot_fe_solution_2d(
            elemNodeTags=elemNodeTags,
            nodeTags=nodeTags,
            nodeCoords=nodeCoords,
            U=U,
            tag_to_dof=tag_to_dof,
            show_mesh=False,
            ax=ax,
        )

        # Favourable band overlay (set axes limits first so axhspan is correct)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        plot_favourable_band(ax, t + dt, c, L_hab, band_y0=band_y0)

        ax.set_title(
            f"t = {t+dt:.2f} an  |  c = {c:.1f} km/an  |  "
            f"θ = {theta}  |  max(u) = {U.max():.3f}"
        )
        ax.set_xlabel("x  [km]")
        ax.set_ylabel("y  [km]")
        ax.set_aspect("equal")

        if not display.add_frame(fig, ax):
            print("Display closed — stopping.")
            break

    display.end()
    gmsh_finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-order",     type=int,   default=1)
    parser.add_argument("-hc",        type=float, default=20)
    parser.add_argument("--dt",       type=float, default=0.5)
    parser.add_argument("--nsteps",   type=int,   default=80)
    parser.add_argument("--theta",    type=float, default=1.0)
    parser.add_argument("--country",  type=str,   default="Italy")
    parser.add_argument("--band-y0",  type=float, default=-200.0,
                        help="Vertical starting position of the favourable band")
    args = parser.parse_args()
    main(args.hc, args.order, args.dt, args.nsteps, args.theta, args.country, args.band_y0)