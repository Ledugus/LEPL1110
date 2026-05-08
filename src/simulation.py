import numpy as np
from scipy.spatial import KDTree
from scipy.sparse.linalg import factorized
from mesh import *
from altitude import *

from gmsh_utils import (
    gmsh_init,
    gmsh_finalize,
    prepare_quadrature_and_basis,
    get_jacobians,
)
from stiffness import assemble_stiffness_and_rhs, assemble_rhs_only
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
def habitat_réchauffement(x, t, c, L_hab, band_y0=0.0):
    yi = x[1] - (band_y0 + c * t)
    exp = 6
    return (1 - (yi / L_hab) ** exp) / (1 + (yi / L_hab) ** exp), -yi


def habitat_saison(x, t, c, L_hab, band_y0=0.0):
    yi = x[1] - (band_y0 * np.cos(t * 2 * np.pi))
    exp = 6
    return (1 - (yi / L_hab) ** exp) / (1 + (yi / L_hab) ** exp), -yi


# ---------------------------------------------------------------------------
# Nonlinear source term  f(u, x, t)
# ---------------------------------------------------------------------------
def f_source_binaire(
    u, x, t, c, L_hab, r_tilde, K_cap, r_fn, band_y0=0.0, habitat=habitat_réchauffement
):
    m = habitat(x, t, c, L_hab, band_y0=band_y0)[0]
    if m > 0:
        r_x = r_fn(x)
        return r_x * u * (1.0 - u / K_cap)
    else:
        return -r_tilde * u


def f_source_non_binaire(
    u, x, t, c, L_hab, r_tilde, K_cap, r_fn, band_y0=0.0, habitat=habitat_réchauffement
):
    m = habitat(x, t, c, L_hab, band_y0=band_y0)[0]
    r_x = r_fn(x)
    return ((m + 1) * r_x * u * (1.0 - u / K_cap) + (1 - m) * (-r_tilde * u)) / 2


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
def make_explicit_source(
    U,
    dof_tree,
    dof_coords,
    t,
    c,
    L_hab,
    r_tilde,
    K,
    r_fn,
    band_y0=0.0,
    habitat=habitat_réchauffement,
):
    def _f(x):
        _, idx = dof_tree.query(x[:2])
        u_n = max(U[idx], 0.0)
        return f_source(
            u_n, x, t, c, L_hab, r_tilde, K, r_fn, band_y0=band_y0, habitat=habitat
        )

    return _f


# ---------------------------------------------------------------------------
# Favourable band overlay on plot
# ---------------------------------------------------------------------------
def plot_favourable_band(ax, t, c, L_hab, band_y0=0.0, habitat=habitat_réchauffement):
    y_center = habitat((0, 0), t, c, L_hab, band_y0=band_y0)[1]
    y0 = y_center - L_hab
    y1 = y_center + L_hab
    ax.axhspan(y0, y1, alpha=0.14, color="limegreen", zorder=6)
    ax.axhline(y0, linestyle="--", linewidth=1.0, color="forestgreen", zorder=7)
    ax.axhline(y1, linestyle="--", linewidth=1.0, color="forestgreen", zorder=7)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def simulate(
    order,
    dt,
    nstep,
    theta,
    mesh,
    D=0.5,
    c=5,
    band_y0=0.0,
    climate="warming",
    initial_condition=None,
    show=False,
):

    # --- Physical parameters ---
    D = D  # base diffusion coefficient  [km²/an]
    r = 1.0  # base growth rate             [1/an]
    r_tilde = 1  # mortality rate outside band  [1/an]
    K_cap = 3  # carrying capacity
    c = c  # climate shift speed (northward, y-axis) [km/an]

    # --- Terrain effect parameters ---
    elev_opt = 20  # altitude optimale de l'espèce [m]
    elev_width = 30  # demi-largeur de la niche altitudinale [m]

    if climate == "seasonal":
        dt /= 50
        D *= 50  # hypothèse d'une espèce d'animal avec une dynamique plus rapide qu'une plante
        r *= 50
        r_tilde *= 50
        habitat = habitat_saison
    elif climate == "warming":
        habitat = habitat_réchauffement
    else:
        raise ValueError(f"Unknown climate type: {climate}")

    # ------------------------------------------------------------------
    # Mesh
    # ------------------------------------------------------------------
    (
        elemType,
        nodeTags,
        nodeCoords,
        elemTags,
        elemNodeTags,
        bnds,
        bndsTags,
        bounds,
        center,
    ) = mesh

    # --- changing climate parameters ---
    x_min, x_max, y_min, y_max = bounds
    L = x_max - x_min
    H = y_max - y_min
    L_hab = H / 10  # Habitat width (demi-largeur du band de climat favorable)

    # Gaussian IC centred at the lower-middle of the domain
    x0 = [(x_min + x_max) / 2.0 + 500, y_min + H * 0.1]
    sigma = min(L, H) / 8.0
    u0_max = K_cap

    # ------------------------------------------------------------------
    # Elevation model
    # ------------------------------------------------------------------
    print("Getting elevation model...", end="")
    proj = pyproj.Proj("EPSG:3857")
    print("Elev model...", end="")
    elev_model = ElevationModel(
        "src/world.tif", proj, center_km=center,
        bounds_km=(x_min, x_max, y_min, y_max)   # déjà calculés juste avant
    )
    print("done")

    # Optional: visualise slope on mesh before simulation
    # values = compute_node_field(nodeCoords, elev_model, field="elevation")
    # plot_mesh_2d(elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bndsTags, node_values=values, colorbar_label="Elevation (m)")

    # ------------------------------------------------------------------
    # DOF bookkeeping
    # ------------------------------------------------------------------
    max_tag = int(np.max(nodeTags))
    tag_to_index = np.zeros(max_tag + 1, dtype=int)
    for i, tag in enumerate(nodeTags):
        tag_to_index[int(tag)] = i

    unique_dof_tags = np.unique(elemNodeTags)
    num_dofs = len(unique_dof_tags)
    tag_to_dof = np.full(max_tag + 1, -1, dtype=int)
    dof_coords = np.zeros((num_dofs, 3))
    all_coords = nodeCoords.reshape(-1, 3)

    for i, tag in enumerate(unique_dof_tags):
        tag_to_dof[int(tag)] = i
        dof_coords[i] = all_coords[tag_to_index[int(tag)]]

    # KDTree for fast nearest-DOF lookup in kappa / r_fn
    dof_tree = KDTree(dof_coords[:, :2])

    # ------------------------------------------------------------------
    # Terrain fields at every DOF (computed once)
    # ------------------------------------------------------------------
    print("Computing elevation at DOFs...", end="")
    elev_at_dof = np.array([elev_model.get_elevation(x[0], x[1]) for x in dof_coords])
    print("done")

    def kappa(x):
        return D

    # ------------------------------------------------------------------
    # Spatially variable growth rate  r(x) — Gaussian niche in altitude
    # ------------------------------------------------------------------
    def r_fn(x):
        _, idx = dof_tree.query(x[:2])
        e = elev_at_dof[idx]
        return r * np.exp(-((e - elev_opt) ** 2) / (2.0 * elev_width**2))

    # ------------------------------------------------------------------
    # Quadrature, basis functions, Jacobians
    # ------------------------------------------------------------------
    print("Quadrature, basis functions", end="...")
    xi, w, N, gN = prepare_quadrature_and_basis(elemType, order)
    jac, det, coords = get_jacobians(elemType, xi)
    print("done")
    # ------------------------------------------------------------------
    # Initial condition
    # ------------------------------------------------------------------
    if initial_condition is not None:
        U = initial_condition
        print("Initial condition : loaded")
    else:
        U = np.array([u_init(x, x0, u0_max, sigma) for x in dof_coords], dtype=float)
        print("Initial condition : Gaussian")

    # ------------------------------------------------------------------
    # Mass matrix  M  (time-independent)
    # ------------------------------------------------------------------
    print("Assemble mass", end="...")
    M_lil = assemble_mass(elemTags, elemNodeTags, det, w, N, tag_to_dof)
    M = M_lil.tocsr()
    print("done")

    print("Assemble stiffness...", end="")
    K_lil, _ = assemble_stiffness_and_rhs(
        elemTags,
        elemNodeTags,
        jac,
        det,
        coords,
        w,
        N,
        gN,
        kappa,
        lambda x: 0.0,
        tag_to_dof,
    )
    K = K_lil.tocsr()
    print("done")
    # ------------------------------------------------------------------
    # Precompute theta-scheme matrices and factorize A once
    # ------------------------------------------------------------------
    A = M + theta * dt * K
    B = M - (1.0 - theta) * dt * K
    print("Factorizing A...", end="")
    solve = factorized(A.tocsc())
    print("done")
    # ------------------------------------------------------------------
    # Time loop
    # ------------------------------------------------------------------
    total_populations = []
    total_populations.append(M.sum(axis=1).A1 @ U)
    values = []
    values.append(U)
    if show:
        display = VideoDisplay()
        fig, ax = display.get_figure()

    print("Entering time loop")
    for step in range(nstep):
        if step % 10 == 0:
            print(f"  step {step+1}/{nstep}  ({100*step/nstep:.1f}%)")
        t = step * dt

        # Source terms frozen at U_n (semi-implicit)
        f_n = make_explicit_source(
            U,
            dof_tree,
            dof_coords,
            t,
            c,
            L_hab,
            r_tilde,
            K_cap,
            r_fn,
            band_y0,
            habitat=habitat,
        )
        f_np1 = make_explicit_source(
            U,
            dof_tree,
            dof_coords,
            t + dt,
            c,
            L_hab,
            r_tilde,
            K_cap,
            r_fn,
            band_y0,
            habitat=habitat,
        )

        # Stiffness + RHS with spatially variable kappa
        # Only assemble F, not K
        F_n = assemble_rhs_only(
            elemTags, elemNodeTags, det, coords, w, N, f_n, tag_to_dof
        )
        F_np1 = assemble_rhs_only(
            elemTags, elemNodeTags, det, coords, w, N, f_np1, tag_to_dof
        )

        # theta-scheme with precomputed B and prefactored A
        rhs = B @ U + dt * (theta * F_np1 + (1.0 - theta) * F_n)
        U = solve(rhs)

        # Enforce positivity
        U = np.maximum(U, 0.0)
        total_populations.append(M.sum(axis=1).A1 @ U)
        values.append(U)

        # --- Plot ---
        if show:
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
            plot_favourable_band(ax, t + dt, c, L_hab, band_y0=band_y0, habitat=habitat)

            ax.set_title(
                f"t = {t+dt:.2f} an  |  c = {c:.1f} km/an  |  "
                f"θ = {theta}  | Pop={total_populations[-1]:.2f}"
            )
            ax.set_xlabel("x  [km]")
            ax.set_ylabel("y  [km]")
            ax.set_aspect("equal")

            if not display.add_frame(fig, ax):
                print("Display closed — stopping.")
                break

    if show:
        display.end()
    print("Finished simulation")
    return values, total_populations
