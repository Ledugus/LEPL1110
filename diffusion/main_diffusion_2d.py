# main_diffusion_1d.py
import argparse
import numpy as np

from gmsh_utils import (
    gmsh_init, gmsh_finalize,
    prepare_quadrature_and_basis, get_jacobians,
    border_dofs_from_tags, 
    build_2d_rectangle_mesh
)
from stiffness import assemble_stiffness_and_rhs
from mass import assemble_mass
from dirichlet import theta_step
from plot_utils import setup_interactive_figure, plot_mesh_2d, plot_fe_solution_2d
import matplotlib.pyplot as plt


def main(L, H, h, order):
    r_tilde = 1
    r = 1
    def size_field(x, y): return h

    parser = argparse.ArgumentParser(description="Diffusion 1D with theta-scheme (Gmsh high-order FE)")
    parser.add_argument("-order", type=int, default=1)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=1.0e-03)
    parser.add_argument("--nsteps", type=int, default=500)
    args = parser.parse_args()

    gmsh_init("panpan_2d")

    dt = args.dt
    nstep = args.nsteps
    T = dt * nstep

    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags = build_2d_rectangle_mesh(L=L, H=H, size_field=size_field, order=order)

    plot_mesh_2d(elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags)

    unique_dofs_tags = np.unique(elemNodeTags)
    num_dofs = len(unique_dofs_tags)
    max_tag = int(np.max(nodeTags))
    dof_coords = np.zeros((num_dofs, 3))
    all_coords = nodeCoords.reshape(-1, 3)
    tag_to_dof = np.full(max_tag + 1, -1, dtype=int)
    for i, tag in enumerate(unique_dofs_tags):
        tag_to_dof[int(tag)] = i
        dof_coords[i] = all_coords[i]

    xi, w, N, gN = prepare_quadrature_and_basis(elemType, args.order)
    jac, det, coords = get_jacobians(elemType, xi)

    def kappa(x): return 1.0
    def f_source(u, x, t, m):
        if m(x, t) > 0:
            return r * u * (1 - u / K)
        else:
            return -r_tilde * u
    def u0(x): return 0.0
    def u_outer(x, t): return 1.0 * np.sin(11 * np.pi * t / T)

    M_lil = assemble_mass(elemTags, elemNodeTags, det, w, N, tag_to_dof)
    K_lil, F0 = assemble_stiffness_and_rhs(elemTags, elemNodeTags, jac, det, coords, w, N, gN, kappa, lambda x: f_source(x, 0), tag_to_dof)

    M = M_lil.tocsr()
    K = K_lil.tocsr()

    U = np.array([u0(x) for x in dof_coords], dtype=float)

    outer_dofs = border_dofs_from_tags(bnds_tags[0], tag_to_dof)
    dir_dofs = outer_dofs

    _, ax = setup_interactive_figure()

   # Condition initiale : gaussienne
    U = u0 * np.exp(-np.linalg.norm(dof_coords - x0, axis=1)**2 / (2 * sigma**2))

# Boucle temporelle
    for n in range(n_steps):
        t = n * dt

        # Terme source évalué sur U^n (explicite)
        F = assemble_rhs(U, t, m, r, K_cap, r_tilde, ...)

        # Système implicite : (M + dt*K_stiff) * U_new = M*U + dt*F
        A = M + dt * K_stiff
        b = M @ U + dt * F

        # Résolution directe (pas de Dirichlet à imposer !)
        U = spsolve(A, b)

        # Clipping pour éviter u < 0
        U = np.maximum(U, 0.0)
        ax.clear()
        plot_fe_solution_2d(
            elemNodeTags=elemNodeTags,
            nodeTags=nodeTags,
            nodeCoords=nodeCoords,
            U=U,
            tag_to_dof=tag_to_dof,
            show_mesh=False,
            ax=ax
        )
        ax.set_title(f"t = {step * args.dt:.4f}   (theta={args.theta})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.axis('equal')
        plt.pause(0.01)

    gmsh_finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poisson 2D with Gmsh FE")
    parser.add_argument("-order", type=int, default=1)
    parser.add_argument("-L", type=float, default=1.0)
    parser.add_argument("-H", type=float, default=1.0)
    parser.add_argument("-hc", type=float, default=0.1)
    args = parser.parse_args()
    main(args.L, args.H, args.hc, args.order)