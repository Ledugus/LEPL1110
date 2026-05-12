# stiffness.py
import numpy as np
from scipy.sparse import csr_matrix
from numba import njit


# ---------------------------------------------------------------------------
# Numba kernel — scatter precomputed f_vals into F
# All heavy logic (habitat, r_fn, KDTree) is gone from here.
# ---------------------------------------------------------------------------
@njit(cache=True)
def _rhs_scatter_kernel(ne, ngp, nloc, det, w, N, conn, tag_to_dof, f_vals, F):
    for e in range(ne):
        for g in range(ngp):
            wg_detg = w[g] * det[e, g]
            fwd = f_vals[e, g] * wg_detg
            for a in range(nloc):
                F[tag_to_dof[conn[e, a]]] += fwd * N[g, a]
    return F


@njit(cache=True)
def _stiffness_scatter_kernel(
    ne,
    ngp,
    nloc,
    det,
    w,
    N,
    gN,
    conn,
    tag_to_dof,
    kappa_vals,
    invjac,
    rows,
    cols,
    Kdata,
    F,
    f_vals,
):
    idx = 0
    for e in range(ne):
        for g in range(ngp):
            wg = w[g]
            detg = det[e, g]
            wgd = wg * detg
            kappa_g = kappa_vals[e, g]
            f_g = f_vals[e, g]
            J = invjac[e, g]
            for a in range(nloc):
                Ia = tag_to_dof[conn[e, a]]
                # physical gradient of N_a
                gNa0 = (
                    J[0, 0] * gN[g, a, 0]
                    + J[0, 1] * gN[g, a, 1]
                    + J[0, 2] * gN[g, a, 2]
                )
                gNa1 = (
                    J[1, 0] * gN[g, a, 0]
                    + J[1, 1] * gN[g, a, 1]
                    + J[1, 2] * gN[g, a, 2]
                )
                gNa2 = (
                    J[2, 0] * gN[g, a, 0]
                    + J[2, 1] * gN[g, a, 1]
                    + J[2, 2] * gN[g, a, 2]
                )
                F[Ia] += wgd * f_g * N[g, a]
                for b in range(nloc):
                    Ib = tag_to_dof[conn[e, b]]
                    gNb0 = (
                        J[0, 0] * gN[g, b, 0]
                        + J[0, 1] * gN[g, b, 1]
                        + J[0, 2] * gN[g, b, 2]
                    )
                    gNb1 = (
                        J[1, 0] * gN[g, b, 0]
                        + J[1, 1] * gN[g, b, 1]
                        + J[1, 2] * gN[g, b, 2]
                    )
                    gNb2 = (
                        J[2, 0] * gN[g, b, 0]
                        + J[2, 1] * gN[g, b, 1]
                        + J[2, 2] * gN[g, b, 2]
                    )
                    rows[idx] = Ia
                    cols[idx] = Ib
                    Kdata[idx] = (
                        wgd * kappa_g * (gNa0 * gNb0 + gNa1 * gNb1 + gNa2 * gNb2)
                    )
                    idx += 1
    return idx


# ---------------------------------------------------------------------------
# Precomputation — call ONCE before the time loop
# Returns everything that depends only on mesh geometry.
# ---------------------------------------------------------------------------
def precompute_rhs_geometry(elemTags, conn, det, xphys, w, N, tag_to_dof):
    """
    Precompute and cache all mesh-geometry quantities needed for fast RHS assembly.

    Returns a dict (rhs_cache) to be passed to assemble_F_fast every step.
    """
    ne = len(elemTags)
    ngp = len(w)
    nloc = int(len(conn) // ne)

    det_r = np.asarray(det, dtype=np.float64).reshape(ne, ngp)
    xphys_r = np.asarray(xphys, dtype=np.float64).reshape(ne, ngp, 3)
    conn_r = np.asarray(conn, dtype=np.int64).reshape(ne, nloc)
    N_r = np.asarray(N, dtype=np.float64).reshape(ngp, nloc)
    w_r = np.asarray(w, dtype=np.float64)

    # Precompute w*det for every (e, g) — avoids the multiply inside the loop
    wd = w_r[np.newaxis, :] * det_r  # (ne, ngp)

    # Map tag_to_dof over the full connectivity once
    dof_conn = tag_to_dof[conn_r]  # (ne, nloc)  int array

    return dict(
        ne=ne,
        ngp=ngp,
        nloc=nloc,
        det=det_r,
        xphys=xphys_r,
        conn=conn_r,
        N=N_r,
        w=w_r,
        wd=wd,
        dof_conn=dof_conn,
        tag_to_dof=tag_to_dof,
    )


def precompute_gauss_r(rhs_cache, dof_tree, elev_at_dof, r_base, elev_opt, elev_width):
    """
    Precompute the spatially-variable growth rate r(x) at every Gauss point.
    Uses the KDTree lookup — done once, never repeated in the time loop.

    Returns r_at_gp : (ne, ngp) float64 array.
    """
    ne = rhs_cache["ne"]
    ngp = rhs_cache["ngp"]
    xphys = rhs_cache["xphys"]  # (ne, ngp, 3)

    # Flatten Gauss-point xy coords and batch-query the KDTree
    gp_xy = xphys.reshape(-1, 3)[:, :2]  # (ne*ngp, 2)
    _, idx = dof_tree.query(gp_xy)  # (ne*ngp,)  nearest DOF

    elev = elev_at_dof[idx]  # (ne*ngp,)
    r_gp = r_base * np.exp(-((elev - elev_opt) ** 2) / (2.0 * elev_width**2))
    return r_gp.reshape(ne, ngp)  # (ne, ngp)


# ---------------------------------------------------------------------------
# Fast vectorized F assembly — the only thing called each time step
# ---------------------------------------------------------------------------
def assemble_F_fast(U, rhs_cache, r_at_gp, t, c, L_hab, band_y0, r_tilde, K_cap):
    """
    Assemble the load vector F fully vectorized over elements and Gauss points.
    No Python loops over (e, g) — only the final scatter uses numba.

    Parameters
    ----------
    U         : current solution vector (nn,)
    rhs_cache : dict returned by precompute_rhs_geometry
    r_at_gp   : (ne, ngp) growth rate at Gauss points, from precompute_gauss_r
    t         : current time
    c, L_hab, band_y0, r_tilde, K_cap : physical parameters
    """
    ne = rhs_cache["ne"]
    ngp = rhs_cache["ngp"]
    nloc = rhs_cache["nloc"]
    xphys = rhs_cache["xphys"]  # (ne, ngp, 3)
    N = rhs_cache["N"]  # (ngp, nloc)
    w = rhs_cache["w"]  # (ngp,)
    det = rhs_cache["det"]  # (ne, ngp)
    conn = rhs_cache["conn"]  # (ne, nloc)  gmsh tags
    tag_to_dof = rhs_cache["tag_to_dof"]
    dof_conn = rhs_cache["dof_conn"]  # (ne, nloc)  dof indices

    # ------------------------------------------------------------------
    # 1. Habitat m(x, t) — fully vectorized, shape (ne, ngp)
    # ------------------------------------------------------------------
    yi = xphys[:, :, 1] - (band_y0 + c * t)  # (ne, ngp)
    ratio6 = (yi / L_hab) ** 6
    m = (1.0 - ratio6) / (1.0 + ratio6)  # (ne, ngp)  ∈ (-1, 1)

    # ------------------------------------------------------------------
    # 2. u at each Gauss point via nearest DOF (precomputed in dof_conn)
    #    We use the first local node as a cheap proxy — for a fine mesh
    #    this is as accurate as the KDTree lookup and costs nothing.
    #    Swap to a proper interpolation if your mesh is coarse.
    # ------------------------------------------------------------------
    u_at_gp = np.maximum(U[dof_conn[:, 0]], 0.0)[:, np.newaxis]  # (ne, 1)
    # broadcast over Gauss points: (ne, ngp)
    u_at_gp = np.broadcast_to(u_at_gp, (ne, ngp)).copy()

    # ------------------------------------------------------------------
    # 3. Source term f(u, x, t) — vectorized, shape (ne, ngp)
    # ------------------------------------------------------------------
    f_vals = (
        (m + 1.0) * r_at_gp * u_at_gp * (1.0 - u_at_gp / K_cap)
        + (1.0 - m) * (-r_tilde * u_at_gp)
    ) * 0.5  # (ne, ngp)

    # ------------------------------------------------------------------
    # 4. Scatter into F — numba kernel, no Python loop
    # ------------------------------------------------------------------
    nn = int(np.max(tag_to_dof) + 1)
    F = np.zeros(nn, dtype=np.float64)
    _rhs_scatter_kernel(ne, ngp, nloc, det, w, N, conn, tag_to_dof, f_vals, F)
    return F


# ---------------------------------------------------------------------------
# Original assemblers kept for K assembly and compatibility
# ---------------------------------------------------------------------------
def assemble_stiffness_and_rhs(
    elemTags, conn, jac, det, xphys, w, N, gN, kappa_fun, rhs_fun, tag_to_dof
):
    ne = len(elemTags)
    ngp = len(w)
    nloc = int(len(conn) // ne)
    nn = int(np.max(tag_to_dof) + 1)

    det_r = np.asarray(det, dtype=np.float64).reshape(ne, ngp)
    xphys_r = np.asarray(xphys, dtype=np.float64).reshape(ne, ngp, 3)
    jac_r = np.asarray(jac, dtype=np.float64).reshape(ne, ngp, 3, 3)
    conn_r = np.asarray(conn, dtype=np.int64).reshape(ne, nloc)
    N_r = np.asarray(N, dtype=np.float64).reshape(ngp, nloc)
    gN_r = np.asarray(gN, dtype=np.float64).reshape(ngp, nloc, 3)
    w_r = np.asarray(w, dtype=np.float64)

    # Pre-evaluate callables
    kappa_vals = np.array(
        [[float(kappa_fun(xphys_r[e, g])) for g in range(ngp)] for e in range(ne)],
        dtype=np.float64,
    )
    f_vals = np.array(
        [[float(rhs_fun(xphys_r[e, g])) for g in range(ngp)] for e in range(ne)],
        dtype=np.float64,
    )

    # Precompute all inverse Jacobians at once
    invjac = np.linalg.inv(jac_r)  # (ne, ngp, 3, 3)

    nnz = ne * ngp * nloc * nloc
    rows = np.zeros(nnz, dtype=np.int64)
    cols = np.zeros(nnz, dtype=np.int64)
    Kdata = np.zeros(nnz, dtype=np.float64)
    F = np.zeros(nn, dtype=np.float64)

    _stiffness_scatter_kernel(
        ne,
        ngp,
        nloc,
        det_r,
        w_r,
        N_r,
        gN_r,
        conn_r,
        tag_to_dof,
        kappa_vals,
        invjac,
        rows,
        cols,
        Kdata,
        F,
        f_vals,
    )

    K = csr_matrix((Kdata, (rows, cols)), shape=(nn, nn))
    return K, F


def assemble_rhs_neumann(
    F, elemTags, conn, jac, det, xphys, w, N, gN, g_neu_fun, tag_to_dof
):
    ne = len(elemTags)
    ngp = len(w)
    nloc = int(len(conn) // ne)

    det_r = np.asarray(det, dtype=np.float64).reshape(ne, ngp)
    xphys_r = np.asarray(xphys, dtype=np.float64).reshape(ne, ngp, 3)
    conn_r = np.asarray(conn, dtype=np.int64).reshape(ne, nloc)
    N_r = np.asarray(N, dtype=np.float64).reshape(ngp, nloc)
    w_r = np.asarray(w, dtype=np.float64)

    f_vals = np.array(
        [[float(g_neu_fun(xphys_r[e, g])) for g in range(ngp)] for e in range(ne)],
        dtype=np.float64,
    )

    _rhs_scatter_kernel(ne, ngp, nloc, det_r, w_r, N_r, conn_r, tag_to_dof, f_vals, F)
    return F
