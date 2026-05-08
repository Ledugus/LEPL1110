
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from gmsh_utils import gmsh_init, gmsh_finalize
from mesh import build_country_mesh
from simulation import simulate


# ---------------------------------------------------------------------------
# Paramètres communs à tous les tests
# ---------------------------------------------------------------------------
COUNTRY   = "Italy"
ORDER     = 1
THETA     = 1       # Euler implicite — stable pour tous les dt
D         = 0.5
C         = 15    # vitesse climat  [km/an]
BAND_Y0   = -600.0
CLIMATE   = "warming"
T_END     = 10.0    # durée totale de la simulation [an]



def run_one(h, dt, verbose=True):
    """
    Lance une simulation avec maillage h et pas de temps dt.
    Retourne la population totale finale P(T).
    """
    nsteps = max(1, int(round(T_END / dt)))
    nsteps : 1
    dt_eff = T_END / nsteps          # dt légèrement ajusté pour tomber pile sur T_END

    if verbose:
        print(f"  h={h:6.1f} km  dt={dt_eff:.4f} an  nsteps={nsteps}")

    gmsh_init("kpp_conv")
    mesh = build_country_mesh(COUNTRY, mesh_size=h, order=ORDER)
    _, total_pops = simulate(
        ORDER, dt_eff, nsteps, THETA, mesh,
        D=D, c=C, band_y0=BAND_Y0, climate=CLIMATE, show=False,
    )
    gmsh_finalize()

    return total_pops





def test_spatial_convergence_integrated():
    """
    On raffine h (taille de maille) en gardant dt très petit et fixe.
    Référence : solution sur le maillage le plus fin.
    Erreur mesurée par intégrale temporelle de |P_h(t) - P_ref(t)|.
    """
    from scipy.integrate import trapezoid

    print("\n" + "="*60)
    print("TEST DE CONVERGENCE SPATIALE")
    print("="*60)

    h_values = [80, 60, 40, 30, 20, 15]
    DT_FIXED = 0.2
    nsteps   = max(1, int(round(T_END / DT_FIXED)))
    times    = np.linspace(0, T_END, nsteps + 1)

    trajectories = []
    for h in h_values:
        traj = run_one(h, DT_FIXED)          # retourne total_pops complet
        trajectories.append(np.array(traj))
        print(f"    → P(T) = {traj[-1]:.6f}")

    # Référence = maillage le plus fin
    p_ref_traj = trajectories[-1]
    p_ref_final = p_ref_traj[-1]
    norm = trapezoid(p_ref_traj, times)      # pour normaliser l'erreur

    # Erreur intégrée en temps (normalisée) — métrique principale
    errors_int = []
    for traj in trajectories[:-1]:
        E = trapezoid(np.abs(traj - p_ref_traj), times) / norm
        errors_int.append(E)

    

    h_conv = h_values[:-1]


   




    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)

    colors = plt.cm.Blues(np.linspace(0.35, 0.95, len(h_values)))

    for i, (h, traj) in enumerate(zip(h_values, trajectories)):
        lw = 2.5 if i == len(h_values) - 1 else 1.5
        ls = "--" if i == len(h_values) - 1 else "-"
        ax.plot(times, traj, color=colors[i], linewidth=lw, linestyle=ls,
                label=f"h={h} km")
    ax.set_xlabel("Temps [an]")
    ax.set_ylabel("Population totale P(t)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)


    return h_values, trajectories



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_spatial_convergence_integrated()