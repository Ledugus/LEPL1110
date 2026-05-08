import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from gmsh_utils import gmsh_init, gmsh_finalize
from simulation import simulate
from mesh import build_country_mesh


def save_initial_condition(
    D,
    h=20,
    nsteps=10,
    order=1,
    dt=0.5,
    country="Italy",
    theta=1,
    climate="warming",
    band_y0=-600,
):
    gmsh_init("kpp_fisher")
    mesh = build_country_mesh(country, mesh_size=h, order=order)
    c = 0  # stationary climate
    U_values, total_pops = simulate(
        order,
        dt,
        nsteps,
        theta,
        mesh,
        D,
        c,
        band_y0,
        climate,
        show=False,
    )
    filename = f"data/initial_D={D:.1f}.npy"
    np.save(filename, U_values[-1])
    print(f"Saved initial condition to '{filename}', total_pop={total_pops[-1]}")
    gmsh_finalize()
    return U_values[-1]


def test_velocity(
    D,
    h=20,
    nsteps=100,
    order=1,
    dt=0.5,
    country="Italy",
    theta=1,
    climate="warming",
    band_y0=-600,
):

    filename = f"data/initial_D={D:.1f}.npy"
    if os.path.exists(filename):
        initial_condition_path = filename
        initial_condition = np.load(initial_condition_path)
    else:
        initial_condition = save_initial_condition(
            D,
            h=h,
            order=order,
            dt=dt,
            country=country,
            theta=theta,
            climate=climate,
            band_y0=band_y0,
        )

    gmsh_init("kpp_fisher")

    mesh = build_country_mesh(country, mesh_size=h, order=order)
    times = np.arange(nsteps + 1) * dt
    tested_velocities = [5, 10, 15]
    for c in tested_velocities:
        _, total_pops = simulate(
            order,
            dt,
            nsteps,
            theta,
            mesh,
            D,
            c,
            band_y0,
            climate,
            initial_condition=initial_condition.copy(),
            show=False,
        )
        plt.plot(times, total_pops, label=f"c={c}")

    gmsh_finalize()
    plt.title(f"Population totale au cours du temps, D={D:.1f}")
    plt.ylabel("Population totale [individus]")
    plt.xlabel("Temps [années]")
    plt.legend()
    plt.savefig(f"figures/velocity_D={D:.1f}.pdf")
    plt.show()


if __name__ == "__main__":
    test_velocity(0.5, nsteps=100)
