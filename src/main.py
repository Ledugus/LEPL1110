import argparse
import numpy as np
import matplotlib.pyplot as plt

from gmsh_utils import (
    border_dofs_from_tags,
    build_2d_rectangle_mesh,
    gmsh_init,
    gmsh_finalize,
    prepare_quadrature_and_basis,
    get_jacobians,
    mesh1,
    mesh2,
)
from stiffness_node import assemble_stiffness_and_rhs_node_wise
from dirichlet import solve_dirichlet
from errors import compute_L2_H1_errors

from plot_utils import plot_mesh_2d, plot_fe_solution_2d
import gmsh
