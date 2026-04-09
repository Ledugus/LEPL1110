import numpy as np
import matplotlib.pyplot as plt
from dolfin import *
from matplotlib.animation import FuncAnimation

# ------------------------
# Parameters
# ------------------------
D = 0.1       # diffusion coefficient
r = 1.0       # growth rate
K = 1.0       # carrying capacity
T = 5.0       # final time
num_steps = 50
dt = T / num_steps

# ------------------------
# Mesh and Function Space
# ------------------------
nx, ny = 30, 30
mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, 'P', 1)

# Plot mesh
plt.figure()
plot(mesh)
plt.title("Mesh")
plt.show()

# ------------------------
# Initial condition (Gaussian)
# ------------------------
u_n = Function(V)

u0_expr = Expression(
    "exp(-((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5)) / 0.02)",
    degree=2
)

u_n.interpolate(u0_expr)

# ------------------------
# Variational problem
# ------------------------
u = Function(V)
v = TestFunction(V)

F = (u - u_n)/dt * v * dx \
    + D * dot(grad(u), grad(v)) * dx \
    - r * u * (1 - u/K) * v * dx

# ------------------------
# Prepare plot
# ------------------------
plt.figure()
p = plot(u_n)
plt.colorbar(p)

# Store frames
frames = []

# ------------------------
# Time-stepping
# ------------------------
for n in range(num_steps):
    solve(F == 0, u)
    
    # Update previous solution
    u_n.assign(u)
    
    # Store solution for animation
    frames.append(u.compute_vertex_values(mesh))

# ------------------------
# Animation
# ------------------------
fig, ax = plt.subplots()
coords = mesh.coordinates()

triangles = mesh.cells()

def update(frame):
    ax.clear()
    z = frames[frame]
    
    tpc = ax.tripcolor(coords[:,0], coords[:,1], triangles, z, shading='gouraud')
    ax.set_title(f"t = {frame*dt:.2f}")
    return tpc,

anim = FuncAnimation(fig, update, frames=num_steps, interval=100)

plt.show()