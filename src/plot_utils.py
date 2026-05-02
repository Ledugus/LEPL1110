# plot_utils.py
import os, sys
import subprocess
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import gmsh
from abc import ABC, abstractmethod

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None


def plot_fe_solution_high_order(
    elemType, elemNodeTags, nodeCoords, U, M=80, show_nodes=False, ax=None, label=None
):
    """
    Plot 1D high-order FE solution by sampling each element and evaluating gmsh basis.
    Assumes U is aligned with gmsh's compact node ordering (0..nn-1).
    """
    _, _, _, nloc, _, _ = gmsh.model.mesh.getElementProperties(elemType)

    u = np.linspace(-1.0, 1.0, int(M))
    pts3 = np.zeros((len(u), 3), dtype=float)
    pts3[:, 0] = u
    uvw = pts3.reshape(-1).tolist()

    _, bf, _ = gmsh.model.mesh.getBasisFunctions(elemType, uvw, "Lagrange")
    N = np.asarray(bf, dtype=float).reshape(len(u), nloc)

    if ax is None:
        fig, ax = plt.subplots()

    ne = int(len(elemNodeTags) // nloc)
    _, _, coords_flat = gmsh.model.mesh.getJacobians(elemType, uvw)
    coords = np.asarray(coords_flat, dtype=float).reshape(ne, len(u), 3)

    for e in range(ne):
        tags_e = np.asarray(elemNodeTags[e * nloc : (e + 1) * nloc], dtype=int) - 1
        Ue = U[tags_e]

        x = coords[e, :, 0]
        uh = N @ Ue

        order = np.argsort(x)
        ax.plot(x[order], uh[order], label=label if (e == 0) else None)

    if show_nodes:
        Xn = np.asarray(nodeCoords, dtype=float).reshape(-1, 3)[:, 0]
        ax.plot(Xn, U, "o", markersize=4)

    ax.set_xlabel("x")
    ax.set_ylabel("u_h")
    ax.grid(True)
    return ax


def setup_interactive_figure(xlim=None, ylim=None):
    plt.ion()
    fig, ax = plt.subplots()
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    return fig, ax


def plot_mesh_2d(
    elemType,
    nodeTags,
    nodeCoords,
    elemTags,
    elemNodeTags,
    bnds,
    bnds_tags,
    tag_to_index=None,
    node_values=None, 
    colorbar_label="Field value",  
):

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri

    coords = nodeCoords.reshape(-1, 3)
    x = coords[:, 0]
    y = coords[:, 1]

    if tag_to_index is None:
        max_node_tag = int(np.max(nodeTags))
        tag_to_index = np.zeros(max_node_tag + 1, dtype=int)
        for i, tag in enumerate(nodeTags):
            tag_to_index[int(tag)] = i

    num_elements = len(elemTags)
    nodes_per_elem = len(elemNodeTags) // num_elements

    # keep only triangle corner nodes
    all_nodes = elemNodeTags.reshape(num_elements, nodes_per_elem)
    corner_nodes = all_nodes[:, :3]

    tri_indices = tag_to_index[corner_nodes.astype(int)]

    mesh_triang = tri.Triangulation(x, y, tri_indices)

    fig, ax = plt.subplots(figsize=(10, 5))

   
    if node_values is not None:
        sc = ax.tripcolor(
            mesh_triang,
            node_values,
            shading="gouraud",
            cmap="viridis"
        )
        plt.colorbar(sc, ax=ax, label=colorbar_label)
    else:
        ax.triplot(mesh_triang, color="black", lw=0.5, alpha=0.4)

    # ---------------------------------------
    # Boundary points
    # ---------------------------------------
    colors = ["red", "darkblue", "orange", "mediumpurple", "pink"]

    for i, (name, dim) in enumerate(bnds):
        tags = bnds_tags[i]
        indices = tag_to_index[tags.astype(int)]

        ax.scatter(
            x[indices],
            y[indices],
            label=name,
            s=15,
            zorder=3,
            marker="o",
            facecolor="None",
            edgecolor=colors[i % len(colors)],
        )

    ax.set_aspect("equal")

    ax.legend(
        frameon=True,
        framealpha=1,
        ncols=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
    )

    plt.axis("off")
    plt.savefig("mesh.pdf", bbox_inches="tight")
    plt.show()


def plot_fe_solution_2d(
    elemNodeTags,
    nodeCoords,
    nodeTags,
    U,
    tag_to_dof,
    show_mesh=False,
    ax=None,
    label=None,
):

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # 1. Map coordinates to our compact DoF indices
    num_dofs = len(U)
    coords_mapped = np.zeros((num_dofs, 2))
    all_coords = nodeCoords.reshape(-1, 3)

    for i, tag in enumerate(nodeTags):
        dof_idx = tag_to_dof[int(tag)]
        if dof_idx != -1:
            coords_mapped[dof_idx] = all_coords[i, :2]

    x = coords_mapped[:, 0]
    y = coords_mapped[:, 1]

    # 2. Determine nodes per element dynamically
    total_nodes_in_elems = len(elemNodeTags)
    # Standard Lagrange triangle node counts: Order 1=3, Order 2=6, Order 3=10, Order 4=15
    for possible_n in [3, 6, 10, 15, 21]:
        if total_nodes_in_elems % possible_n == 0:
            nodes_per_elem = possible_n
            break
    # 3. Reshape and extract ONLY the 3 corner nodes for Matplotlib
    conn_reshaped = elemNodeTags.reshape(-1, nodes_per_elem)
    # Map the GMSH tags to our 0...N-1 indices
    triangles = tag_to_dof[conn_reshaped[:, :3].astype(int)]
    # 4. Plotting
    U = np.array(U).flatten()
    contour = ax.tricontourf(
        x, y, triangles, U, levels=100, cmap="seismic", vmin=-2.0, vmax=2.0
    )

    if show_mesh:
        ax.triplot(x, y, triangles, color="white", linewidth=0.2, alpha=0.3)

    return contour


class Display(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def add_frame(self, fig, ax, step):
        pass

    @abstractmethod
    def get_figure(self):
        pass

    @abstractmethod
    def end(self):
        pass


class VideoDisplay(Display):
    def __init__(self, keep_frames=False) -> None:
        super().__init__()
        self.filename = "simulation.mp4"
        self.keep_frames = keep_frames
        self.last_step = 0
        os.makedirs("frames", exist_ok=True)

    def add_frame(self, fig, ax):
        fig.savefig(
            f"frames/frame_{self.last_step:04d}.png", dpi=100, bbox_inches="tight"
        )

        self.last_step += 1
        return True

    def get_figure(self):
        return plt.subplots()

    def end(self):
        video_created = False
        # Try to find ffmpeg in PATH first, then fall back to the bundled binary.
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None and imageio_ffmpeg is not None:
            try:
                ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                ffmpeg_path = None
        
        if ffmpeg_path is None:
            print(f"Warning: ffmpeg not found in PATH. Video file '{self.filename}' was not created.")
            print("To create videos, install ffmpeg: https://ffmpeg.org/download.html")
            print("On Windows with Chocolatey: choco install ffmpeg")
        else:
            try:
                subprocess.run(
                    [
                        ffmpeg_path,
                        "-y",
                        "-framerate",
                        "30",
                        "-i",
                        "frames/frame_%04d.png",
                        "-vf",
                        "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2",
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        self.filename,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
                video_created = True
            except subprocess.CalledProcessError as e:
                print(f"Warning: ffmpeg failed to create video. Error: {e}")
            except Exception as e:
                print(f"Warning: Error creating video: {e}")
        
        if video_created:
            print("Video written to file", self.filename)
        
        if not self.keep_frames:
            for f in glob.glob("frames/*.png"):
                os.remove(f)

        if video_created:
            try:
                if sys.platform == "win32":
                    os.startfile("simulation.mp4")
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", "simulation.mp4"])
                else:
                    subprocess.Popen(
                        ["xdg-open", "simulation.mp4"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
            except (FileNotFoundError, OSError):
                pass  # Video viewer not available; file was created successfully


class InteractiveDisplay(Display):
    def __init__(self) -> None:
        super().__init__()
        self.fig, self.ax = setup_interactive_figure()

    def get_figure(self):
        return self.fig, self.ax

    def add_frame(self, fig, ax):
        if not plt.fignum_exists(fig.number):
            return False
        plt.pause(0.002)
        return True

    def end(self):
        pass
