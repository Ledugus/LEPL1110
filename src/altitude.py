import rasterio
from rasterio.transform import rowcol
import numpy as np
from scipy.ndimage import sobel


class ElevationModel:
    def __init__(self, tif_path, proj, center_km=(0.0, 0.0)):
        self.src = rasterio.open(tif_path)
        self.elevation = self.src.read(1).astype(float)
        self.transform = self.src.transform
        self.proj = proj
        self.center_km = np.array(center_km)

        # Résolution spatiale du pixel en mètres
        # transform[0] = pixel width en unités CRS (degrés pour EPSG:4326)
        # On convertit en mètres via la projection au centre du raster
        res_x_deg = abs(self.transform[0])   # degrés/pixel en x
        res_y_deg = abs(self.transform[4])   # degrés/pixel en y

        # Convertir la résolution en mètres (approximation locale au centre)
        cx = self.transform[2] + res_x_deg * self.elevation.shape[1] / 2
        cy = self.transform[5] - res_y_deg * self.elevation.shape[0] / 2
        x0, y0 = proj(cx, cy)
        x1, y1 = proj(cx + res_x_deg, cy + res_y_deg)
        self.res_x_m = abs(x1 - x0)   # mètres par pixel en x
        self.res_y_m = abs(y1 - y0)   # mètres par pixel en y

        # Pente en m/m (gradient spatial correct)
        # Sobel retourne une somme pondérée sur 3 pixels → diviser par 8*res
        dx = sobel(self.elevation, axis=1) / (8.0 * self.res_x_m)
        dy = sobel(self.elevation, axis=0) / (8.0 * self.res_y_m)
        self.slope = np.sqrt(dx**2 + dy**2)   # sans unité (m/m), ex: 0.3 = 30%

    def get_indices(self, x_km, y_km):
        # Annuler le recentrage avant la conversion inverse
        X = (x_km + self.center_km[0]) * 1000.0   # → mètres EPSG:3857
        Y = (y_km + self.center_km[1]) * 1000.0

        lon, lat = self.proj(X, Y, inverse=True)
        row, col = rowcol(self.transform, lon, lat)

        row = int(np.clip(row, 0, self.elevation.shape[0] - 1))
        col = int(np.clip(col, 0, self.elevation.shape[1] - 1))
        return row, col

    def get_elevation(self, x_km, y_km):
        row, col = self.get_indices(x_km, y_km)
        return float(self.elevation[row, col])

    def get_slope(self, x_km, y_km):
        row, col = self.get_indices(x_km, y_km)
        return float(self.slope[row, col])
    

def compute_node_field(nodeCoords, elev_model, field="slope"):
    coords = nodeCoords.reshape(-1, 3)
    values = []

    for x, y, _ in coords:
        if field == "elevation":
            val = elev_model.get_elevation(x, y)
        elif field == "slope":
            val = elev_model.get_slope(x, y)
        else:
            raise ValueError("field must be 'elevation' or 'slope'")
        values.append(val)

    return np.array(values)


import numpy as np
import matplotlib.pyplot as plt

# Paramètres
r0     = 1.0    # taux de croissance maximal [1/s]
z_opt  = 1000   # altitude optimale [m]
sigma_z = 400   # largeur caractéristique [m]

# Altitude
z = np.linspace(-500, 3500, 1000)

# Fonction r(z)
r = r0 * np.exp(-((z - z_opt) ** 2) / (2 * sigma_z ** 2))

# Plot
fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(z, r, color="#3266ad", linewidth=2.5, label=r"$r(z)$")
ax.fill_between(z, r, alpha=0.08, color="#3266ad")
ax.axvline(z_opt, color="#a32d2d", linewidth=1.5, linestyle="--", label=r"$z_\mathrm{opt}$")
ax.axhline(r0, color="gray", linewidth=0.8, linestyle=":", label=r"$r_0$")

ax.set_xlabel("Altitude $z$ (m)", fontsize=13)
ax.set_ylabel(r"$r(z)$", fontsize=13)
ax.set_title(r"Taux de croissance $r(z) = r_0 \exp\!\left(-\frac{(z - z_\mathrm{opt})^2}{2\,\sigma_z^2}\right)$", fontsize=12)
ax.legend(fontsize=12)
ax.set_xlim(z.min(), z.max())
ax.set_ylim(0, r0 * 1.15)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("r_altitude.pdf", bbox_inches="tight")
plt.show()


