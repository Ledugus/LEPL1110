import rasterio
from rasterio.transform import rowcol
from rasterio.windows import from_bounds
import numpy as np
from scipy.ndimage import sobel


class ElevationModel:
    def __init__(self, tif_path, proj, center_km=(0.0, 0.0), bounds_km=None):
        """
        bounds_km : (x_min, x_max, y_min, y_max) en km dans le repère centré.
        Si fourni, seule cette fenêtre est lue depuis le .tif → rapide même
        avec un fichier mondial de 1.5 GB.
        """
        self.proj = proj
        self.center_km = np.array(center_km)

        with rasterio.open(tif_path) as src:

            if bounds_km is not None:
                # Convertir les bounds km → lon/lat pour rasterio
                margin = 50  # km de marge autour du domaine
                x_min_km, x_max_km, y_min_km, y_max_km = bounds_km
                corners_km = [
                    (x_min_km - margin, y_min_km - margin),
                    (x_max_km + margin, y_max_km + margin),
                ]
                lons, lats = [], []
                for xk, yk in corners_km:
                    X = (xk + center_km[0]) * 1000.0
                    Y = (yk + center_km[1]) * 1000.0
                    lon, lat = proj(X, Y, inverse=True)
                    lons.append(lon)
                    lats.append(lat)

                window = from_bounds(
                    left=min(lons),
                    bottom=min(lats),
                    right=max(lons),
                    top=max(lats),
                    transform=src.transform,
                )
                self.elevation = src.read(1, window=window).astype(float)
                self.transform = src.window_transform(window)
                print(
                    f"  Fenêtre lue : {self.elevation.shape} pixels "
                    f"(au lieu de {src.height}×{src.width})"
                )
            else:
                # Fallback : tout lire (ancien comportement)
                self.elevation = src.read(1).astype(float)
                self.transform = src.transform

        # Résolution en mètres (identique à avant)
        res_x_deg = abs(self.transform[0])
        res_y_deg = abs(self.transform[4])
        cx = self.transform[2] + res_x_deg * self.elevation.shape[1] / 2
        cy = self.transform[5] - res_y_deg * self.elevation.shape[0] / 2
        x0, y0 = proj(cx, cy)
        x1, y1 = proj(cx + res_x_deg, cy + res_y_deg)
        self.res_x_m = abs(x1 - x0)
        self.res_y_m = abs(y1 - y0)

        dx = sobel(self.elevation, axis=1) / (8.0 * self.res_x_m)
        dy = sobel(self.elevation, axis=0) / (8.0 * self.res_y_m)
        self.slope = np.sqrt(dx**2 + dy**2)

    # get_indices, get_elevation, get_slope : inchangés
    def get_indices(self, x_km, y_km):
        X = (x_km + self.center_km[0]) * 1000.0
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
r0 = 1.0  # taux de croissance maximal [1/s]
z_opt = 1000  # altitude optimale [m]
sigma_z = 400  # largeur caractéristique [m]

# Altitude
z = np.linspace(-500, 3500, 1000)

# Fonction r(z)
r = r0 * np.exp(-((z - z_opt) ** 2) / (2 * sigma_z**2))

if __name__ == "__main__":
    plt.figure(figsize=(8, 5))
    plt.plot(z, r, label="Taux de croissance r(z)")
    plt.axvline(z_opt, color="r", linestyle="--", label="Altitude optimale z_opt")
    plt.title("Taux de croissance en fonction de l'altitude")
    plt.xlabel("Altitude (m)")
    plt.ylabel("Taux de croissance r(z) [1/s]")
    plt.legend()
    plt.grid()
    plt.show()
