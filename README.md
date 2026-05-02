Code pour le cours LEPL1110 - Élements finis

---

Simulation numérique par éléments finis de la diffusion d'une espèce dans un climat changeant.

### Usage

#### Installation

Assurez vous d'avoir installé les dépendances avec :

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt

```

#### Utilisation

Pour lancer le script principal, utilisez la commande `make main`.

Par défaut, le résultat de la simulation est stocké dans un fichier `simulation.mp4` à la racine du projet.

Pour changer de pays, il faudra au préalable télécharger les données d'altitude sur https://portal.opentopography.org/raster?opentopoID=OTSDEM.032021.4326.1 pour la bonne zone géographique.

### Paramètres disponibles

| Argument    | Défaut   | Description                                                                      |
| ----------- | -------- | -------------------------------------------------------------------------------- |
| `--country` | `Italy`  | Pays simulé (nom anglais, ex: `France`, `Spain`)                                 |
| `-hc`       | `20`     | Taille des mailles du mesh (km)                                                  |
| `-order`    | `1`      | Ordre des éléments finis (1 ou 2)                                                |
| `--dt`      | `0.5`    | Pas de temps (années)                                                            |
| `--nsteps`  | `80`     | Nombre de pas de temps                                                           |
| `--theta`   | `1.0`    | Schéma temporel (0 = Euler explicite, 1 = Euler implicite, 0.5 = Crank-Nicolson) |
| `--band-y0` | `-200.0` | Position initiale (km) de la bande climatique favorable                          |

## Structure du projet

```
.
├── src/
│   ├── main.py           # Script principal — logique de simulation et boucle temporelle
│   ├── mesh.py           # Construction du mesh 2D à partir des frontières du pays (GeoJSON)
│   ├── altitude.py       # Lecture du MNT (.tif), calcul de la pente et de l'élévation aux DOFs
│   ├── stiffness.py      # Assemblage de la matrice de rigidité et du vecteur RHS
│   ├── mass.py           # Assemblage de la matrice de masse
│   ├── dirichlet.py      # Schéma θ et application des conditions de Dirichlet
│   ├── gmsh_utils.py     # Utilitaires gmsh (init, quadrature, jacobiens)
│   ├── plot_utils.py     # Affichage des solutions EF 2D et génération de la vidéo
│   └── italy.tif     # Modèle numérique de terrain (MNT) — fichier GeoTIFF A CHANGER SI AUTRE PAYS
├── requirements.txt
├── Makefile
└── README.md
```
