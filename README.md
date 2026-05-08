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

### Exécution

```bash
make main
```

`src/main.py` exécute `test_velocity(D=0.5, nsteps=100)`. Il compare l'évolution de la population totale pour plusieurs vitesses de déplacement du climat (`c = 5, 10, 15` km/an) et enregistre la figure dans `figures/velocity_D=0.5.pdf`.

La première exécution génère et sauvegarde la condition initiale dans `data/initial_D=0.5.npy` (simulation stationnaire préalable). Les exécutions suivantes rechargent ce fichier directement.

Un script de tests de convergence est également disponible :

```bash
make convergence
```

Il raffine progressivement la taille de maille `h` (de 80 à 15 km) à pas de temps fixe et trace les trajectoires de population totale `P(t)`.

## Données nécessaires

Le projet s'appuie sur :

- `src/ne_10m_admin_0_countries.zip` — frontières des pays (Natural Earth) pour construire le maillage,
- `src/world.tif` — modèle numérique de terrain au format GeoTIFF pour l'altitude (non inclus),
- des dossiers de sortie `data/` et `figures/` (à créer avant la première exécution).
  Pour obtenir le raster d'altitude, télécharger depuis OpenTopography :  
  https://portal.opentopography.org/raster?opentopoID=OTSDEM.032021.4326.1

Pour changer de pays, modifier le paramètre `country` dans `main.py` ou `convergence.py` (valeur par défaut : `"Italy"`).

## Paramètres principaux

| Paramètre    | Signification                               | Valeur par défaut |
| ------------ | ------------------------------------------- | ----------------- |
| `D`          | Coefficient de diffusion spatiale [km²/an]  | 0.5               |
| `c`          | Vitesse de déplacement du climat [km/an]    | 5 / 10 / 15       |
| `r`          | Taux de croissance maximal [1/an]           | 1.0               |
| `r_tilde`    | Taux de mortalité hors habitat [1/an]       | 0.1               |
| `K`          | Capacité de charge (portance)               | 3                 |
| `theta`      | Paramètre du schéma θ (1 = Euler implicite) | 1                 |
| `elev_opt`   | Altitude optimale de l'espèce [m]           | 20                |
| `elev_width` | Largeur de la niche altitudinale [m]        | 30                |

## Structure du projet

```
.
├── Makefile
├── README.md
├── requirements.txt
├── src/
│   ├── altitude.py       # Lecture du MNT (GeoTIFF), calcul altitude / pente par nœud
│   ├── convergence.py    # Tests de convergence spatiale (raffinement de h)
│   ├── dirichlet.py      # Schéma θ et réduction du système avec conditions de Dirichlet
│   ├── errors.py         # Calcul des erreurs L2 et H1 par rapport à une solution exacte
│   ├── gmsh_utils.py     # Initialisation gmsh, quadrature, fonctions de base
│   ├── main.py           # Point d'entrée : test_velocity et sauvegarde CI
│   ├── mass.py           # Assemblage de la matrice de masse globale
│   ├── mesh.py           # Construction du maillage 2D à partir des frontières d'un pays
│   ├── plot_utils.py     # Visualisation 2D, affichage interactif et génération de vidéos
│   ├── simulation.py     # Boucle temporelle, dynamique KPP-Fisher, modèle d'altitude
│   ├── stiffness.py      # Assemblage matrice de rigidité et second membre
│   └── test.py           # Exploration / prototype FEniCS (non utilisé en production)
```
