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

### Structure du projet

Tous les scripts `python` sont dans le `src`.
Le scripts `main.py` commande toute la simulation et contient la logique principale. Les autres scripts contiennent les différents modules utiles à la résolution des éléments finis, ou de l'affichage des solutions



