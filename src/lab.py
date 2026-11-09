"""
- `parameters` représente une liste qui contient les paramètres de notre modèle : les coefficients du polynôme.
- `xs` contient les abscisses de différents points de données.
- `ys` représente les ordonnées attendues des différents points de données.
"""

def model(parameters, x):
    """
    Calcule les sorties du modèle étant donné les paramètres et les abscisses des points de données.
    """
    return ...


def loss(parameters, xs, ys):
    """
    Calcule l'erreur moyenne étant donné les paramètres, ainsi que les abscisses et ordonnées des points de données.
    """
    return ...


def grad(parameters, xs, ys):
    """
    Calcule le gradient de l'erreur moyenne par rapport aux paramètres. La sortie est une liste de la même taille que `parameters`.
    """
    return ...
