"""
- `parameters` représente une liste qui contient les paramètres de notre modèle : les coefficients du polynôme.
- `xs` contient les abscisses de différents points de données.
- `ys` représente les ordonnées attendues des différents points de données.
"""


def model(parameters, x):
    """
    Calcule les sorties du modèle étant donné les paramètres et les abscisses des points de données.
    """
    prediction = 0.0
    for k in range(len(parameters)):
        prediction = prediction + parameters[k] * x**k
    return prediction


def loss(parameters, xs, ys):
    """
    Calcule l'erreur moyenne étant donné les paramètres, ainsi que les abscisses et ordonnées des points de données.
    """
    squared_errors_sum = 0.0
    for x, y in zip(xs, ys):
        prediction = model(parameters, x)
        squared_errors_sum = squared_errors_sum + (prediction - y) ** 2
    return squared_errors_sum / len(xs)


def grad(parameters, xs, ys):
    """
    Calcule le gradient de l'erreur moyenne par rapport aux paramètres. La sortie est une liste de la même taille que `parameters`.
    """
    gradient = []
    for k in range(len(parameters)):
        derivative_sum = 0.0
        for x, y in zip(xs, ys):
            prediction = model(parameters, x)
            derivative_sum = derivative_sum + 2 * (prediction - y) * x**k
        gradient.append(derivative_sum / len(xs))
    return gradient
