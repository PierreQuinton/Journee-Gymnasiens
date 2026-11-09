"""Lab runner: data, automatic checks and the animated training.

This file imports your three functions from `lab.py` and runs them. In
VS Code, press F5 (or select the "Lab" configuration in the dropdown
next to the Run button). There is nothing to change in this file.

The training animation starts once the three checks pass.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

from lab import grad, loss, model


# ---------------------------------------------------------------------------
# Checks (run the file to use them)
# ---------------------------------------------------------------------------


def fail(title, details=""):
    """Print a short error message and stop before the animation."""
    print("\n" + "-" * 62)
    print(title)
    if details:
        print(details)
    print("-" * 62)
    print("Corrigez, puis relancez le programme (F5).")
    sys.exit(1)


def checked(function_name, function, named_args, expected, hint):
    """Run one check case and compare with the expected result.

    Prints NOTHING when everything matches. On failure, shows the tested
    computation with the expected and obtained values, and stops before
    the animation. `named_args` is a list of (name, value) pairs, used
    both for the call and for the failure display.
    """
    args = [value for name, value in named_args]
    case = (
        f"{function_name}("
        + ", ".join(f"{name} = {value}" for name, value in named_args)
        + ")"
    )
    try:
        actual = function(*args)
    except Exception as error:
        fail(
            f"La fonction {function_name} n'est pas encore correcte :\n"
            f"elle a planté avec l'erreur {type(error).__name__}: {error}.",
            f"{case}\n"
            "Vérifiez les noms de variables et les tailles des tableaux.",
        )
    if actual is Ellipsis:
        step = {"model": "Étape 1", "loss": "Étape 2", "grad": "Étape 3"}[
            function_name
        ]
        fail(
            f"La fonction {function_name} n'est pas encore implémentée :\n"
            "elle renvoie encore le symbole `...`.",
            f"Commencez par l'implémenter ({step} du polycopié), puis\n"
            "relancez le programme.",
        )
    if np.shape(actual) != np.shape(expected):
        fail(
            f"La fonction {function_name} n'est pas encore correcte :\n"
            "mauvaise taille du résultat.",
            f"{case}\n"
            f"Taille attendue {np.shape(expected)}, taille obtenue\n"
            f"{np.shape(actual)}.",
        )
    try:
        matches = np.allclose(actual, expected)
    except Exception as error:
        fail(
            f"La fonction {function_name} n'est pas encore correcte :\n"
            f"son résultat n'est pas un simple nombre ou tableau NumPy\n"
            f"({type(error).__name__}: {error}).",
            f"{case}\n"
            "Cause classique : une branche du code sans `return` correct",
        )
    if not matches:
        fail(
            f"La fonction {function_name} n'est pas encore correcte.",
            f"{case}\n"
            f"attendu  = {expected}\n"
            f"obtenu   = {actual}\n\n"
            f"{hint}",
        )


def reference_model(parameters, xs):
    """Plain polynomial evaluation, used by the checks as reference."""
    return sum(p * xs**k for k, p in enumerate(parameters))


def check_model():
    """Check `model` on several polynomial evaluations."""
    rng = np.random.default_rng(0)
    cases = [
        # le cas simple, à refaire à la main
        (
            (np.array([4.0, 1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0, -1.5])),
            "Aide : la prédiction en x doit être\n"
            "    p0 + p1 * x + p2 * x**2 + p3 * x**3\n"
            "avec p_k = parameters[k].",
        ),
        # quelques autres valeurs, degré 3
        (
            (rng.normal(size=4), rng.normal(size=6)),
            "Aide : les coefficients p_k viennent de `parameters` dans\n"
            "l'ordre, et chaque p_k multiplie xs**k.",
        ),
        # degré 7 et entrées à plusieurs dimensions (utile pour le bonus)
        (
            (rng.normal(size=8), rng.normal(size=(5, 3))),
            "Aide : une boucle sur les coefficients garde la formule\n"
            "valable pour n'importe quel nombre de paramètres et n'importe\n"
            "quelle taille d'entrée :\n"
            "    sum(p * xs**k for k, p in enumerate(parameters))",
        ),
    ]
    for (parameters, xs), hint in cases:
        checked(
            "model",
            model,
            [("parameters", parameters), ("xs", xs)],
            reference_model(parameters, xs),
            hint,
        )
    print("Le modèle a l'air bon.")


def check_loss():
    """Check `loss` on a few small data sets."""
    rng = np.random.default_rng(0)
    cases = [
        # le cas simple, à refaire à la main (le modèle prédit x - x**2)
        (
            (
                np.array([0.0, 1.0, -1.0, 0.0]),
                np.array([-1.0, 0.0, 1.0, 2.0]),
                np.array([2.0, 0.0, -1.0, 3.0]),
            ),
            "Aide : le coût est la moyenne des carrés des différences\n"
            "entre les prédictions du modèle et les cibles :\n"
            "    np.mean((model(parameters, xs) - ys) ** 2)",
        ),
        # quelques autres valeurs, avec des jeux de données de tailles variées
        (
            (rng.normal(size=4), rng.normal(size=5), rng.normal(size=5)),
            "Aide : attention à bien élever au carré et à bien faire la\n"
            "moyenne (np.mean) sur les points de données.",
        ),
        (
            (rng.normal(size=7), rng.normal(size=9), rng.normal(size=9)),
            "Aide : le coût doit rester UN seul nombre, quelle que soit\n"
            "la taille du jeu de données.",
        ),
    ]
    for (parameters, xs, ys), hint in cases:
        differences = reference_model(parameters, xs) - ys
        checked(
            "loss",
            loss,
            [("parameters", parameters), ("xs", xs), ("ys", ys)],
            np.mean(differences**2),
            hint,
        )
    print("Le coût a l'air bon.")


def check_grad():
    """Check `grad` against the exact derivative of the loss."""
    rng = np.random.default_rng(0)
    cases = [
        # le cas simple, à refaire à la main
        (
            (
                np.array([0.5, -1.0, 2.0, 0.3]),
                np.array([-2.0, -1.0, 1.0, 2.0]),
                np.array([2.0, 0.0, -1.0, 3.0]),
            ),
            "Aide : dL/dp_k est la moyenne sur les points de données de\n"
            "    2 * (prédiction - cible) * x**k\n"
            "une entrée par coefficient, dans le même ordre que `parameters`.",
        ),
        # quelques autres valeurs, avec des nombres de coefficients variés
        (
            (rng.normal(size=5), rng.normal(size=4), rng.normal(size=4)),
            "Aide : chaque entrée du gradient ressemble au coût, avec un\n"
            "facteur x**k en plus (et le facteur de 2 qui disparaît si l'on\n"
            "dérive par rapport à p_k, et non par rapport à x).",
        ),
        (
            (rng.normal(size=8), rng.normal(size=6), rng.normal(size=6)),
            "Aide : dL/dp_k est la moyenne sur les points de données de\n"
            "    2 * (prédiction - cible) * x**k\n"
            "une entrée par coefficient, dans le même ordre que `parameters`.",
        ),
    ]
    for (parameters, xs, ys), hint in cases:
        differences = reference_model(parameters, xs) - ys
        expected = 2 / len(xs) * np.array(
            [np.sum(differences * xs**k) for k in range(len(parameters))]
        )
        checked(
            "grad",
            grad,
            [("parameters", parameters), ("xs", xs), ("ys", ys)],
            expected,
            hint,
        )
    print("Le gradient a l'air bon.")


# ---------------------------------------------------------------------------
# Data and visualisation
# ---------------------------------------------------------------------------


def animate_descent():
    """Precompute a whole gradient descent and display it as an animation."""
    # Data points (x, y).
    data = np.array(
        [
            [-2.0, -2.0],
            [-1.0, 4.0],
            [1.0, -6.0],
            [2.0, 1.0],
        ]
    )
    xs, ys = data[:, 0], data[:, 1]

    # Gradient descent settings.
    parameters = np.zeros(4)  # initial parameters: degree-3 polynomial
    learning_rate = 0.02
    n_iter = 250

    # Precompute the whole trajectory, so that the loss plot can show every
    # iteration from the very first one.
    parameters_history = []
    losses = []
    for _ in range(n_iter):
        parameters_history.append(parameters.copy())
        losses.append(loss(parameters, xs, ys))
        # `asarray` accepte aussi bien le tableau NumPy attendu qu'une
        # simple liste Python (style for-loops) renvoyée par l'élève.
        parameters = parameters - learning_rate * np.asarray(
            grad(parameters, xs, ys)
        )
    losses = np.array(losses)

    # Camera: fix the axes from the precomputed trajectory so nothing moves
    # during the animation except the artists we update.
    grid_xs = np.linspace(xs.min() - 0.8, xs.max() + 0.8, 200)
    grid_ys = np.concatenate([model(p, grid_xs) for p in parameters_history])

    # Left plot: data, current model curve, estimated points and their
    # 25%-opaque error segments towards the targets.
    fig, (ax_data, ax_loss) = plt.subplots(
        1, 2, figsize=(10, 4.5), layout="constrained"
    )

    (model_curve,) = ax_data.plot(grid_xs, model(parameters_history[0], grid_xs))
    ax_data.scatter(xs, ys, color="black", label="data")
    (estimated_points,) = ax_data.plot(
        xs, model(parameters_history[0], xs), "o", color="tab:red"
    )
    errors = LineCollection([], colors="tab:red", alpha=0.25, label="errors")
    ax_data.add_collection(errors)
    ax_data.set_xlim(grid_xs[0], grid_xs[-1])
    ax_data.set_ylim(grid_ys.min() - 0.3, grid_ys.max() + 0.3)
    ax_data.set_xlabel("x")
    ax_data.set_ylabel("y")
    ax_data.set_title(f"iteration 0, loss = {losses[0]:.2e}")
    ax_data.legend(loc="upper right", fontsize=8)

    # Right plot: loss against iteration; the shape is known in advance.
    ax_loss.plot(np.arange(n_iter), losses, color="tab:blue")
    (loss_marker,) = ax_loss.plot([0], [losses[0]], "o", color="tab:blue")
    ax_loss.set_xlim(0, n_iter - 1)
    ax_loss.set_ylim(0, losses.max() * 1.05)
    ax_loss.set_xlabel("iteration")
    ax_loss.set_ylabel("mean squared error")

    def update(frame):
        parameters = parameters_history[frame]
        estimated_ys = model(parameters, xs)

        model_curve.set_ydata(model(parameters, grid_xs))
        estimated_points.set_ydata(estimated_ys)
        errors.set_segments(
            np.stack(
                [
                    np.column_stack([xs, ys]),
                    np.column_stack([xs, estimated_ys]),
                ],
                axis=1,
            )
        )
        ax_data.set_title(f"iteration {frame}, loss = {losses[frame]:.2e}")
        loss_marker.set_data([frame], [losses[frame]])

    ani = FuncAnimation(fig, update, frames=n_iter, interval=40)
    plt.show()


if __name__ == "__main__":
    check_model()
    check_loss()
    check_grad()
    animate_descent()
