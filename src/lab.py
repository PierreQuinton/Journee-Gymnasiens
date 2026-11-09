

def model(parameters, xs):
    predictions = []
    for x in xs:
        # somme des termes p_k * x**k
        prediction = 0.0
        for k in range(len(parameters)):
            prediction = prediction + parameters[k] * x**k
        predictions.append(prediction)
    return predictions


def loss(parameters, xs, ys):
    squared_errors_sum = 0.0
    for x, y in zip(xs, ys):
        prediction = model(parameters, [x])[0]
        squared_errors_sum = squared_errors_sum + (prediction - y) ** 2
    return squared_errors_sum / len(xs)


def grad(parameters, xs, ys):
    gradient = []
    for k in range(len(parameters)):
        derivative_sum = 0.0
        for x, y in zip(xs, ys):
            prediction = model(parameters, [x])[0]
            derivative_sum = derivative_sum + 2 * (prediction - y) * x**k
        gradient.append(derivative_sum / len(xs))
    return gradient
