"""Kiem tra doc lap cac phep tinh trong bai BT04 bang NumPy float64."""

import json
from pathlib import Path

import numpy as np


X = np.array([0.1, 0.2, 1.0], dtype=np.float64)
Y = np.array([0.0, 1.0], dtype=np.float64)
LEARNING_RATE = 0.1
EPSILON = 1e-6
TOLERANCE = 1e-9


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def finite_difference(loss_fn, weights):
    result = np.zeros_like(weights)
    for index in np.ndindex(weights.shape):
        plus, minus = weights.copy(), weights.copy()
        plus[index] += EPSILON
        minus[index] -= EPSILON
        result[index] = (loss_fn(plus) - loss_fn(minus)) / (2.0 * EPSILON)
    return result


def one_neuron(weights):
    z = float(weights @ X)
    prediction = float(sigmoid(z))
    loss = float(np.logaddexp(0.0, -z))  # BCE voi nhan y = 1.
    gradient = (prediction - 1.0) * X
    return {
        "z": z, "p": prediction, "loss": loss,
        "gradient": gradient.tolist(),
    }


def two_neurons(weights, activation):
    z = weights @ X
    if activation == "tanh":
        first = np.tanh(z[0])
        first_derivative = 1.0 - first ** 2
    elif activation == "relu":
        first = max(0.0, z[0])
        first_derivative = float(z[0] > 0.0)
    else:
        raise ValueError("activation must be tanh or relu")
    a = np.array([first, sigmoid(z[1])])
    # Cung gia dinh voi bai Word: softmax SAU cac ham kich hoat.
    shifted = a - np.max(a)
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    loss = float(-np.log(probabilities[1]))
    derivative = np.array([first_derivative, a[1] * (1.0 - a[1])])
    delta = (probabilities - Y) * derivative
    gradient = np.outer(delta, X)
    return {
        "z": z.tolist(), "activation": a.tolist(),
        "probability": probabilities.tolist(), "loss": loss,
        "activation_derivative": derivative.tolist(),
        "delta": delta.tolist(), "gradient": gradient.tolist(),
    }


def main():
    weights = np.full(3, 0.25, dtype=np.float64)
    before = one_neuron(weights)
    gradient = np.array(before["gradient"])
    updated = weights - LEARNING_RATE * gradient
    after = one_neuron(updated)
    numerical = finite_difference(lambda w: one_neuron(w)["loss"], weights)
    error = float(np.max(np.abs(numerical - gradient)))
    require(error < TOLERANCE, "Exercise 1 gradient check failed")
    require(after["loss"] < before["loss"], "Exercise 1 loss did not decrease")
    result = {
        "exercise1": {
            "before": before, "weights_after": updated.tolist(), "after": after,
            "finite_difference_max_error": error,
        }
    }

    weights = np.full((2, 3), 0.5, dtype=np.float64)
    vector = weights @ X
    loop = np.array([
        sum(weights[j, i] * X[i] for i in range(3)) for j in range(2)
    ])
    require(np.allclose(vector, loop, rtol=0, atol=1e-15),
            "Exercise 2 vectorization check failed")
    result["exercise2"] = {
        "z": vector.tolist(), "output": sigmoid(vector).tolist(),
        "vectorization_max_error": float(np.max(np.abs(vector - loop))),
    }

    result["exercise3"] = {}
    for activation in ["tanh", "relu"]:
        before = two_neurons(weights, activation)
        gradient = np.array(before["gradient"])
        numerical = finite_difference(
            lambda w: two_neurons(w, activation)["loss"], weights
        )
        error = float(np.max(np.abs(numerical - gradient)))
        require(error < TOLERANCE, f"Exercise 3 {activation} gradient check failed")
        updated = weights - LEARNING_RATE * gradient
        after = two_neurons(updated, activation)
        require(after["loss"] < before["loss"], "Exercise 3 loss did not decrease")
        probabilities = np.array(after["probability"])
        require(np.all(probabilities > 0.0), "Non-positive probability")
        require(abs(probabilities.sum() - 1.0) < 1e-12, "Invalid probability sum")
        result["exercise3"][activation] = {
            "before": before, "weights_after": updated.tolist(), "after": after,
            "finite_difference_max_error": error,
            "finite_difference_gradient": numerical.tolist(),
        }

    output = Path(__file__).resolve().with_name("ket_qua_tinh_toan.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print("PASS: forward pass, vectorization, gradients and one-step updates.")
    print(f"Saved: {output.name}")
    print(f"Gradient tolerance: {TOLERANCE:.0e}; finite-difference step: {EPSILON:.0e}")


if __name__ == "__main__":
    main()
