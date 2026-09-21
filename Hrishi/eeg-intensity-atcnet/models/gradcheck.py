"""Finite-difference sanity check for NumpyCNNRegressor's manual backprop, before trusting
a real training run on it. Run: python models/gradcheck.py
"""
import numpy as np
from numpy_cnn import NumpyCNNRegressor


def check():
    rng = np.random.default_rng(1)
    model = NumpyCNNRegressor(n_features=10, seed=1)
    X = rng.normal(size=(4, 10))
    y = rng.normal(size=4)

    def loss_fn():
        pred = model.forward(X)
        return np.mean((pred - y) ** 2)

    loss0 = loss_fn()
    pred = model.forward(X)
    diff = pred - y
    dOut = 2 * diff / len(X)
    model.backward(dOut)  # this also applies an Adam step -- so re-fetch grads before comparing values changed
    # Instead, redo analytic grad without applying update: recompute manually for conv1.W[0,0,0]
    print("loss:", loss0)
    eps = 1e-4
    # perturb conv1 weight and re-measure loss, compare to a fresh analytic pass
    model2 = NumpyCNNRegressor(n_features=10, seed=1)
    base = model2.conv1.W[0, 0, 0]
    model2.conv1.W[0, 0, 0] = base + eps
    lp = np.mean((model2.forward(X) - y) ** 2)
    model2.conv1.W[0, 0, 0] = base - eps
    lm = np.mean((model2.forward(X) - y) ** 2)
    numeric_grad = (lp - lm) / (2 * eps)

    model3 = NumpyCNNRegressor(n_features=10, seed=1)
    pred3 = model3.forward(X)
    dOut3 = 2 * (pred3 - y) / len(X)
    # manual analytic grad for conv1.W[0,0,0] via same code path as backward(), without applying the update
    x0, a1, h1, a2, h2, pooled, d1, r1 = None, None, None, None, None, None, None, None
    # simplest: monkeypatch _adam_step to just record grad instead of updating
    recorded = {}
    orig = model3.conv1._adam_step
    def spy(name, grad, state):
        recorded[name] = grad.copy()
        return orig(name, grad, state)
    model3.conv1._adam_step = spy
    model3.backward(dOut3)
    analytic_grad = recorded["W"][0, 0, 0]

    print(f"numeric grad:  {numeric_grad:.6f}")
    print(f"analytic grad: {analytic_grad:.6f}")
    rel_err = abs(numeric_grad - analytic_grad) / (abs(numeric_grad) + abs(analytic_grad) + 1e-8)
    print(f"relative error: {rel_err:.6f}  ({'OK' if rel_err < 1e-2 else 'MISMATCH -- backprop has a bug'})")


if __name__ == "__main__":
    check()
