"""
Pure-NumPy CNN (dilated-conv stack -> global pooling -> dense head), an ATCNet-inspired
stand-in trained with manual backprop + Adam.

Why this exists: the sandbox this was first run in has pip/PyPI network access blocked
(confirmed 403 "blocked-by-allowlist" from both pypi.org and download.pytorch.org), so
torch and scikit-learn cannot be installed there. This module is a real, actually-trained
model on the real data -- not a mock -- so we can get genuine numbers today without those
packages. It is a simplified stand-in for the full attention+TCN ATCNet in
models/atcnet.py (that PyTorch version is the "real" ATCNet deliverable, meant to be run
wherever torch is installable -- e.g. the user's own machine). This one only has a
2-layer dilated Conv1d stack (no multi-head self-attention, no sliding-window branches),
because implementing MSA backprop by hand reliably in the time available was not worth
the bug risk versus the value of a second, more faithful run.

Architecture: standardize -> Conv1d(1->8, k=3) -> ReLU -> Conv1d(8->8, k=3, dilation=2)
-> ReLU -> global average pool over the feature axis -> Dense(8->16) -> ReLU ->
Dense(16->1). Loss: MSE. Optimizer: Adam, hand-implemented.
"""
from __future__ import annotations

import numpy as np


def he_init(rng, shape):
    fan_in = shape[1] if len(shape) == 2 else np.prod(shape[1:])
    return rng.normal(0, np.sqrt(2.0 / fan_in), size=shape)


class Conv1d:
    """Conv1d with 'same' padding via explicit dilation-aware shifts. Input (B, Cin, L)."""

    def __init__(self, rng, in_ch, out_ch, kernel_size=3, dilation=1):
        self.K = kernel_size
        self.dilation = dilation
        self.pad = ((kernel_size - 1) * dilation) // 2
        self.W = he_init(rng, (out_ch, in_ch, kernel_size)) * 0.5
        self.b = np.zeros(out_ch)
        self.cache = None

    def forward(self, X):
        B, Cin, L = X.shape
        Xp = np.pad(X, ((0, 0), (0, 0), (self.pad, self.pad)))
        out = np.zeros((B, self.W.shape[0], L))
        for k in range(self.K):
            seg = Xp[:, :, k * self.dilation: k * self.dilation + L]  # (B, Cin, L)
            out += np.einsum("oi,bil->bol", self.W[:, :, k], seg)
        out += self.b[None, :, None]
        self.cache = (X, Xp)
        return out

    def backward(self, dOut, lr_state):
        X, Xp = self.cache
        B, Cin, L = X.shape
        dW = np.zeros_like(self.W)
        db = dOut.sum(axis=(0, 2))
        dXp = np.zeros_like(Xp)
        for k in range(self.K):
            seg = Xp[:, :, k * self.dilation: k * self.dilation + L]
            dW[:, :, k] = np.einsum("bol,bil->oi", dOut, seg)
            dXp[:, :, k * self.dilation: k * self.dilation + L] += np.einsum("oi,bol->bil", self.W[:, :, k], dOut)
        dX = dXp[:, :, self.pad:self.pad + L] if self.pad > 0 else dXp
        self._adam_step("W", dW, lr_state)
        self._adam_step("b", db, lr_state)
        return dX

    def _adam_step(self, name, grad, state):
        key = (id(self), name)
        m, v = state.get(key, (np.zeros_like(grad), np.zeros_like(grad)))
        beta1, beta2, eps, lr, t = state["beta1"], state["beta2"], state["eps"], state["lr"], state["t"]
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        state[key] = (m, v)
        mhat = m / (1 - beta1 ** t)
        vhat = v / (1 - beta2 ** t)
        update = lr * mhat / (np.sqrt(vhat) + eps)
        if name == "W":
            self.W -= update
        else:
            self.b -= update


class Dense:
    def __init__(self, rng, in_dim, out_dim):
        self.W = he_init(rng, (out_dim, in_dim)) * 0.5
        self.b = np.zeros(out_dim)
        self.cache = None

    def forward(self, X):  # X: (B, in_dim)
        self.cache = X
        return X @ self.W.T + self.b

    def backward(self, dOut, lr_state):
        X = self.cache
        dW = dOut.T @ X
        db = dOut.sum(axis=0)
        dX = dOut @ self.W
        Conv1d._adam_step(self, "W", dW, lr_state)
        Conv1d._adam_step(self, "b", db, lr_state)
        return dX


def relu(x):
    return np.maximum(0, x)


def relu_grad(x, dOut):
    return dOut * (x > 0)


class NumpyCNNRegressor:
    def __init__(self, n_features, seed=42):
        rng = np.random.default_rng(seed)
        self.conv1 = Conv1d(rng, 1, 8, kernel_size=3, dilation=1)
        self.conv2 = Conv1d(rng, 8, 8, kernel_size=3, dilation=2)
        self.dense1 = Dense(rng, 8, 16)
        self.dense2 = Dense(rng, 16, 1)
        self.lr_state = {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "lr": 1e-3, "t": 0}

    def forward(self, X):  # X: (B, n_features) -> treat as (B, 1, F)
        x0 = X[:, None, :]
        a1 = self.conv1.forward(x0)
        h1 = relu(a1)
        a2 = self.conv2.forward(h1)
        h2 = relu(a2)
        pooled = h2.mean(axis=2)  # global average pool -> (B, 8)
        d1 = self.dense1.forward(pooled)
        r1 = relu(d1)
        out = self.dense2.forward(r1)[:, 0]
        self._cache = (x0, a1, h1, a2, h2, pooled, d1, r1)
        return out

    def backward(self, dOut):
        x0, a1, h1, a2, h2, pooled, d1, r1 = self._cache
        self.lr_state["t"] += 1
        dOut2 = dOut[:, None]
        dr1 = self.dense2.backward(dOut2, self.lr_state)
        dd1 = relu_grad(d1, dr1)
        dpooled = self.dense1.backward(dd1, self.lr_state)
        dh2 = np.repeat(dpooled[:, :, None], h2.shape[2], axis=2) / h2.shape[2]
        da2 = relu_grad(a2, dh2)
        dh1 = self.conv2.backward(da2, self.lr_state)
        da1 = relu_grad(a1, dh1)
        self.conv1.backward(da1, self.lr_state)

    def params(self):
        return [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b,
                self.dense1.W, self.dense1.b, self.dense2.W, self.dense2.b]

    def fit(self, X, y, epochs=40, batch_size=256, verbose=True):
        n = len(X)
        rng = np.random.default_rng(0)
        for epoch in range(epochs):
            order = rng.permutation(n)
            total_loss = 0.0
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                xb, yb = X[idx], y[idx]
                pred = self.forward(xb)
                diff = pred - yb
                loss = np.mean(diff ** 2)
                total_loss += loss * len(idx)
                dOut = 2 * diff / len(idx)
                self.backward(dOut)
            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                print(f"  epoch {epoch:3d}  train_mse={total_loss / n:.4f}")

    def predict(self, X, batch_size=2048):
        preds = []
        for start in range(0, len(X), batch_size):
            preds.append(self.forward(X[start:start + batch_size]))
        return np.concatenate(preds)
