"""Hopfield energy on a linear chain with orthogonal layers: definiteness versus signal.

For E(z) = sum_k 0.5|z_k|^2 - sum_k <z_k, g Q_k z_{k-1}> with Q_k orthogonal, the change
of basis z_k -> Q_k ... Q_1 z_k turns the activity Hessian into I - g (S + S^T) (x) I,
S the shift on n layers, so its eigenvalues are exactly 1 - 2 g cos(k pi / (n + 1)).

  * g > 1 / (2 cos(pi / (n + 1))), which includes the isometric g = 1: indefinite.
  * below that: positive definite, and the equilibrium response to a drive at the first
    layer decays geometrically along depth.
  * at g = 1/2 the Hessian is half the path Laplacian, kappa = cot^2(pi / (2n + 2)) =
    Theta(n^2), and the response decays only as 1/n.

So a response that survives depth without the Hessian going indefinite needs the critical
gain, where the conditioning is quadratic again. This checks all three numerically on
random orthogonal layers.
"""
from __future__ import annotations
import numpy as np


def chain_hessian(n, w, g, rng):
    H = np.eye(n * w)
    for k in range(1, n):
        q, r = np.linalg.qr(rng.normal(size=(w, w)))
        Q = g * q * np.sign(np.diag(r))
        H[k * w:(k + 1) * w, (k - 1) * w:k * w] -= Q
        H[(k - 1) * w:k * w, k * w:(k + 1) * w] -= Q.T
    return H


def main():
    rng = np.random.default_rng(0)
    w = 8
    print("n,g,lambda_min,lambda_max,pred_min,pred_max,kappa,cot2,response_last_over_first")
    for n in (15, 31, 63):
        for g in (1.0, 0.6, 0.5, 0.45, 0.3):
            H = chain_hessian(n, w, g, rng)
            ev = np.linalg.eigvalsh(H)
            pred = 1 - 2 * g * np.cos(np.arange(1, n + 1) * np.pi / (n + 1))
            kappa, resp = float("nan"), float("nan")
            if ev.min() > 0:
                kappa = ev.max() / ev.min()
                b = np.zeros(n * w); b[:w] = rng.normal(size=w)
                z = np.linalg.solve(H, b)
                resp = np.linalg.norm(z[-w:]) / np.linalg.norm(z[:w])
            cot2 = 1 / np.tan(np.pi / (2 * (n + 1))) ** 2
            print(f"{n},{g},{ev.min():.6f},{ev.max():.6f},{pred.min():.6f},{pred.max():.6f},"
                  f"{kappa:.2f},{cot2:.2f},{resp:.3e}")


if __name__ == "__main__":
    main()
