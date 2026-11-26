"""Logistic regression baseline (PyTorch, GPU), tuned by 5-fold cross-validation.

1. For each C, run 5-fold CV INSIDE the 50,000 training images (the validation set is not used).
2. Keep the C with the best mean CV accuracy and refit it on all 50,000 training images.
3. Report accuracy and log loss on the 10,000 validation images, with a standard error.

Model: scores = X @ W + b (10 scores per image) -> softmax -> probabilities.
Loss : mean cross-entropy + ||W||^2 / (2 C n)  (same objective as scikit-learn's C).
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold  # only produces row indices, not a model

from common import SEED, load_train_val

DEVICE = "cuda"
C_GRID = [0.001, 0.01, 0.1, 1, 10]  # small C = strong penalty (simpler model)
K = 5
MAX_ITER = 500
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

X_train, y_train, X_val, y_val = load_train_val()
Xt, Xv = (torch.as_tensor(a, device=DEVICE) for a in (X_train, X_val))
yt, yv = (torch.as_tensor(a, device=DEVICE).long() for a in (y_train, y_val))
print(f"train {tuple(Xt.shape)}, validation {tuple(Xv.shape)}, device {DEVICE}")


def fit_logreg(X, y, C):
    """Fit by full-batch L-BFGS (a standard solver for smooth convex problems)."""
    torch.manual_seed(SEED)
    n = X.shape[0]
    model = torch.nn.Linear(784, 10).to(DEVICE)  # 784 weights + 1 intercept per class
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=MAX_ITER, line_search_fn="strong_wolfe")

    def closure():  # LBFGS calls this repeatedly to re-evaluate the loss
        optimizer.zero_grad()
        loss = F.cross_entropy(model(X), y) + model.weight.pow(2).sum() / (2 * C * n)
        loss.backward()
        return loss

    optimizer.step(closure)
    n_iter = optimizer.state_dict()["state"][0]["n_iter"]  # to check we did not hit MAX_ITER
    return model, n_iter


def evaluate(model, X, y):
    with torch.no_grad():
        scores = model(X)
        return (scores.argmax(1) == y).float().mean().item(), F.cross_entropy(scores, y).item()


# ---- 1. 5-fold CV inside the training data ----
folds = list(StratifiedKFold(K, shuffle=True, random_state=SEED).split(X_train, y_train))
cv = {}
for C in C_GRID:
    accs, iters = [], []
    for tr, te in folds:
        tr, te = torch.as_tensor(tr, device=DEVICE), torch.as_tensor(te, device=DEVICE)
        model, n_iter = fit_logreg(Xt[tr], yt[tr], C)
        accs.append(evaluate(model, Xt[te], yt[te])[0])
        iters.append(n_iter)
    cv[C] = (float(np.mean(accs)), float(np.std(accs, ddof=1)))
    print(f"C = {C:<6} CV accuracy = {cv[C][0]:.4f} +/- {cv[C][1]:.4f} (sd over folds), "
          f"max L-BFGS iterations used = {max(iters)}", flush=True)

best_C = max(cv, key=lambda c: cv[c][0])

# ---- 2. refit on all 50,000 training images, 3. score on validation ----
model, n_iter = fit_logreg(Xt, yt, best_C)
acc, logloss = evaluate(model, Xv, yv)
se = float(np.sqrt(acc * (1 - acc) / len(yv)))  # standard error of an accuracy
print(f"\nChosen C = {best_C} (refit used {n_iter} of {MAX_ITER} iterations)")
print(f"Validation accuracy = {acc:.4f} +/- {se:.4f} (standard error, n = {len(yv)}), log loss = {logloss:.4f}")

torch.save(model.state_dict(), OUT / "logreg.pt")  # kept for the one-time test evaluation
json.dump({"K": K, "grid": C_GRID, "cv": {str(c): v for c, v in cv.items()}, "best_C": best_C,
           "val_accuracy": acc, "val_se": se, "val_logloss": logloss, "n_val": len(yv)},
          open(OUT / "logreg.json", "w"), indent=2)
