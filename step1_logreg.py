"""Step 1.3: multinomial logistic regression baseline, trained on the GPU with PyTorch.

Model: scores = X @ W + b (10 scores per image) -> softmax -> probabilities.
Loss : cross-entropy + L2 penalty on W. Same objective as scikit-learn's
       LogisticRegression(C=C): sum_i CE_i + ||W||^2 / (2C), divided by n.
Try several C, keep the one with the best VALIDATION accuracy, report it with its SE.
"""
import numpy as np
import torch
import torch.nn.functional as F

from common import SEED, load_train_val

DEVICE = "cuda"
torch.manual_seed(SEED)

X_train, y_train, X_val, y_val = (
    torch.as_tensor(a, device=DEVICE) for a in load_train_val()
)
y_train, y_val = y_train.long(), y_val.long()  # class labels must be integers
print(f"train {tuple(X_train.shape)}, validation {tuple(X_val.shape)}, device {DEVICE}")


def fit_logreg(X, y, C, max_iter=200):
    """Fit by full-batch L-BFGS (a standard solver for smooth convex problems)."""
    n = X.shape[0]
    model = torch.nn.Linear(784, 10).to(DEVICE)  # 784 weights + 1 intercept per class
    optimizer = torch.optim.LBFGS(
        model.parameters(), max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure():  # LBFGS calls this repeatedly to re-evaluate the loss
        optimizer.zero_grad()
        loss = F.cross_entropy(model(X), y) + model.weight.pow(2).sum() / (2 * C * n)
        loss.backward()
        return loss

    optimizer.step(closure)  # runs up to max_iter L-BFGS iterations
    return model


def accuracy(model, X, y):
    with torch.no_grad():
        return (model(X).argmax(dim=1) == y).float().mean().item()


C_GRID = [0.001, 0.01, 0.1, 1, 10]  # small C = strong penalty (simple model)

results = {}
for C in C_GRID:
    model = fit_logreg(X_train, y_train, C)
    results[C] = accuracy(model, X_val, y_val)
    print(f"C = {C:<6}  validation accuracy = {results[C]:.4f}", flush=True)

best_C = max(results, key=results.get)
acc = results[best_C]
n = len(y_val)
se = np.sqrt(acc * (1 - acc) / n)  # standard error of an accuracy

print(f"\nChosen C = {best_C}")
print(f"Validation accuracy = {acc:.4f} +/- {se:.4f} (standard error, n = {n})")
