"""Tune PyTorch multinomial logistic regression by five-fold CV within training."""
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn.functional as F

from workflow import arguments, atomic, data, host, output, read_json, record_config, save_json

SEED = 0
C_GRID = [0.001, 0.01, 0.1, 1, 10]
K = 5


def fit_logreg(X, y, C, max_iter):
    torch.manual_seed(SEED)
    model = torch.nn.Linear(784, 10).to(X.device)
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(model(X), y) + model.weight.square().sum() / (2 * C * len(y))
        loss.backward()
        return loss

    optimizer.step(closure)
    return model, optimizer.state[next(model.parameters())]["n_iter"]


@torch.no_grad()
def evaluate(model, X, y):
    scores = model(X)
    return (scores.argmax(1) == y).float().mean().item(), F.cross_entropy(scores, y).item()


def main():
    args = arguments(__doc__).parse_args()
    out = output(args.smoke)
    device = "cpu" if args.smoke else "cuda"
    max_iter = 25 if args.smoke else 500
    if args.smoke:
        torch.set_num_threads(2)
    config = {
        **host(), "device": device, "torch": str(torch.__version__), "sklearn": sklearn.__version__,
        "max_iter": max_iter, "K": K, "grid": C_GRID, "seed": SEED,
    }
    record_config(out / "logreg_config.json", config)
    checkpoint, result = out / "logreg.pt", out / "logreg.json"
    if checkpoint.exists() and result.exists():
        print(f"Logistic regression already saved in {out}")
        return
    X_train, y_train, X_val, y_val = data(args.smoke)
    Xt, Xv = (torch.as_tensor(a, device=device) for a in (X_train, X_val))
    yt, yv = (torch.as_tensor(a, device=device).long() for a in (y_train, y_val))
    folds = StratifiedKFold(K, shuffle=True, random_state=SEED)
    fold_indices = list(folds.split(X_train, y_train))
    cv = {}
    for C in C_GRID:
        records = []
        for fold, (tr, held_out) in enumerate(fold_indices, 1):
            path = out / "logreg_folds" / f"C{C}_fold{fold}.json"
            if path.exists():
                record = read_json(path)
            else:
                tr, held_out = (torch.as_tensor(i, device=device) for i in (tr, held_out))
                model, n_iter = fit_logreg(Xt[tr], yt[tr], C, max_iter)
                accuracy, _ = evaluate(model, Xt[held_out], yt[held_out])
                record = {"fold": fold, "accuracy": accuracy, "iterations": n_iter}
                save_json(path, record)
            records.append(record)
        accuracies = [record["accuracy"] for record in records]
        cv[str(C)] = {
            "folds": records, "mean_accuracy": float(np.mean(accuracies)),
            "sd_accuracy": float(np.std(accuracies, ddof=1)),
        }
        print(f"C={C}: {cv[str(C)]['mean_accuracy']:.4f} mean CV accuracy", flush=True)

    best_C = max(C_GRID, key=lambda C: cv[str(C)]["mean_accuracy"])
    model, n_iter = fit_logreg(Xt, yt, best_C, max_iter)
    accuracy, logloss = evaluate(model, Xv, yv)
    atomic(checkpoint, lambda path: torch.save(model.cpu().state_dict(), path))
    save_json(result, {
        "K": K, "grid": C_GRID, "cv": cv, "best_C": best_C, "refit_iterations": n_iter,
        "val_accuracy": accuracy, "val_se": float(np.sqrt(accuracy * (1 - accuracy) / len(yv))),
        "val_logloss": logloss, "n_val": len(yv), "n_train": len(yt),
    })
    print(f"Chosen C={best_C}; validation accuracy={accuracy:.4f}, log loss={logloss:.4f}")


if __name__ == "__main__":
    main()
