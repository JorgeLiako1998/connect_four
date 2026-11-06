#!/usr/bin/env python3
"""Train the position evaluator and save it to models/model.joblib.

    python train.py                       # ALL positions of ALL games (chunked, memory-safe)
    python train.py --max-games 50000     # only the first 50,000 games
    python train.py --per-game 4          # sample 4 positions per game (fast, in-memory)

With --per-game 0 (the default) every position is used. They are stored
compactly (1 byte per cell) and fed to the network in chunks over several
epochs, with random left-right mirroring as augmentation, so 500,000 games
(~11.6M positions) fit in 8 GB of RAM.
"""
import argparse
import copy
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from core import MODEL_PATH, build_dataset, mirror_features


def train_in_memory(X, y, args):
    model = MLPClassifier(hidden_layer_sizes=tuple(args.hidden), early_stopping=True,
                          max_iter=args.iters, random_state=args.seed, verbose=True)
    model.fit(X, y)
    print(f"validation accuracy: {model.best_validation_score_:.3f}", flush=True)
    return model


def train_chunked(df, args):
    n_games = min(len(df), args.max_games)
    n_val = min(args.val_games, n_games // 10)
    t = time.time()
    print(f"building training set from {n_games - n_val} games...", flush=True)
    X, y = build_dataset(df.iloc[:n_games - n_val], n_games - n_val, mirror=False, as_float=False)
    Xv, yv = build_dataset(df.iloc[n_games - n_val:n_games], n_val, mirror=False, as_float=True)
    print(f"{len(X):,} training positions, {len(Xv):,} validation positions "
          f"({time.time() - t:.0f}s); class counts {np.bincount(y)}", flush=True)

    rng = np.random.default_rng(args.seed)
    model = MLPClassifier(hidden_layer_sizes=tuple(args.hidden), batch_size=512,
                          random_state=args.seed)
    classes = np.array([0, 1, 2])
    best, best_acc, bad = None, -1.0, 0
    for epoch in range(1, args.iters + 1):
        t = time.time()
        order = rng.permutation(len(X))
        for i in range(0, len(order), args.chunk):
            idx = np.sort(order[i:i + args.chunk])
            xb = X[idx].astype(np.float32)
            flip = rng.random(len(xb)) < 0.5            # random mirroring
            xb[flip] = mirror_features(xb[flip])
            model.partial_fit(xb, y[idx], classes=classes)
        acc = model.score(Xv, yv)
        print(f"epoch {epoch}: loss {model.loss_:.4f}  validation accuracy {acc:.3f}  "
              f"({time.time() - t:.0f}s)", flush=True)
        if acc > best_acc:
            best, best_acc, bad = copy.deepcopy(model), acc, 0
            save(best, args.out)                        # keep the best model on disk as we go
        else:
            bad += 1
            if bad >= args.patience:
                print("no improvement, stopping early", flush=True)
                break
    print(f"best validation accuracy: {best_acc:.3f}", flush=True)
    return best


def save(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/games.csv")
    ap.add_argument("--max-games", type=int, default=500000)
    ap.add_argument("--per-game", type=int, default=0,
                    help="random positions per game; 0 = use ALL positions (chunked training)")
    ap.add_argument("--iters", type=int, default=30, help="max epochs (chunked) or iterations")
    ap.add_argument("--patience", type=int, default=3, help="stop after this many epochs without improvement")
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 64], help="hidden layer sizes")
    ap.add_argument("--chunk", type=int, default=262144, help="positions per training chunk")
    ap.add_argument("--val-games", type=int, default=5000, help="games held out for validation")
    ap.add_argument("--out", default=MODEL_PATH)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    np.random.seed(args.seed)

    df = pd.read_csv(args.data)
    if args.per_game == 0:
        model = train_chunked(df, args)
    else:
        X, y = build_dataset(df, args.max_games, args.per_game, args.seed)
        print(f"{len(X):,} positions from {min(len(df), args.max_games)} games; class counts {np.bincount(y)}", flush=True)
        model = train_in_memory(X, y, args)
    save(model, args.out)
    print(f"saved model to {args.out}", flush=True)


if __name__ == "__main__":
    main()
