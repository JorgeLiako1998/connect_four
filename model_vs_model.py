#!/usr/bin/env python3
"""Play two models against each other and report wins / ties.

    python model_vs_model.py                                   # model.joblib vs model_20k.joblib
    python model_vs_model.py --a models/model.joblib --b models/model.joblib   # same model both sides
    python model_vs_model.py --games 200 --depth 8 --opening 6

Each game starts with --opening random plies (so games differ), and every
opening is played twice with the sides swapped, so neither model gets an
unfair first-move advantage. Use --depth 0 to compare the raw models with no
tactical search.
"""
import argparse
import csv
import random
import time

import joblib

from core import best_move
from simulate import Board, ROWS, COLS


def play(opening, strat1, strat2):
    """Returns (moves, winner) where winner is 1, 2 or 0 (draw)."""
    b, player, moves = Board(), 1, []
    strategies = {1: strat1, 2: strat2}
    for col in opening:
        row = b.drop(col, player)
        moves.append(col)
        if b.is_win(row, col, player):
            return moves, player
        player = 3 - player
    while True:
        if len(moves) == ROWS * COLS:
            return moves, 0
        col = strategies[player](b, player)
        row = b.drop(col, player)
        moves.append(col)
        if b.is_win(row, col, player):
            return moves, player
        player = 3 - player


def random_opening(n):
    b, player, cols = Board(), 1, []
    for _ in range(n):
        col = random.choice(b.valid_moves())
        row = b.drop(col, player)
        cols.append(col)
        if b.is_win(row, col, player):
            break
        player = 3 - player
    return cols


def make_strategy(model, depth):
    if depth == 0:
        # raw model only: no search, no forced-win/block overrides beyond the model itself
        from core import score_moves
        return lambda b, p: max((s := score_moves(model, b, p, depth=1)), key=lambda m: s[m]["score"])
    return lambda b, p: best_move(model, b, p, depth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="models/model.joblib", help="model A")
    ap.add_argument("--b", default="models/model_20k.joblib", help="model B")
    ap.add_argument("--games", type=int, default=100, help="total games (rounded up to an even number)")
    ap.add_argument("--depth", type=int, default=6, help="tactical search depth (0 = off)")
    ap.add_argument("--opening", type=int, default=4, help="random opening plies")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=None, help="optional file to save every game")
    args = ap.parse_args()
    random.seed(args.seed)

    A = make_strategy(joblib.load(args.a), args.depth)
    B = make_strategy(joblib.load(args.b), args.depth)
    pairs = (args.games + 1) // 2
    a_wins = b_wins = draws = 0
    as_p1 = {"a": 0, "b": 0}  # wins by whoever moved first
    p1_wins = p2_wins = 0
    rows = []
    t = time.time()
    for i in range(pairs):
        opening = random_opening(args.opening)
        for a_is_p1 in (True, False):
            s1, s2 = (A, B) if a_is_p1 else (B, A)
            moves, w = play(opening, s1, s2)
            if w == 0:
                draws += 1
            elif (w == 1) == a_is_p1:
                a_wins += 1
            else:
                b_wins += 1
            p1_wins += w == 1
            p2_wins += w == 2
            rows.append([len(rows), "A" if a_is_p1 else "B", "B" if a_is_p1 else "A",
                         " ".join(map(str, moves)), w, len(moves)])
        if (i + 1) % 10 == 0:
            print(f"  {2 * (i + 1)} games played ({time.time() - t:.0f}s)", flush=True)

    n = 2 * pairs
    print(f"\n{n} games, search depth {args.depth}, {args.opening} random opening plies")
    print(f"A = {args.a}")
    print(f"B = {args.b}\n")
    print(f"A wins: {a_wins:4d}  ({a_wins / n:.1%})")
    print(f"B wins: {b_wins:4d}  ({b_wins / n:.1%})")
    print(f"Ties:   {draws:4d}  ({draws / n:.1%})")
    print(f"\n(by seat: first player won {p1_wins}, second player won {p2_wins})")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["game_id", "p1_model", "p2_model", "moves", "winner", "num_moves"])
            w.writerows(rows)
        print(f"saved games to {args.csv}")


if __name__ == "__main__":
    main()
