#!/usr/bin/env python3
"""Play the hybrid perfect player against random / smart / the model and count results.

    python perfect_test.py                       # 10 games per opponent per seat
    python perfect_test.py --games 20 --opponents smart model
"""
import argparse
import random
import time

import joblib

import simulate as sim
from core import MODEL_PATH, make_model_strategy
from perfect import PerfectPlayer
from simulate import Board, ROWS, COLS


def play(perfect, opp, perfect_seat):
    b, player, moves = Board(), 1, []
    while True:
        if player == perfect_seat:
            col, _ = perfect.choose(moves)
        else:
            col = opp(b, player)
        row = b.drop(col, player)
        moves.append(col)
        if b.is_win(row, col, player):
            return moves, player
        if len(moves) == ROWS * COLS:
            return moves, 0
        player = 3 - player


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10, help="games per opponent per seat")
    ap.add_argument("--opponents", nargs="+", default=["random", "smart", "model"],
                    choices=["random", "smart", "model"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    model = joblib.load(MODEL_PATH)
    perfect = PerfectPlayer(model)
    opponents = {"random": sim.random_strategy, "smart": sim.smart_strategy,
                 "model": make_model_strategy(model, 6)}
    for name in args.opponents:
        # the model is deterministic, so it only needs one game per seat
        n = 1 if name == "model" else args.games
        for seat in (1, 2):
            w = d = l = 0
            t = time.time()
            lost_games = []
            for _ in range(n):
                moves, winner = play(perfect, opponents[name], seat)
                if winner == seat: w += 1
                elif winner == 0: d += 1
                else:
                    l += 1; lost_games.append("".join(map(str, moves)))
            print(f"perfect as P{seat} vs {name}: win {w}  draw {d}  loss {l}   ({time.time() - t:.0f}s)", flush=True)
            for g in lost_games:
                print("   lost game:", g, flush=True)


if __name__ == "__main__":
    main()
