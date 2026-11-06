#!/usr/bin/env python3
"""Play the saved model against random / smart, as P1 and P2.

    python test.py                 # 100 games per matchup
    python test.py --games 500 --opponents smart
"""
import argparse
import random

import joblib

import simulate as sim
from core import MODEL_PATH, make_model_strategy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--games", type=int, default=100, help="games per matchup")
    ap.add_argument("--opponents", nargs="+", default=["random", "smart"], choices=["random", "smart"])
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    random.seed(args.seed)

    sim.STRATEGIES["model"] = make_model_strategy(joblib.load(args.model))
    n = args.games
    for opp in args.opponents:
        for side, me in (("P1", 1), ("P2", 2)):
            p1, p2 = ("model", opp) if me == 1 else (opp, "model")
            w = d = 0
            for _ in range(n):
                winner = sim.play_game(p1, p2)[1]
                w += winner == me
                d += winner == 0
            print(f"model as {side} vs {opp}: win {w/n:.1%}  draw {d/n:.1%}  loss {1-(w+d)/n:.1%}")


if __name__ == "__main__":
    main()
