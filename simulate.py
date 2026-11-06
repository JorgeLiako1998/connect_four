#!/usr/bin/env python3
"""Run N Connect Four games and save every move plus the outcome.

Usage:
    python simulate.py -n 10000
    python simulate.py -n 5000 --p1 random --p2 smart --seed 42 --out games.jsonl

Output: one JSON object per line (JSONL), e.g.
    {"game_id": 0, "p1": "random", "p2": "smart",
     "moves": [3, 3, 2, ...],      # column (0-6) played, in order; P1 moves first
     "winner": 1,                  # 1, 2, or 0 for draw
     "num_moves": 23}

Because moves are just a column sequence, any game can be replayed exactly.
"""
import argparse
import csv
import json
import os
import random
import time
from collections import Counter

ROWS, COLS = 6, 7


class Board:
    def __init__(self):
        self.grid = [[0] * COLS for _ in range(ROWS)]  # grid[0] is the bottom row
        self.heights = [0] * COLS

    def valid_moves(self):
        return [c for c in range(COLS) if self.heights[c] < ROWS]

    def drop(self, col, player):
        row = self.heights[col]
        self.grid[row][col] = player
        self.heights[col] += 1
        return row

    def undo(self, col):
        self.heights[col] -= 1
        self.grid[self.heights[col]][col] = 0

    def is_win(self, row, col, player):
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            for sign in (1, -1):
                r, c = row + dr * sign, col + dc * sign
                while 0 <= r < ROWS and 0 <= c < COLS and self.grid[r][c] == player:
                    count += 1
                    r += dr * sign
                    c += dc * sign
            if count >= 4:
                return True
        return False

    def wins_if_played(self, col, player):
        row = self.drop(col, player)
        won = self.is_win(row, col, player)
        self.undo(col)
        return won


# ---- strategies: f(board, player) -> column -------------------------------

def random_strategy(board, player):
    return random.choice(board.valid_moves())


def smart_strategy(board, player):
    """Win if possible, block opponent's win, else prefer centre columns."""
    moves = board.valid_moves()
    for m in moves:
        if board.wins_if_played(m, player):
            return m
    for m in moves:
        if board.wins_if_played(m, 3 - player):
            return m
    weights = [4 - abs(m - 3) for m in moves]  # centre weighted
    return random.choices(moves, weights=weights)[0]


STRATEGIES = {"random": random_strategy, "smart": smart_strategy}


def play_game(p1, p2):
    board = Board()
    strategies = {1: STRATEGIES[p1], 2: STRATEGIES[p2]}
    moves, player = [], 1
    while True:
        col = strategies[player](board, player)
        row = board.drop(col, player)
        moves.append(col)
        if board.is_win(row, col, player):
            return moves, player
        if len(moves) == ROWS * COLS:
            return moves, 0
        player = 3 - player


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("-n", "--iterations", type=int, default=1000, help="number of games")
    ap.add_argument("--p1", choices=STRATEGIES, default="random", help="player 1 strategy")
    ap.add_argument("--p2", choices=STRATEGIES, default="random", help="player 2 strategy")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    ap.add_argument("--out", default=None, help="output file (default: data/games_<timestamp>.jsonl)")
    ap.add_argument("--append", action="store_true", help="append to --out instead of overwriting")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", f"games_{int(time.time())}.csv"
    )
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    as_csv = out.endswith(".csv")

    results = Counter()
    write_header = as_csv and not (args.append and os.path.exists(out) and os.path.getsize(out) > 0)
    with open(out, "a" if args.append else "w", newline="") as f:
        writer = csv.writer(f) if as_csv else None
        if write_header:
            writer.writerow(["game_id", "p1", "p2", "moves", "winner", "num_moves"])
        for i in range(args.iterations):
            moves, winner = play_game(args.p1, args.p2)
            results[winner] += 1
            if as_csv:
                # moves stored as space-separated columns, e.g. "3 3 2 4"
                writer.writerow([i, args.p1, args.p2, " ".join(map(str, moves)), winner, len(moves)])
            else:
                f.write(json.dumps({
                    "game_id": i, "p1": args.p1, "p2": args.p2,
                    "moves": moves, "winner": winner, "num_moves": len(moves),
                }) + "\n")

    n = args.iterations
    print(f"Saved {n} games to {out}")
    print(f"P1 ({args.p1}) wins: {results[1]} ({results[1]/n:.1%})")
    print(f"P2 ({args.p2}) wins: {results[2]} ({results[2]/n:.1%})")
    print(f"Draws: {results[0]} ({results[0]/n:.1%})")


if __name__ == "__main__":
    main()
