#!/usr/bin/env python3
"""Explain WHY a model won or lost in model-vs-model games.

    python analyze_games.py --csv data/mvm_400.csv
    python analyze_games.py --csv data/mvm_400.csv --depth 8 --examples 3

For every decisive game it replays the moves and looks at each position where
the eventual loser had to move, using a deeper search (--depth) than the one
used during play. The first move that turns a not-yet-lost position into a
forced loss is the "blunder". It then reports:
  * which model blundered, how early, and in which columns
  * how the winner's four was formed (horizontal / vertical / diagonal)
  * what the loser's model thought of the blunder at play time
  * a few example games with the board shown just before the blunder
Writes the report to data/model_vs_model_analysis.md as well.
"""
import argparse
import collections
import io
import statistics
import sys

import joblib
import pandas as pd

from core import board_from_moves, render, score_moves
from simulate import Board, COLS, ROWS

LOST = -4.0           # score_moves gives <= -5 for a forced loss
DIRS = {(0, 1): "horizontal", (1, 0): "vertical", (1, 1): "diagonal /", (1, -1): "diagonal \\"}


def win_direction(board, col, player):
    row = board.heights[col] - 1
    for (dr, dc), name in DIRS.items():
        n = 1
        for sign in (1, -1):
            r, c = row + dr * sign, col + dc * sign
            while 0 <= r < ROWS and 0 <= c < COLS and board.grid[r][c] == player:
                n += 1
                r += dr * sign
                c += dc * sign
        if n >= 4:
            return name
    return "?"


def bucket(col):
    d = abs(col - 3)
    return ["centre (3)", "inner (2,4)", "mid (1,5)", "edge (0,6)"][d]


def analyze_game(moves, winner, loser_model, opening, depth, play_depth):
    """Return dict describing the loser's first blunder (or why none was found)."""
    loser = 3 - winner
    b, player = Board(), 1
    prev = None                       # the loser's previous decision (index, col)
    for i, col in enumerate(moves):
        if player == loser and i >= opening:
            scores = score_moves(loser_model, b, loser, depth)
            best = max(s["score"] for s in scores.values())
            played = scores[col]
            if best <= LOST:
                if prev is None:
                    return dict(kind="already lost out of the random opening", move=i)
                # Every move loses now, but the previous decision looked fine at this
                # depth: it walked into a forced loss just past the search horizon.
                pi, pcol = prev
                before = moves[:pi]
                pb, _, _ = board_from_moves(before)
                shallow = score_moves(loser_model, pb, loser, play_depth)
                return dict(kind="blunder", deep=True, move=pi, col=pcol, moves_before=before,
                            played_score=shallow[pcol]["score"], played_win=shallow[pcol]["win"],
                            safe_col=None, note="loss only visible 2 moves later")
            if played["score"] <= LOST:
                # what did the model think at play time (shallow search)?
                shallow = score_moves(loser_model, b, loser, play_depth)
                safe = [m for m, s in scores.items() if s["score"] > LOST]
                safe_best = max(safe, key=lambda m: shallow[m]["score"])
                return dict(kind="blunder", deep=False, move=i, col=col, moves_before=moves[:i],
                            played_score=shallow[col]["score"], played_win=shallow[col]["win"],
                            safe_col=safe_best, safe_score=shallow[safe_best]["score"],
                            safe_win=shallow[safe_best]["win"], n_safe=len(safe),
                            note=played["note"])
            prev = (i, col)
        b.drop(col, player)
        player = 3 - player
    return dict(kind=f"no blunder found within depth {depth}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/mvm_400.csv")
    ap.add_argument("--a", default="models/model.joblib")
    ap.add_argument("--b", default="models/model_20k.joblib")
    ap.add_argument("--depth", type=int, default=8, help="analysis search depth")
    ap.add_argument("--play-depth", type=int, default=6, help="depth used when the games were played")
    ap.add_argument("--opening", type=int, default=4, help="random opening plies to skip")
    ap.add_argument("--examples", type=int, default=2, help="example games to print per model")
    ap.add_argument("--report", default="data/model_vs_model_analysis.md")
    args = ap.parse_args()

    models = {"A": joblib.load(args.a), "B": joblib.load(args.b)}
    df = pd.read_csv(args.csv)
    out = io.StringIO()

    def say(s=""):
        print(s)
        out.write(s + "\n")

    wins = collections.Counter(); lose_kind = {"A": collections.Counter(), "B": collections.Counter()}
    lengths = {"A": [], "B": []}; directions = {"A": collections.Counter(), "B": collections.Counter()}
    blunders = {"A": [], "B": []}; draws = 0
    cols_used = {"A": collections.Counter(), "B": collections.Counter()}

    print(f"analysing {len(df)} games at depth {args.depth} (this takes a few minutes)...", file=sys.stderr)
    for gi, r in df.iterrows():
        moves = list(map(int, r["moves"].split()))
        names = {1: r["p1_model"], 2: r["p2_model"]}
        for i, c in enumerate(moves):
            if i >= args.opening:
                cols_used[names[1 if i % 2 == 0 else 2]][c] += 1
        w = int(r["winner"])
        if w == 0:
            draws += 1
            continue
        wm, lm = names[w], names[3 - w]
        wins[wm] += 1
        lengths[wm].append(len(moves))
        b, _, _ = board_from_moves(moves)
        directions[wm][win_direction(b, moves[-1], w)] += 1
        res = analyze_game(moves, w, models[lm], args.opening, args.depth, args.play_depth)
        res["game"] = gi; res["winner_model"] = wm; res["moves"] = moves; res["len"] = len(moves)
        lose_kind[lm][res["kind"]] += 1
        if res["kind"] == "blunder":
            blunders[lm].append(res)
        if (gi + 1) % 50 == 0:
            print(f"  {gi + 1}/{len(df)}", file=sys.stderr, flush=True)

    n = len(df)
    say(f"# Why models win and lose ({n} games, analysis depth {args.depth})\n")
    say(f"A = `{args.a}`  B = `{args.b}`\n")
    say(f"| | A | B |\n|---|---|---|")
    say(f"| wins | {wins['A']} ({wins['A']/n:.0%}) | {wins['B']} ({wins['B']/n:.0%}) |")
    say(f"| avg length of games it won | {statistics.mean(lengths['A']) if lengths['A'] else 0:.1f} moves | "
        f"{statistics.mean(lengths['B']) if lengths['B'] else 0:.1f} moves |")
    say(f"\nTies: {draws} ({draws/n:.0%})\n")

    say("## How the winning four was formed\n")
    say("| | " + " | ".join(DIRS.values()) + " |\n|---|" + "---|" * len(DIRS))
    for m in "AB":
        tot = sum(directions[m].values()) or 1
        say(f"| {m} wins | " + " | ".join(f"{directions[m][d]} ({directions[m][d]/tot:.0%})" for d in DIRS.values()) + " |")

    say("\n## Why each model LOST\n")
    for m in "AB":
        lost = sum(lose_kind[m].values())
        say(f"### Model {m}: lost {lost} games")
        for k, v in lose_kind[m].most_common():
            say(f"- {v} ({v/max(lost,1):.0%}): {k}")
        bl = blunders[m]
        if bl:
            mv = [x["move"] + 1 for x in bl]
            say(f"- blunder came at move {statistics.median(mv):.0f} on median (earliest {min(mv)}, latest {max(mv)})")
            cc = collections.Counter(bucket(x["col"]) for x in bl)
            say("- columns of the losing move: " + ", ".join(f"{k} {v}" for k, v in cc.most_common()))
            say(f"- at play time (depth {args.play_depth}) the model gave the losing move avg win-chance "
                f"{statistics.mean(x['played_win'] for x in bl):.0%}")
            direct = [x for x in bl if not x["deep"]]
            say(f"- {len(direct)} were direct (the move lost by force within {args.depth} plies), "
                f"{len(bl) - len(direct)} were deep (the loss only showed 2 moves later)")
            if direct:
                say(f"- for direct blunders, best safe move had avg win-chance "
                    f"{statistics.mean(x['safe_win'] for x in direct):.0%} vs "
                    f"{statistics.mean(x['played_win'] for x in direct):.0%} for the losing move; "
                    f"the model preferred the losing move in "
                    f"{sum(x['played_score'] >= x['safe_score'] for x in direct)} of {len(direct)}; "
                    f"safe alternatives left: avg {statistics.mean(x['n_safe'] for x in direct):.1f} of 7 columns")
            nt = collections.Counter(x["note"] or "forced loss further ahead" for x in direct)
            if nt:
                say("- direct blunder type: " + ", ".join(f"{k} {v}" for k, v in nt.most_common()))
        say("")

    say("## Where each model likes to play (moves after the random opening)\n")
    say("| | " + " | ".join(f"col {c}" for c in range(COLS)) + " |\n|---|" + "---|" * COLS)
    for m in "AB":
        tot = sum(cols_used[m].values()) or 1
        say(f"| {m} | " + " | ".join(f"{cols_used[m][c]/tot:.0%}" for c in range(COLS)) + " |")

    say(f"\n## Example games\n")
    for m in "AB":
        direct = [x for x in blunders[m] if not x["deep"]]
        bl = sorted(direct, key=lambda x: -(x["safe_score"] - x["played_score"]))[:args.examples]
        for x in bl:
            b, p, _ = board_from_moves(x["moves_before"])
            say(f"### Model {m} loses (game {x['game']}, won by {x['winner_model']} in {x['len']} moves)")
            say(f"Position before move {x['move'] + 1}, model {m} ({'X' if p == 1 else 'O'}) to move:\n")
            say("```\n" + render(b) + "\n```")
            say(f"- Model {m} played column **{x['col']}** (shallow-search win-chance {x['played_win']:.0%}) "
                f"which loses by force.")
            say(f"- Safe move was column **{x['safe_col']}** ({x['safe_win']:.0%} win-chance), "
                f"but the model rated it lower.")
            say(f"- Rest of game: " + " ".join(map(str, x["moves"][x["move"]:])) + "\n")

    with open(args.report, "w") as f:
        f.write(out.getvalue())
    print(f"\nreport saved to {args.report}", file=sys.stderr)


if __name__ == "__main__":
    main()
