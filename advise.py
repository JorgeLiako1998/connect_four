#!/usr/bin/env python3
"""Ask the model for the best move in any position.

    python advise.py                      # best opening move (empty board)
    python advise.py --moves 3 3 2        # position after columns 3,3,2 (P1 first)
    python advise.py --interactive        # type moves one at a time, get advice each turn

Columns are 0-6 (left to right). X = player 1, O = player 2.
"""
import argparse

import joblib

from core import MODEL_PATH, board_from_moves, render, score_moves


def show_advice(model, moves):
    board, player, winner = board_from_moves(moves)
    print(render(board))
    if winner:
        print(f"\nGame over: player {winner} ({'X' if winner == 1 else 'O'}) won.")
        return False
    if not board.valid_moves():
        print("\nGame over: draw.")
        return False
    scores = score_moves(model, board, player)
    best = max(scores, key=lambda m: scores[m]["score"])
    print(f"\nPlayer {player} ({'X' if player == 1 else 'O'}) to move. Model's estimate per column:")
    print("col  win%  draw%  loss%  note")
    for m in sorted(scores, key=lambda m: -scores[m]["score"]):
        s = scores[m]
        star = "  <-- best" if m == best else ""
        print(f" {m}   {s['win']:4.0%}  {s['draw']:5.0%}  {s['loss']:5.0%}  {s['note']}{star}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--moves", type=int, nargs="*", default=[], help="columns played so far")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--me", type=int, choices=[1, 2], default=1,
                    help="which player you are in --interactive mode (1 moves first)")
    args = ap.parse_args()
    model = joblib.load(args.model)

    moves = list(args.moves)
    if not args.interactive:
        show_advice(model, moves)
        return

    while True:
        board, player, winner = board_from_moves(moves)
        mine = player == args.me
        if mine:
            if not show_advice(model, moves):
                break
            scores = score_moves(model, board, player)
            best = max(scores, key=lambda m: scores[m]["score"])
            prompt = f"\nYOUR move (column 0-6, Enter = play {best}, 'u' undo, 'q' quit): "
        else:
            print(render(board))
            if winner or not board.valid_moves():
                print("\nGame over.")
                break
            prompt = "\nOPPONENT's move (column 0-6, 'u' undo, 'q' quit): "
        raw = input(prompt).strip()
        if raw == "q":
            break
        if raw == "" and mine:
            raw = str(best)
        if raw == "u":
            moves = moves[:-1]
        elif raw.isdigit() and 0 <= int(raw) <= 6:
            try:
                board_from_moves(moves + [int(raw)])
                moves.append(int(raw))
            except ValueError:
                print("That column is full.")
        print()
    print("moves so far:", " ".join(map(str, moves)))


if __name__ == "__main__":
    main()
