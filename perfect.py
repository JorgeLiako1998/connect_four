"""Hybrid perfect-play Connect Four: exact C solver + learned model as a fallback.

    from perfect import PerfectPlayer
    p = PerfectPlayer(model)                 # model = loaded models/model.joblib (or None)
    col, info = p.choose(moves)              # moves = columns played so far, P1 first
    p.analyse(moves)                         # {col: 'win'|'draw'|'loss'|'unknown'}

How it chooses (per move):
  1. For every legal column it asks the exact solver (c4solver.c) whether the move
     wins, draws or loses by force. Each question has a node budget.
  2. A proven win is played (the fastest one once the board is fuller).
  3. If nothing is proven winning but some moves are still "unknown" (the solver ran
     out of budget, only happens early in the game), it picks among the unknown/drawing
     moves with the model + 10-ply search, never choosing a proven loss.
  4. If everything is proven lost, it resists as long as possible.

Connect Four is a first-player win. So as Player 1, opening in column 3, it keeps the
win once the solver can see it. As Player 2 it can only win or draw if the opponent errs.
"""
import ctypes
import os
import subprocess

from core import board_from_moves, score_moves

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "libc4.dylib" if os.uname().sysname == "Darwin" else "libc4.so")
SRC = os.path.join(HERE, "c4solver.c")
ORDER = [3, 2, 4, 1, 5, 0, 6]
UNKNOWN, ILLEGAL = -999, -1000
STRONG_FROM = 14            # exact (fast-win) scores are cheap only once the board has filled


def _load():
    if not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(SRC):
        subprocess.check_call(["clang", "-O3", "-march=native", "-shared", "-fPIC", SRC, "-o", LIB])
    lib = ctypes.CDLL(LIB)
    lib.c4_init.argtypes = [ctypes.c_int]
    lib.c4_set_budget.argtypes = [ctypes.c_longlong]
    lib.c4_solve.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.c4_init(24)         # 16M table entries, ~270 MB
    return lib


class PerfectPlayer:
    def __init__(self, model=None, budget=30_000_000, depth=10):
        """budget = max solver nodes per question (~1-3 s); depth = fallback search depth."""
        self.lib = _load()
        self.model = model
        self.budget = budget
        self.depth = depth
        self.cache = {}                     # (sequence, weak) -> score, only for solved results

    # ---- solver access ----
    def _solve(self, seq, weak, budget=None):
        key = (seq, weak)
        if key in self.cache:
            return self.cache[key]
        if budget is None:
            # the first few moves are too big to solve in a second or two, so don't waste time there
            budget = self.budget if len(seq) >= 8 else self.budget // 10
        self.lib.c4_set_budget(budget)
        v = self.lib.c4_solve(seq.encode(), int(weak))
        if v != UNKNOWN:
            self.cache[key] = v
        return v

    def _status(self, moves, col, board):
        """Outcome of playing `col` for the player to move: win / draw / loss / unknown."""
        if board.wins_if_played(col, self._mover(moves)):
            return "win", None
        seq = "".join(map(str, moves)) + str(col)
        v = self._solve(seq, True)
        if v == UNKNOWN:
            return "unknown", None
        # v is from the opponent's point of view
        return ("win" if v < 0 else "draw" if v == 0 else "loss"), v

    @staticmethod
    def _mover(moves):
        return 1 if len(moves) % 2 == 0 else 2

    def analyse(self, moves):
        board, _, _ = board_from_moves(moves)
        return {c: self._status(moves, c, board)[0] for c in board.valid_moves()}

    # ---- move choice ----
    def _model_pick(self, moves, allowed):
        board, player, _ = board_from_moves(moves)
        if self.model is None or len(allowed) == 1:
            return min(allowed, key=ORDER.index)
        s = score_moves(self.model, board, player, self.depth)
        return max(allowed, key=lambda c: (s[c]["score"], -ORDER.index(c)))

    def _fastest(self, moves, cols, prefer_max):
        """Among proven-same-class cols, use exact scores (fast win / slow loss)."""
        if len(cols) == 1 or len(moves) < STRONG_FROM:
            return None
        board, _, _ = board_from_moves(moves)
        best = {}
        for c in cols:
            if board.wins_if_played(c, self._mover(moves)):
                return c
            v = self._solve("".join(map(str, moves)) + str(c), False)
            if v == UNKNOWN:
                return None
            best[c] = -v                      # from our point of view
        top = max(best.values())
        return min((c for c in cols if best[c] == top), key=ORDER.index)

    def choose(self, moves):
        """Return (column, info) where info is 'win'/'draw'/'loss'/'unknown' with how it was found."""
        board, player, _ = board_from_moves(moves)
        legal = board.valid_moves()
        status = {}
        for c in sorted(legal, key=ORDER.index):
            status[c] = self._status(moves, c, board)[0]
            if status[c] == "win":            # no need to look at the rest
                wins = [k for k, v in status.items() if v == "win"]
                pick = self._fastest(moves, wins, True) or wins[0]
                return pick, "proven win"
        unknown = [c for c, s in status.items() if s == "unknown"]
        draws = [c for c, s in status.items() if s == "draw"]
        if unknown:
            return self._model_pick(moves, unknown + draws), "not solved yet (model + search)"
        if draws:
            return self._model_pick(moves, draws), "proven draw"
        pick = self._fastest(moves, legal, False) or self._model_pick(moves, legal)
        return pick, "lost position (resisting)"
