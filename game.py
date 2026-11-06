#!/usr/bin/env python3
"""Connect Four with drag and drop, plus live advice from the trained model.

    python game.py                  # advice mode, you are Player 1 (red, moves first)
    python game.py --me 2           # advice mode, you are Player 2 (yellow)
    python game.py --vs-model       # play AGAINST the model
    python game.py --perfect        # play against the model with the exact solver
    python game.py --vs-model --me 2 --depth 10 --hints

Advice mode (default): you enter BOTH your move and your opponent's.
  * Drag the piece from the top tray and drop it over a column.
  * On your turn the model's suggested column is highlighted (green) and the
    side panel shows win/draw/loss estimates. Double-click the tray piece to
    auto-play the suggestion.

Play-against-model mode (--vs-model): you drag your piece, the model replies by
itself. Hints are off by default (press H to show them). --depth sets how many
plies ahead the model searches (higher = stronger but slower, default 8).

Perfect mode: the model asks an exact solver whether each move wins, draws or
loses, and only falls back to the learned model while a position is still too
big to solve (the first few moves).

Keys: U = undo, N = new game, S = swap sides, H = hints, Esc = quit,
      M = cycle mode: advice -> you vs model -> you vs PERFECT model -> advice.
"""
import argparse
import os

import joblib

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

from core import MODEL_PATH, best_move, board_from_moves, score_moves  # noqa: E402
from simulate import COLS, ROWS  # noqa: E402

CELL = 90
BX, BY = 20, 120                   # board top-left
BW, BH = COLS * CELL, ROWS * CELL
PANEL_X = BX + BW + 30
W, H = PANEL_X + 270, BY + BH + 40

BG = (24, 28, 38)
BLUE = (36, 76, 168)
HOLE = (24, 28, 38)
COLORS = {1: (220, 60, 60), 2: (245, 205, 50)}
NAMES = {1: "Red", 2: "Yellow"}
GREEN = (80, 220, 120)
TEXT = (230, 233, 240)
DIM = (140, 148, 165)
TRAY = (BX + BW // 2, 58)          # where the draggable piece rests
R = CELL // 2 - 8


def find_line(grid, player):
    """Return the 4 winning cells (row, col) for player, or []."""
    for r in range(ROWS):
        for c in range(COLS):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = [(r + dr * i, c + dc * i) for i in range(4)]
                if all(0 <= rr < ROWS and 0 <= cc < COLS and grid[rr][cc] == player for rr, cc in cells):
                    return cells
    return []


def cell_center(row, col):
    return BX + col * CELL + CELL // 2, BY + (ROWS - 1 - row) * CELL + CELL // 2


class Game:
    def __init__(self, me, model, vs_model=False, depth=8, hints=None, perfect=None):
        self.me = me
        self.model = model
        self.perfect = perfect          # PerfectPlayer (created on first use) or None
        self.use_perfect = perfect is not None
        self.last_info = ""
        self.vs_model = vs_model and model is not None
        self.depth = depth
        self.hints = (not self.vs_model) if hints is None else hints
        self.moves = []
        self.drag = None            # (x, y) while dragging
        self.fall = None            # animation dict
        self.last_click = 0
        self.think_at = 0
        self.refresh()

    # ---- state ----
    def refresh(self):
        self.board, self.player, self.winner = board_from_moves(self.moves)
        self.full = not self.board.valid_moves()
        self.win_cells = find_line(self.board.grid, self.winner) if self.winner else []
        self.scores, self.best = None, None
        if self.model and self.hints and not self.over and self.player == self.me:
            self.scores = score_moves(self.model, self.board, self.player)
            self.best = max(self.scores, key=lambda m: self.scores[m]["score"])
        if self.model_turn:
            self.think_at = pygame.time.get_ticks() + 450   # short pause so the move is visible
        if self.over and self.moves:
            result = f"player {self.winner} wins" if self.winner else "draw"
            print(f"game over ({result}), {len(self.moves)} moves: {''.join(map(str, self.moves))}", flush=True)

    @property
    def over(self):
        return bool(self.winner) or self.full

    @property
    def model_turn(self):
        return self.vs_model and not self.over and self.player != self.me

    def play(self, col):
        if self.over or self.fall or col not in self.board.valid_moves():
            return
        row = self.board.heights[col]
        x, _ = cell_center(row, col)
        self.fall = dict(col=col, row=row, player=self.player, x=x, y=BY - R, vy=0.0,
                         target=cell_center(row, col)[1])

    def undo(self):
        if self.moves and not self.fall:
            self.moves.pop()
            if self.vs_model:                    # also take back the model's reply
                _, player, _ = board_from_moves(self.moves)
                if player != self.me and self.moves:
                    self.moves.pop()
            self.refresh()

    def new(self):
        self.moves, self.fall, self.drag, self.last_click = [], None, None, 0
        self.refresh()

    def swap(self):
        self.me = 3 - self.me
        self.refresh()

    def ensure_perfect(self):
        if self.perfect is None:
            from perfect import PerfectPlayer          # compiles the C solver on first use
            self.perfect = PerfectPlayer(self.model, depth=self.depth)

    def toggle_mode(self):
        """Cycle: advice -> you vs model -> you vs PERFECT model -> advice."""
        if self.model is None or self.fall:
            return
        if not self.vs_model:
            self.vs_model, self.use_perfect = True, False
        elif not self.use_perfect:
            self.ensure_perfect()
            self.use_perfect = True
        else:
            self.vs_model, self.use_perfect = False, False
        self.hints = not self.vs_model
        self.last_info = ""
        self.refresh()

    def toggle_hints(self):
        self.hints = not self.hints
        self.refresh()

    def column_at(self, x, y):
        if BX <= x < BX + BW and y < BY + BH:
            return (x - BX) // CELL
        return None

    # ---- events ----
    def on_down(self, pos, now):
        if self.over or self.fall or self.model_turn:
            return
        tx, ty = TRAY
        if (pos[0] - tx) ** 2 + (pos[1] - ty) ** 2 <= (R + 8) ** 2:
            if now - self.last_click < 350 and self.best is not None:
                self.play(self.best)        # double-click = take the suggestion
                self.last_click = 0
                return
            self.last_click = now
            self.drag = pos

    def on_move(self, pos):
        if self.drag:
            self.drag = pos

    def on_up(self, pos):
        if self.drag:
            col = self.column_at(*pos)
            self.drag = None
            if col is not None:
                self.play(col)

    def update(self):
        f = self.fall
        if not f:
            if self.model_turn and pygame.time.get_ticks() >= self.think_at:
                if self.use_perfect:
                    col, self.last_info = self.perfect.choose(self.moves)
                else:
                    col = best_move(self.model, self.board, self.player, self.depth)
                self.play(col)
            return
        f["vy"] += 2.2
        f["y"] += f["vy"]
        if f["y"] >= f["target"]:
            self.moves.append(f["col"])
            self.fall = None
            self.refresh()

    # ---- drawing ----
    def draw(self, screen, fonts):
        screen.fill(BG)
        big, mid, small = fonts
        hover = self.column_at(*self.drag) if self.drag else None

        # column highlight
        if hover is not None:
            pygame.draw.rect(screen, (50, 58, 80), (BX + hover * CELL, BY, CELL, BH))
        if self.best is not None:
            pygame.draw.rect(screen, GREEN, (BX + self.best * CELL + 4, BY - 14, CELL - 8, 8), border_radius=4)

        # pieces (drawn first, board with holes over them is simulated by blue frame)
        pygame.draw.rect(screen, BLUE, (BX, BY, BW, BH), border_radius=12)
        for r in range(ROWS):
            for c in range(COLS):
                v = self.board.grid[r][c]
                pygame.draw.circle(screen, COLORS.get(v, HOLE), cell_center(r, c), R)
        for r, c in self.win_cells:
            pygame.draw.circle(screen, (255, 255, 255), cell_center(r, c), R + 3, 4)
        if self.fall:
            f = self.fall
            pygame.draw.circle(screen, COLORS[f["player"]], (f["x"], int(f["y"])), R)

        # tray + draggable piece
        pygame.draw.rect(screen, (34, 40, 55), (BX, 8, BW, BY - 40), border_radius=12)
        if self.model_turn and not self.fall:
            pygame.draw.circle(screen, COLORS[self.player], TRAY, R)
            screen.blit(small.render("model is thinking...", True, DIM), (TRAY[0] + R + 14, TRAY[1] - 8))
        elif not self.over and not self.fall:
            pos = self.drag or TRAY
            pygame.draw.circle(screen, COLORS[self.player], pos, R)
            pygame.draw.circle(screen, (255, 255, 255), pos, R, 2)
            if not self.drag:
                screen.blit(small.render("drag me onto a column", True, DIM), (TRAY[0] + R + 14, TRAY[1] - 8))

        # side panel
        x, y = PANEL_X, 20
        screen.blit(big.render("Connect Four", True, TEXT), (x, y)); y += 46
        screen.blit(small.render(f"You are {NAMES[self.me]} (Player {self.me})", True, COLORS[self.me]), (x, y)); y += 22
        if self.vs_model:
            mode = "Mode: you vs PERFECT model" if self.use_perfect else f"Mode: you vs model (depth {self.depth})"
        else:
            mode = "Mode: advice (you enter both moves)"
        screen.blit(small.render(mode, True, DIM), (x, y)); y += 22
        if self.use_perfect and self.vs_model and self.last_info:
            screen.blit(small.render(f"model: {self.last_info}", True, DIM), (x, y))
        y += 6

        opp = "Model" if self.vs_model else "Opponent"
        if self.winner:
            who = "YOU WIN!" if self.winner == self.me else f"{opp} wins"
            msg, col = who, COLORS[self.winner]
        elif self.full:
            msg, col = "Draw", TEXT
        else:
            mine = self.player == self.me
            msg, col = ("YOUR move" if mine else f"{opp.upper()}'s move"), COLORS[self.player]
        screen.blit(mid.render(msg, True, col), (x, y)); y += 40

        if self.scores:
            screen.blit(small.render("Model advice (win / draw / loss)", True, DIM), (x, y)); y += 24
            for c in range(COLS):
                s = self.scores.get(c)
                label = f"col {c}"
                if s is None:
                    screen.blit(small.render(f"{label}  full", True, DIM), (x, y)); y += 26
                    continue
                best = c == self.best
                screen.blit(small.render(label, True, GREEN if best else TEXT), (x, y))
                bx, bw = x + 60, 130
                pygame.draw.rect(screen, (60, 66, 84), (bx, y + 2, bw, 14), border_radius=3)
                pygame.draw.rect(screen, GREEN if best else (110, 170, 230), (bx, y + 2, int(bw * s["win"]), 14), border_radius=3)
                screen.blit(small.render(f"{s['win']:.0%}", True, TEXT), (bx + bw + 8, y))
                if s["note"]:
                    screen.blit(small.render(s["note"], True, GREEN), (x + 60, y + 16))
                    y += 14
                y += 26
        elif not self.over and self.player != self.me and not self.vs_model:
            screen.blit(small.render("Drag your opponent's piece to", True, DIM), (x, y)); y += 20
            screen.blit(small.render("the column they played.", True, DIM), (x, y))
        elif self.model is None:
            screen.blit(small.render("No model found: run train.py", True, DIM), (x, y))
        elif self.vs_model and not self.hints and not self.over and self.player == self.me:
            screen.blit(small.render("Hints are off (press H to show).", True, DIM), (x, y))

        hint = ["U undo  N new  S swap side", "M mode  H hints  Esc quit"]
        for i, t in enumerate(hint):
            screen.blit(small.render(t, True, DIM), (x, H - 54 + i * 22))

        # moves so far
        text = "moves: " + "".join(map(str, self.moves))     # all moves, copy-paste friendly
        screen.blit(small.render(text, True, DIM), (BX, H - 28))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--me", type=int, choices=[1, 2], default=1)
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--vs-model", action="store_true", help="play against the model")
    ap.add_argument("--depth", type=int, default=8, help="model search depth in vs-model mode")
    ap.add_argument("--hints", action="store_true", help="show advice in vs-model mode")
    ap.add_argument("--perfect", action="store_true",
                    help="vs-model mode: the model uses the exact solver (never loses once the win is solved)")
    args = ap.parse_args()

    try:
        model = joblib.load(args.model)
    except FileNotFoundError:
        if args.vs_model:
            raise SystemExit(f"No model at {args.model}; run train.py first.")
        print(f"No model at {args.model}; running without advice (run train.py first).")
        model = None

    pygame.init()
    pygame.display.set_caption("Connect Four")
    screen = pygame.display.set_mode((W, H))
    fonts = (pygame.font.SysFont("helvetica", 34, bold=True),
             pygame.font.SysFont("helvetica", 26, bold=True),
             pygame.font.SysFont("helvetica", 16))
    clock = pygame.time.Clock()
    perfect = None
    if args.perfect:
        from perfect import PerfectPlayer
        perfect = PerfectPlayer(model, depth=args.depth)
        args.vs_model = True
    game = Game(args.me, model, vs_model=args.vs_model, depth=args.depth,
                hints=True if args.hints else None, perfect=perfect)

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                game.on_down(e.pos, pygame.time.get_ticks())
            elif e.type == pygame.MOUSEMOTION:
                game.on_move(e.pos)
            elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                game.on_up(e.pos)
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_u:
                    game.undo()
                elif e.key == pygame.K_n:
                    game.new()
                elif e.key == pygame.K_s:
                    game.swap()
                elif e.key == pygame.K_m:
                    game.toggle_mode()
                elif e.key == pygame.K_h:
                    game.toggle_hints()
        game.update()
        game.draw(screen, fonts)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
