#!/usr/bin/env python3
"""Connect Four with drag and drop, plus live advice from the trained model.

    python game.py              # you are Player 1 (red, moves first)
    python game.py --me 2       # you are Player 2 (yellow)

How to play (you enter BOTH your move and your opponent's):
  * Drag the piece from the top tray and drop it over a column.
  * On your turn the model's suggested column is highlighted (green) and the
    side panel shows win/draw/loss estimates. Double-click the tray piece to
    auto-play the suggestion.
  * Keys: U = undo, N = new game, S = swap which side you are, Esc = quit.
"""
import argparse
import os

import joblib

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

from core import MODEL_PATH, board_from_moves, score_moves  # noqa: E402
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
    def __init__(self, me, model):
        self.me = me
        self.model = model
        self.moves = []
        self.drag = None            # (x, y) while dragging
        self.fall = None            # animation dict
        self.last_click = 0
        self.refresh()

    # ---- state ----
    def refresh(self):
        self.board, self.player, self.winner = board_from_moves(self.moves)
        self.full = not self.board.valid_moves()
        self.win_cells = find_line(self.board.grid, self.winner) if self.winner else []
        self.scores, self.best = None, None
        if self.model and not self.winner and not self.full and self.player == self.me:
            self.scores = score_moves(self.model, self.board, self.player)
            self.best = max(self.scores, key=lambda m: self.scores[m]["score"])

    @property
    def over(self):
        return bool(self.winner) or self.full

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
            self.refresh()

    def new(self):
        self.moves, self.fall, self.drag, self.last_click = [], None, None, 0
        self.refresh()

    def swap(self):
        self.me = 3 - self.me
        self.refresh()

    def column_at(self, x, y):
        if BX <= x < BX + BW and y < BY + BH:
            return (x - BX) // CELL
        return None

    # ---- events ----
    def on_down(self, pos, now):
        if self.over or self.fall:
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
        if not self.over and not self.fall:
            pos = self.drag or TRAY
            pygame.draw.circle(screen, COLORS[self.player], pos, R)
            pygame.draw.circle(screen, (255, 255, 255), pos, R, 2)
            if not self.drag:
                screen.blit(small.render("drag me onto a column", True, DIM), (TRAY[0] + R + 14, TRAY[1] - 8))

        # side panel
        x, y = PANEL_X, 20
        screen.blit(big.render("Connect Four", True, TEXT), (x, y)); y += 46
        screen.blit(small.render(f"You are {NAMES[self.me]} (Player {self.me})", True, COLORS[self.me]), (x, y)); y += 34

        if self.winner:
            who = "YOU WIN!" if self.winner == self.me else "Opponent wins"
            msg, col = who, COLORS[self.winner]
        elif self.full:
            msg, col = "Draw", TEXT
        else:
            mine = self.player == self.me
            msg, col = ("YOUR move" if mine else "OPPONENT's move"), COLORS[self.player]
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
        elif not self.over and self.player != self.me:
            screen.blit(small.render("Drag your opponent's piece to", True, DIM), (x, y)); y += 20
            screen.blit(small.render("the column they played.", True, DIM), (x, y))
        elif self.model is None:
            screen.blit(small.render("No model found: run train.py", True, DIM), (x, y))

        hint = ["U undo   N new   S swap side", "Esc quit"]
        for i, t in enumerate(hint):
            screen.blit(small.render(t, True, DIM), (x, H - 54 + i * 22))

        # moves so far
        screen.blit(small.render("moves: " + " ".join(map(str, self.moves[-18:])), True, DIM), (BX, H - 28))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--me", type=int, choices=[1, 2], default=1)
    ap.add_argument("--model", default=MODEL_PATH)
    args = ap.parse_args()

    try:
        model = joblib.load(args.model)
    except FileNotFoundError:
        print(f"No model at {args.model}; running without advice (run train.py first).")
        model = None

    pygame.init()
    pygame.display.set_caption("Connect Four")
    screen = pygame.display.set_mode((W, H))
    fonts = (pygame.font.SysFont("helvetica", 34, bold=True),
             pygame.font.SysFont("helvetica", 26, bold=True),
             pygame.font.SysFont("helvetica", 16))
    clock = pygame.time.Clock()
    game = Game(args.me, model)

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
        game.update()
        game.draw(screen, fonts)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
