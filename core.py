"""Shared helpers: board encoding, dataset building, model-based move scoring."""
import numpy as np

from simulate import Board, COLS, ROWS

MODEL_PATH = "models/model.joblib"


def encode(grid, me):
    """42 cells -> 84 features: [my pieces, opponent pieces]."""
    g = np.array(grid)
    return np.concatenate([(g == me).ravel(), (g == 3 - me).ravel()]).astype(np.float32)


def mirror_features(X):
    """Flip boards left-right. X has shape (n, 84) = 2 planes of 6x7."""
    return X.reshape(-1, 2, ROWS, COLS)[:, :, :, ::-1].reshape(len(X), -1)


def build_dataset(df, max_games, per_game=None, seed=0, mirror=True, as_float=True):
    """Positions after moves, labelled with the final result from the point of
    view of the player who just moved (2 = won, 1 = draw, 0 = lost).
    Mirrored boards are added too. If per_game is set, only that many random
    positions are kept from each game (keeps memory manageable on huge datasets).
    Returns (X float32, y)."""
    rng = np.random.default_rng(seed)
    games = list(zip(df["moves"][:max_games], df["winner"][:max_games]))
    step = 2 if mirror else 1
    cap = sum(per_game if per_game else len(m.split()) for m, _ in games) * step
    X = np.empty((cap, 2 * ROWS * COLS), dtype=np.uint8)
    y = np.empty(cap, dtype=np.uint8)
    n = 0
    for moves_str, winner in games:
        cols = list(map(int, moves_str.split()))
        keep = set(range(len(cols)))
        if per_game and per_game < len(cols):
            keep = set(rng.choice(len(cols), per_game, replace=False).tolist())
        b, player = Board(), 1
        for i, col in enumerate(cols):
            b.drop(col, player)
            if i in keep:
                label = 1 if winner == 0 else (2 if winner == player else 0)
                X[n] = encode(b.grid, player); y[n] = label; n += 1
                if mirror:
                    X[n] = encode([row[::-1] for row in b.grid], player); y[n] = label; n += 1
            player = 3 - player
    X, y = X[:n], y[:n].astype(int)
    return (X.astype(np.float32) if as_float else X), y


def board_from_moves(moves):
    """Replay a column list. Returns (board, player_to_move, winner or None)."""
    b, player = Board(), 1
    for i, col in enumerate(moves):
        if col not in b.valid_moves():
            raise ValueError(f"column {col} is not playable at move {i + 1}")
        row = b.drop(col, player)
        if b.is_win(row, col, player):
            return b, 3 - player, player
        player = 3 - player
    return b, player, None


SEARCH_DEPTH = 10         # plies searched for forced wins/losses
WIN = 1000
ORDER = [3, 2, 4, 1, 5, 0, 6]


def _negamax(board, player, depth, alpha, beta):
    """Value for `player` to move. Only forced wins/losses are scored: +/-(WIN+depth)
    (faster win / slower loss is better); anything unresolved is 0."""
    if depth == 0:
        return 0
    best = -WIN * 2
    moves = [c for c in ORDER if board.heights[c] < ROWS]
    if not moves:
        return 0
    for c in moves:                       # immediate win first
        row = board.drop(c, player)
        won = board.is_win(row, c, player)
        board.undo(c)
        if won:
            return WIN + depth
    for c in moves:
        board.drop(c, player)
        v = -_negamax(board, 3 - player, depth - 1, -beta, -alpha)
        board.undo(c)
        if v > best:
            best = v
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def score_moves(model, board, player, depth=SEARCH_DEPTH):
    """Return {col: dict(score, win, draw, loss, note)} for every legal move.

    1. A search of `depth` plies finds moves that force a win or lose by force
       (including moves that let the opponent win on the next turn).
    2. Everything the search can't settle is ranked by the model:
       score = P(win) - P(loss) for `player` after the move."""
    classes = list(model.classes_) if model is not None else []
    out = {}
    for m in board.valid_moves():
        if board.wins_if_played(m, player):
            out[m] = dict(score=10.0, win=1.0, draw=0.0, loss=0.0, note="wins now")
            continue
        board.drop(m, player)
        v = -_negamax(board, 3 - player, depth - 1, -WIN * 3, WIN * 3)
        if v > 0:
            out[m] = dict(score=5.0 + v / 10000, win=1.0, draw=0.0, loss=0.0, note="forced win")
        elif v < 0:
            note = ("opponent wins next turn" if v == -(WIN + depth - 1)
                    else "loses by force")
            out[m] = dict(score=-5.0 + v / 10000, win=0.0, draw=0.0, loss=1.0, note=note)
        elif model is not None:
            p = model.predict_proba([encode(board.grid, player)])[0]
            d = dict(zip(classes, p))
            out[m] = dict(score=d.get(2, 0) - d.get(0, 0), win=d.get(2, 0),
                          draw=d.get(1, 0), loss=d.get(0, 0), note="")
        else:
            out[m] = dict(score=0.0, win=0.0, draw=0.0, loss=0.0, note="")
        board.undo(m)
    # label moves that stop the opponent's immediate win
    for m in out:
        if not out[m]["note"] and board.wins_if_played(m, 3 - player):
            out[m]["note"] = "blocks opponent win"
    return out


def best_move(model, board, player, depth=SEARCH_DEPTH):
    s = score_moves(model, board, player, depth)
    return max(s, key=lambda m: s[m]["score"])


def make_model_strategy(model, depth=SEARCH_DEPTH):
    return lambda board, player: best_move(model, board, player, depth)


def render(board):
    sym = {0: ".", 1: "X", 2: "O"}
    lines = [" ".join(sym[board.grid[r][c]] for c in range(COLS)) for r in reversed(range(len(board.grid)))]
    return "\n".join(lines) + "\n" + " ".join(map(str, range(COLS)))
