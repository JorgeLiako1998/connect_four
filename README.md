# Connect Four: simulate, train, and get move advice

Simulate Connect Four games, train a model on them, then play with live advice or ask for the best move in any position.

## Files

| File | Purpose |
|---|---|
| `simulate.py` | Plays N games and saves every move and result to a CSV |
| `train.py` | Trains the model on the CSV and saves it to `models/model.joblib` |
| `game.py` | Drag-and-drop game window with live advice |
| `advise.py` | Command-line best-move advice for any position |
| `test.py` | Plays the model against `random` and `smart` |
| `core.py` | Shared code (not run directly) |

## Setup

```bash
cd /Users/jliak/connect_four
pip3 install numpy pandas scikit-learn joblib pygame
```

## 1. Generate games (`simulate.py`)

```bash
# 10,000 random vs random games -> data/games_<timestamp>.csv
python3 simulate.py -n 10000

# smart vs smart, saved to a chosen file
python3 simulate.py -n 500000 --p1 smart --p2 smart --seed 42 --out data/games.csv

# add more games to an existing CSV
python3 simulate.py -n 5000 --p1 smart --p2 smart --out data/games.csv --append
```

Strategies: `random`, and `smart` (wins if it can, blocks, otherwise prefers the centre).

CSV columns: `game_id, p1, p2, moves, winner, num_moves`
- `moves`: space-separated columns (0-6) in play order, P1 first
- `winner`: 1, 2, or 0 for a draw

## 2. Train the model (`train.py`)

```bash
python3 train.py                                # all games in data/games.csv, 4 positions per game
python3 train.py --max-games 50000              # only the first 50,000 games (faster)
python3 train.py --per-game 8 --iters 200       # more data and more training rounds
python3 train.py --data data/other.csv --out models/mine.joblib
```

- It replays each game and labels positions with the final result (win, draw or loss) for the player who just moved. Mirrored boards are added as extra data.
- `--per-game` keeps that many random positions from each game. This matters for big datasets: 500,000 games would be about 23 million rows and won't fit in 8 GB of memory.
- It prints a progress line per training round, then `validation accuracy` and `saved model`.
- To run it in the background and watch progress:
  ```bash
  python3 train.py > models/train.log 2>&1 &
  tail -f models/train.log
  ```
- Don't run two trainings at once. They write to the same model file.

## 3. Play with advice (`game.py`)

```bash
python3 game.py            # you are Red (Player 1, moves first)
python3 game.py --me 2     # you are Yellow (Player 2)
```

You enter both your move and your opponent's:
- Drag the piece from the top tray and drop it over a column.
- On your turn, the suggested column gets a green marker and the side panel shows a win bar for each column. Double-click the tray piece to play the suggestion.
- On the opponent's turn, drag the piece to the column they played.
- Keys: `U` undo, `N` new game, `S` swap which side you are, `Esc` quit.

## 4. Command-line advice (`advise.py`)

```bash
python3 advise.py                        # best opening move
python3 advise.py --moves 3 3 2          # position after columns 3, 3, 2 (P1 first)
python3 advise.py --interactive          # you are Player 1
python3 advise.py --interactive --me 2   # you are Player 2
```

Interactive mode alternates turns. On your turn, type a column (0-6) or press Enter to play the suggestion. On the opponent's turn, type the column they played. `u` undoes and `q` quits.

Board symbols: `X` = Player 1, `O` = Player 2.

## 5. Test the model (`test.py`)

```bash
python3 test.py --games 15       # games per matchup; larger numbers are slow
python3 test.py --games 20 --opponents smart
```

## How the advice is calculated

1. **Search:** a search about 10 plies deep (roughly 5 moves each) finds moves that win by force, lose by force, or let the opponent win next turn. These are shown as `forced win`, `loses by force`, `opponent wins next turn`, `wins now` and `blocks opponent win`.
2. **Model:** all other moves are ranked by the neural network's estimate of win, draw and loss for that position.

The search is exact for tactics. The model's percentages are only a rough guess.

## Notes

- Train on `smart` vs `smart` games. In `random` vs `smart` data, one side almost always wins, so the model learns a biased picture.
- Use `--seed` for repeatable results.
- `models/model.joblib` is loaded by default. Use `--model path` on `game.py`, `advise.py` and `test.py` to load another file.
