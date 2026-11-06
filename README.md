# Connect Four: how to win, and how to run the code

# 1. How to win a game (based on what we found)

These tips come from 400 games between two trained models (500,000-game model vs 20,000-game model), replayed with a deeper search to find where games were decided. They describe how these computer players win and lose. They are not a proven solution to Connect Four.

**1. Start in the centre.**
- When the models chose their own opening move, the winners picked columns 2, 3 or 4 in 88% of wins (new model) and 71% (old model). Column 3 was the most common start (41% and 31%).
- Columns 0 and 6 almost never started a win.
- The model that used the centre more also won more: 59% against 36%, with 5% ties.

**2. Stay away from the edge columns (0 and 6) unless you have a reason.**
- 35% of the new model's losing moves and 43% of the old model's were in the edge columns. Those columns made up only 26% and 31% of the moves they played.

**3. Games are decided late, by one mistake.**
- Nearly every decisive game had one move after which the loss was forced: 86% of the new model's losses, 92% of the old one's.
- The median blunder came around move 25–28, on a crowded board where only about 2 of 7 columns were still safe.
- Before each move late in the game, check three things:
  - Can the opponent win right now if I play here?
  - Does my move let them win on the cell directly above it?
  - Does it leave them a way to make two threats at once?

**4. Look further ahead than your opponent.**
- 61–81% of the losing moves looked fine to a 6-move look-ahead and only lost two moves later. A deeper search (this project's advice uses 10 plies) catches them.

**5. Don't trust the percentages. Trust the labels.**
- The model was overconfident: it gave losing moves about 65% win chance on average, and 98–100% in some late-game examples.
- The labels from the search are reliable: `wins now`, `forced win`, `blocks opponent win`, `opponent wins next turn`, `loses by force`.
- Treat the percentages only as a tiebreaker between moves that aren't flagged.

**6. Build horizontal and diagonal threats.**
- About half of all wins were horizontal and about 45% were diagonal. Vertical wins were rare (8 of 379), because a vertical threat is easy to block.
- The classic winning shape is an open three on the bottom row, with empty cells on both sides. For example, after the moves 3, 3, 4, 4 (you hold the bottom of columns 3 and 4, your opponent sits on top of each), playing column 2 or 5 is a forced win, because you then threaten both ends of the row. The search confirmed this.

**7. Use the tools while you play.** `game.py` or `advise.py --interactive` shows the search labels and the model's estimate for every column on your turn (see section 2).

# 2. How to run the code

## Files

| File | Purpose |
|---|---|
| `simulate.py` | Plays N games and saves every move and result to a CSV |
| `train.py` | Trains the model on the CSV and saves it to `models/model.joblib` |
| `game.py` | Drag-and-drop game window with live advice |
| `advise.py` | Command-line best-move advice for any position |
| `test.py` | Plays the model against `random` and `smart` |
| `model_vs_model.py` | Plays two models against each other and reports wins and ties |
| `analyze_games.py` | Explains why models won or lost in `model_vs_model.py` games |
| `core.py` | Shared code (not run directly) |

## Setup

```bash
cd /Users/jliak/connect_four
pip3 install numpy pandas scikit-learn joblib pygame
```

## Generate games (`simulate.py`)

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

## Train the model (`train.py`)

```bash
python3 train.py                                # ALL positions of ALL games in data/games.csv
python3 train.py --hidden 256 128 --iters 15    # bigger network, up to 15 passes
python3 train.py --max-games 50000              # only the first 50,000 games
python3 train.py --per-game 4                   # sample 4 positions per game (fast)
python3 train.py --data data/other.csv --out models/mine.joblib
```

- It labels each position with the final result (win, draw or loss) for the player who just moved.
- With the default `--per-game 0` it uses every position, trained in chunks so 500,000 games (about 11.5 million positions) fit in 8 GB of memory. Boards are randomly mirrored left to right as extra data.
- 5,000 games are held out for validation. It prints the validation accuracy after each pass, stops early if there's no improvement for 3 passes, and saves the best model so far to `models/model.joblib` after every improving pass.
- A pass over 500,000 games took about 2 minutes with the default network.
- To run it in the background and watch progress:
  ```bash
  python3 -u train.py > models/train.log 2>&1 &
  tail -f models/train.log
  ```
- Don't run two trainings at once. They write to the same model file. Back up a model you want to keep, for example `cp models/model.joblib models/model_old.joblib`.

## Play with advice (`game.py`)

```bash
python3 game.py            # you are Red (Player 1, moves first)
python3 game.py --me 2     # you are Yellow (Player 2)
```

You enter both your move and your opponent's:
- Drag the piece from the top tray and drop it over a column.
- On your turn, the suggested column gets a green marker and the side panel shows a win bar for each column. Double-click the tray piece to play the suggestion.
- On the opponent's turn, drag the piece to the column they played.
- Keys: `U` undo, `N` new game, `S` swap which side you are, `Esc` quit.

## Command-line advice (`advise.py`)

```bash
python3 advise.py                        # best opening move
python3 advise.py --moves 3 3 2          # position after columns 3, 3, 2 (P1 first)
python3 advise.py --interactive          # you are Player 1
python3 advise.py --interactive --me 2   # you are Player 2
```

Interactive mode alternates turns. On your turn, type a column (0-6) or press Enter to play the suggestion. On the opponent's turn, type the column they played. `u` undoes and `q` quits.

Board symbols: `X` = Player 1, `O` = Player 2.

## Test the model (`test.py`)

```bash
python3 test.py --games 15       # games per matchup; larger numbers are slow
python3 test.py --games 20 --opponents smart
```

## Model vs model (`model_vs_model.py`)

```bash
python3 model_vs_model.py                                   # models/model.joblib vs models/model_20k.joblib, 100 games
python3 model_vs_model.py --games 400 --csv data/mvm_400.csv
python3 model_vs_model.py --a models/mine.joblib --b models/model.joblib
python3 model_vs_model.py --depth 0                         # raw models, no search
```

Each game starts with 4 random moves (`--opening`), and every opening is played twice with the sides swapped, so neither model gets an unfair first-move advantage. It prints A's wins, B's wins and ties. 100 games takes about 25 seconds at the default search depth.

## Explain wins and losses (`analyze_games.py`)

```bash
python3 analyze_games.py --csv data/mvm_400.csv --depth 8
```

It replays every decisive game, finds the move where the loser threw the game away, and writes a report with example boards to `data/model_vs_model_analysis.md`. It takes a few minutes for 400 games.

## How the advice is calculated

1. **Search:** a search about 10 plies deep (roughly 5 moves each) finds moves that win by force, lose by force, or let the opponent win next turn. These are shown as `wins now`, `forced win`, `blocks opponent win`, `opponent wins next turn` and `loses by force`.
2. **Model:** all other moves are ranked by the neural network's estimate of win, draw and loss for that position.

The search is exact for tactics. The model's percentages are only a rough guess.

## Notes

- Train on `smart` vs `smart` games. In `random` vs `smart` data, one side almost always wins, so the model learns a biased picture.
- Use `--seed` for repeatable results.
- `models/model.joblib` is loaded by default. Use `--model path` on `game.py`, `advise.py` and `test.py` to load another file.
- `model_vs_model.py` compares `models/model.joblib` with `models/model_20k.joblib` by default. Both files need to exist.
