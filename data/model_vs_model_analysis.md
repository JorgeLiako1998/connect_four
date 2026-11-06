# Why models win and lose (400 games, analysis depth 8)

A = `models/model.joblib`  B = `models/model_20k.joblib`

| | A | B |
|---|---|---|
| wins | 237 (59%) | 142 (36%) |
| avg length of games it won | 30.2 moves | 31.5 moves |

Ties: 21 (5%)

## How the winning four was formed

| | horizontal | vertical | diagonal / | diagonal \ |
|---|---|---|---|---|
| A wins | 132 (56%) | 7 (3%) | 49 (21%) | 49 (21%) |
| B wins | 74 (52%) | 1 (1%) | 33 (23%) | 34 (24%) |

## Why each model LOST

### Model A: lost 142 games
- 122 (86%): blunder
- 20 (14%): already lost out of the random opening
- blunder came at move 28 on median (earliest 12, latest 33)
- columns of the losing move: edge (0,6) 43, mid (1,5) 37, inner (2,4) 26, centre (3) 16
- at play time (depth 6) the model gave the losing move avg win-chance 68%
- 23 were direct (the move lost by force within 8 plies), 99 were deep (the loss only showed 2 moves later)
- for direct blunders, best safe move had avg win-chance 57% vs 65% for the losing move; the model preferred the losing move in 23 of 23; safe alternatives left: avg 2.0 of 7 columns
- direct blunder type: loses by force 23

### Model B: lost 237 games
- 217 (92%): blunder
- 20 (8%): already lost out of the random opening
- blunder came at move 25 on median (earliest 10, latest 33)
- columns of the losing move: edge (0,6) 94, mid (1,5) 68, inner (2,4) 38, centre (3) 17
- at play time (depth 6) the model gave the losing move avg win-chance 66%
- 84 were direct (the move lost by force within 8 plies), 133 were deep (the loss only showed 2 moves later)
- for direct blunders, best safe move had avg win-chance 51% vs 62% for the losing move; the model preferred the losing move in 84 of 84; safe alternatives left: avg 2.5 of 7 columns
- direct blunder type: loses by force 84

## Where each model likes to play (moves after the random opening)

| | col 0 | col 1 | col 2 | col 3 | col 4 | col 5 | col 6 |
|---|---|---|---|---|---|---|---|
| A | 14% | 15% | 15% | 16% | 14% | 14% | 12% |
| B | 17% | 15% | 14% | 13% | 12% | 14% | 14% |

## Example games

### Model A loses (game 218, won by B in 32 moves)
Position before move 25, model A (X) to move:

```
. . . X O . .
O . . O X . .
O . . X O . X
X . O X X . O
X . X O O . O
O . X O X . X
0 1 2 3 4 5 6
```
- Model A played column **6** (shallow-search win-chance 45%) which loses by force.
- Safe move was column **2** (44% win-chance), but the model rated it lower.
- Rest of game: 6 2 0 1 2 1 1 1

### Model A loses (game 248, won by B in 32 moves)
Position before move 25, model A (X) to move:

```
. . . X O . .
O . . O X . .
O . . X O . X
X . O X X . O
X . X O O . O
O . X O X . X
0 1 2 3 4 5 6
```
- Model A played column **6** (shallow-search win-chance 45%) which loses by force.
- Safe move was column **2** (44% win-chance), but the model rated it lower.
- Rest of game: 6 2 0 1 2 1 1 1

### Model B loses (game 371, won by A in 28 moves)
Position before move 21, model B (X) to move:

```
. . X . . . .
. . O . . . .
. X X . X . .
. O X . O . .
O X O O O . O
X X O X X . O
0 1 2 3 4 5 6
```
- Model B played column **4** (shallow-search win-chance 98%) which loses by force.
- Safe move was column **6** (98% win-chance), but the model rated it lower.
- Rest of game: 4 6 6 5 5 5 0 3

### Model B loses (game 293, won by A in 36 moves)
Position before move 29, model B (X) to move:

```
. . . . . O X
. . . O . X O
. O . O . O O
O X X X . X X
X X X O . O X
O X O O X O X
0 1 2 3 4 5 6
```
- Model B played column **1** (shallow-search win-chance 100%) which loses by force.
- Safe move was column **3** (100% win-chance), but the model rated it lower.
- Rest of game: 1 3 0 1 0 0 2 2

