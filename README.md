# time-pressure
Analyzing the effect of time pressure in chess

## Approach

This repository provides analysis code for studying time pressure in chess using Jupyter notebooks. The filtered data necessary to run the code is also provided. The raw data is available from the Lichess open database at https://database.lichess.org.

## File description

- `analysis_time.ipynb`: conducts the time pressure analysis for the selected positions
- `scale_time.ipynb`: scales the time pressure analysis to more positions
- `voc_timeipynb`: computes VOCs to select individual positions for the time pressure analysis
- `compute_move_voc.py`: functions for computing VOC
- `convert_stockfish_scores.py`: functions for converting Stockfish results to win probability
- `Data`: contains preprocessed data at the game and move level for the individual positions and at scale (zipped)
