# DSGA Algorithm Stock Selection

This project is a reproduction of the pipeline described in the 2013 research paper Dynamic-Radius Species-Conserving Genetic Algorithm for the Financial Forecasting of Dow Jones Index Stocks by Brown, Pelosi, and Dirska. The code replicates the architecture of the Dynamic-radius Species-conserving Genetic Algorithm (DSGA) to forecast weekly stock movements using engineered classification rules. The implementation uses the original dataset from the UCI Machine Learning Repository and achieves results comparable to those reported in the paper.

## What this project does

- Reproduces the stock selection framework from the 2013 DSGA research paper.
- Implements a custom species-conserving genetic algorithm with dynamic niching.
- Encodes trading rules as 43-bit individuals, covering price change, volume, and dividend features.
- Applies rule-based filters to predict optimal stocks for each week in Q2 2011.
- Evaluates predictions based on weekly returns and compares performance to the paper’s reported outcomes.

  ## What this project offers

- A hands-on look at applying evolutionary algorithms to financial time series data.
- A complete, reproducible implementation of a specialized algorithm from academic research.
- A flexible pipeline for experimentation and extension to other datasets or fitness functions.
