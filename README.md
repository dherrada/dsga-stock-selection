# DSGA Stock Selection

This project is a reproduction of the pipeline described in the 2013 research paper Dynamic-Radius Species-Conserving Genetic Algorithm for the Financial Forecasting of Dow Jones Index Stocks by Brown, Pelosi, and Dirska. The code replicates the architecture of the Dynamic-radius Species-conserving Genetic Algorithm (DSGA) to forecast weekly stock movements using engineered classification rules. The implementation uses the original dataset from the UCI Machine Learning Repository and achieves results similar to those reported in the paper.

## What this project does

- Reproduces the stock selection framework from the 2013 DSGA research paper.
- Implements a custom species-conserving genetic algorithm with dynamic niching.
- Encodes trading rules as 43-bit individuals, covering price change, volume, and dividend features.
- Applies rule-based filters to predict optimal stocks for each week in Q2 2011.
- Evaluates predictions based on weekly returns and compares performance to the paper’s reported outcomes.

## What this project offers

- A look at applying evolutionary algorithms to financial time series data.
- A complete implementation of a specialized algorithm from academic research.
- A pipeline for experimentation and extension to other datasets or fitness functions.

## Setup

Clone the repo and install dependencies:

```bash
git clone git@github.com:your-username/dsga-stock-selection.git
cd dsga-stock-selection
pip install -r requirements.txt
```

Run the full pipeline and generate results:

```bash
python main.py
```

Results will be saved in the results/ directory.

## Citations

Dataset:

Brown, M. (2013). Dow Jones Index [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5788V.

Paper:

Brown, M.S., Pelosi, M.J., & Dirska, H. (2013). Dynamic-Radius Species-Conserving Genetic Algorithm for the Financial Forecasting of Dow Jones Index Stocks. IAPR International Conference on Machine Learning and Data Mining in Pattern Recognition.
