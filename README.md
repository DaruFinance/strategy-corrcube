# strategy-corrcube

**Cross-asset correlation cube.**

> Builds the tensor used by [Cross-asset rolling correlation cube](https://daru.finance/projects/strategy-corrcube) and several sibling models. By Daniel Gatto, [daru.finance](https://daru.finance).

Sample Pearson correlation matrix between the asset-level return series of
the 9-asset universe used throughout the strategy-* research stack
(BTC, DOGE, SOL, BNB, EURUSD, USDJPY, EURGBP, XAUUSD, WTI). The portfolio
site renders this matrix as a 3D bar field; this repository is the
reference implementation that produces the underlying matrix from clean
OHLC feeds and ships a deterministic synthetic demo.

## Reproduce

```bash
git clone https://github.com/DaruFinance/strategy-corrcube
cd strategy-corrcube
pip install -e .
python scripts/cross_asset_corr.py
```

Runs the reproducible synthetic demo (block-structured Gaussian returns,
RNG seed = 42) and writes `figures/corrcube.png` + `corrcube.json`. No
external data required.

## Problem statement

For `K` assets with aligned log-returns `(r₁, …, r_K) ∈ ℝ^K`, the sample
Pearson correlation matrix `ρ̂ ∈ ℝ^{K×K}` summarises first-moment
co-movement. The 3D bar field used on the portfolio site encodes
`|ρ̂_{ij}|` as bar height and `sign(ρ̂_{ij})` as colour.

## Usage

```bash
# Synthetic demo (default — no flags needed)
python scripts/cross_asset_corr.py

# Real OHLC: point --data-dir at a directory of CSVs named like
#   BTCUSDT_30m_3_9.csv, EURUSD_1h_clean.csv, …
python scripts/cross_asset_corr.py \
    --data-dir /path/to/ohlc/csvs \
    --assets BTC,DOGE,SOL,BNB,EURUSD,USDJPY,EURGBP,XAUUSD,WTI \
    --last-n 2000
```

Output: `figures/corrcube.png` + `corrcube.json` (consumed by the portfolio
site at <https://github.com/DaruFinance>).

## References

- Pearson, K. (1895). *Note on regression and inheritance in the case of two parents.*

## License

MIT © Daniel Vieira Gatto.
