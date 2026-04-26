#!/usr/bin/env python3
"""Cross-asset correlation cube.

Two operating modes:

  * synthetic (default):
      Generates 9 correlated-pair Gaussian return series with a block
      structure (intra-crypto, intra-FX, small cross-block) and computes
      the resulting sample Pearson correlation matrix. Reproducible
      (RNG seed = 42), no external data required.

  * --data-dir <dir>:
      Loads OHLC CSVs named with the conventions used in this research
      stack (e.g. BTCUSDT_30m_3_9.csv, EURUSD_1h_clean.csv) and computes
      log-returns + correlation over the last --last-n bars per asset.

Emits:
  figures/corrcube.png   — annotated correlation heatmap (the 3D bar
                           field lives on the portfolio site).
  corrcube.json          — labels, correlation matrix, n_returns.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend so the script runs over SSH / CI
import matplotlib.pyplot as plt


DEFAULT_ASSETS = ["BTC", "DOGE", "SOL", "BNB",
                  "EURUSD", "USDJPY", "EURGBP", "XAUUSD", "WTI"]


def find_file(data_dir: Path, asset: str) -> Path | None:
    # Try the common naming patterns seen across this research stack.
    candidates = [
        data_dir / f"{asset}USDT_30m_3_9.csv",
        data_dir / f"{asset}USDT_1h_3_9.csv",
        data_dir / f"{asset}USDT_15m_3_9.csv",
        data_dir / f"{asset}_1h_clean.csv",
        data_dir / f"{asset}_1h.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    # glob fallback
    for c in data_dir.glob(f"{asset}*.csv"):
        return c
    return None


def log_returns(path: Path, n: int) -> np.ndarray:
    import pandas as pd
    df = pd.read_csv(path)
    col = "close" if "close" in df.columns else df.columns[-1]
    close = df[col].to_numpy(dtype=float)
    close = close[close > 0]
    lr = np.diff(np.log(close))
    return lr[-n:] if len(lr) > n else lr


def synthetic(labels: list[str], n: int = 2000, seed: int = 42) -> np.ndarray:
    """Block-correlated synthetic returns.

    Builds a covariance matrix with a crypto intra-class block (ρ ≈ 0.55),
    an FX intra-class block (ρ ≈ 0.35), and zero cross-block correlation,
    then samples n × k draws via Cholesky factorisation. The Cholesky
    decomposition fails if C is not strictly positive-definite, so a tiny
    1e-6·I jitter is added before the factorisation to absorb any rank loss
    from the block construction.
    """
    rng = np.random.default_rng(seed)
    k = len(labels)
    C = np.eye(k)
    crypto = [i for i, a in enumerate(labels) if a in {"BTC", "DOGE", "SOL", "BNB", "ETH", "LTC"}]
    fx = [i for i, a in enumerate(labels) if a in {"EURUSD", "USDJPY", "EURGBP", "GBPUSD", "AUDUSD"}]
    for i in crypto:
        for j in crypto:
            if i != j:
                C[i, j] = 0.55
    for i in fx:
        for j in fx:
            if i != j:
                C[i, j] = 0.35
    L = np.linalg.cholesky(C + 1e-6 * np.eye(k))
    Z = rng.standard_normal((n, k))
    return Z @ L.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--assets", default=",".join(DEFAULT_ASSETS))
    ap.add_argument("--last-n", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    labels = args.assets.split(",")

    if args.data_dir:
        series = {}
        for a in labels:
            p = find_file(args.data_dir, a)
            if p is None:
                raise SystemExit(f"no OHLC file found for {a} under {args.data_dir}")
            series[a] = log_returns(p, args.last_n)
        m = min(len(s) for s in series.values())
        X = np.stack([series[a][-m:] for a in labels], axis=1)
    else:
        X = synthetic(labels)

    C = np.corrcoef(X, rowvar=False)

    # figure: heatmap (matrix view — 3D cube lives on the website)
    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{C[i, j]:+.2f}", ha="center", va="center",
                    fontsize=8, color="black" if abs(C[i, j]) < 0.5 else "white")
    ax.set_title("Cross-asset correlation")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    fig.tight_layout(); fig.savefig(args.out / "corrcube.png", dpi=160)

    out = {
        "labels": labels,
        "corr": [[round(float(C[i, j]), 3) for j in range(len(labels))] for i in range(len(labels))],
        "n_returns": int(X.shape[0]),
    }
    (args.out.parent / "corrcube.json").write_text(json.dumps(out, indent=2))
    print("wrote", args.out / "corrcube.png", "and", args.out.parent / "corrcube.json")


if __name__ == "__main__":
    main()
