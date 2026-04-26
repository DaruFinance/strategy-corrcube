"""Smoke test: the synthetic demo runs end-to-end and produces a non-empty figure."""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "cross_asset_corr.py"


def test_synthetic_demo(tmp_path: Path) -> None:
    out = tmp_path / "figures"
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert res.returncode == 0, res.stderr

    fig = out / "corrcube.png"
    assert fig.exists() and fig.stat().st_size > 5_000

    payload = json.loads((out.parent / "corrcube.json").read_text())
    assert len(payload["labels"]) == 9
    assert len(payload["corr"]) == 9
    assert all(len(row) == 9 for row in payload["corr"])
    # diagonal must be 1.0
    for i in range(9):
        assert abs(payload["corr"][i][i] - 1.0) < 1e-6


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_synthetic_demo(Path(d))
        print("ok")
