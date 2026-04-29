from __future__ import annotations

from typing import Any

import pandas as pd


def summarize_degradation_results(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    group_cols = ["family", "severity"]
    metric_cols = [col for col in frame.columns if col not in {"sample_id", "family", "severity"}]
    summary = frame.groupby(group_cols, as_index=False)[metric_cols].mean()
    return summary.sort_values(group_cols).reset_index(drop=True)

