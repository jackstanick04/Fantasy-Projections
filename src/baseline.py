"""
Two really simple baselines for predicting points per game:

1. Last-season PPR/game, predict y = prior season result
2. 3-year average PPR/game, predict y = mean of a player's last 1–3 available prior seasons 
(fewer seasons used when history is short)

Outputs: results/tables/baseline_results.csv, results/tables/baseline_by_pos.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT         = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT / "data"

MODEL_TABLE  = DATA_DIR / "model_table.csv"   
STATS_ALL    = DATA_DIR / "player_stats_all.csv" 

RESULTS_DIR  = ROOT / "results" / "tables"

Y_COL        = "y_ppr_per_game"  
POSITIONS    = ["RB", "WR", "TE"]
MIN_GAMES    = 12               

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)



def build_lag_history(stats_path: Path) -> pd.DataFrame:
    """
    Returns a DF with the following columns:
        player_id, season, fp_per_game, fp_lag1, fp_lag2, fp_lag3
    """
    raw = pd.read_csv(stats_path, usecols=["player_id", "season", "games",
                                            "fantasy_points_ppr"])

    # Compute per-game PPR for every player-season that clears MIN_GAMES.
    raw["fp_per_game"] = np.where(
        raw["games"] >= MIN_GAMES,
        raw["fantasy_points_ppr"] / raw["games"],
        np.nan,                        # too few games → not a usable prior
    )

    raw = raw.sort_values(["player_id", "season"]).reset_index(drop=True)

    # Shift within each player to get lag columns.
    raw["fp_lag1"] = raw.groupby("player_id")["fp_per_game"].shift(1)
    raw["fp_lag2"] = raw.groupby("player_id")["fp_per_game"].shift(2)
    raw["fp_lag3"] = raw.groupby("player_id")["fp_per_game"].shift(3)

    return raw[["player_id", "season", "fp_lag1", "fp_lag2", "fp_lag3"]]


def load_model_table(table_path: Path, lag_df: pd.DataFrame) -> pd.DataFrame:
  
    table = pd.read_csv(table_path)

    missing = {"player_id", "season", "split", "position", Y_COL} - set(table.columns)
    if missing:
        raise ValueError(
            f"model_table.csv is missing expected columns: {missing}. "
            "Re-run build_data.py."
        )

    table = table.merge(lag_df, on=["player_id", "season"], how="left")

    # How many rows got no lag at all?
    no_lag = table["fp_lag1"].isna().sum()
    log.info(
        "Lag join complete. %d / %d rows have no fp_lag1 "
        "(rookies or data gaps — excluded from baseline evaluation).",
        no_lag, len(table),
    )
    return table

# Predictions

def predict_last_season(df: pd.DataFrame) -> pd.Series:
    """Predict y = last season's PPR per game (fp_lag1)."""
    return df["fp_lag1"].copy()


def predict_three_year_avg(df: pd.DataFrame) -> pd.Series:
    """
    Predict y = mean of fp_lag1, fp_lag2, fp_lag3, using however many
    are non-NaN.  Returns NaN only when fp_lag1 is also NaN (no history).
    """
    lag_cols = ["fp_lag1", "fp_lag2", "fp_lag3"]
    return df[lag_cols].mean(axis=1, skipna=True)


# Evaluation

def _metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str, model: str) -> dict:
    return {
        "model":  model,
        "split":  label,
        "n":      len(y_true),
        "rmse":   round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mae":    round(float(mean_absolute_error(y_true, y_pred)), 4),
        "bias":   round(float(np.mean(y_pred - y_true)), 4),  # + = overpredict
    }


def evaluate(df: pd.DataFrame, pred_col: str, model_name: str,) -> tuple[list[dict], pd.DataFrame]:

    overall_rows = []

    for split in ["val", "test"]:
        sub = df[df["split"] == split].copy()
        mask = sub[pred_col].notna() & sub[Y_COL].notna()
        sub = sub[mask]

        if sub.empty:
            log.warning("No usable rows in split=%s for model=%s.", split, model_name)
            continue

        overall_rows.append(
            _metrics(sub[Y_COL].values, sub[pred_col].values, split, model_name)
        )

    # Position breakdown on test only
    pos_rows = []
    test_df = df[df["split"] == "test"].copy()
    for pos, grp in test_df.groupby("position"):
        mask = grp[pred_col].notna() & grp[Y_COL].notna()
        grp = grp[mask]
        if grp.empty:
            continue
        row = _metrics(grp[Y_COL].values, grp[pred_col].values, "test", model_name)
        row["position"] = pos
        pos_rows.append(row)

    pos_df = pd.DataFrame(pos_rows).sort_values(["model", "rmse"])
    return overall_rows, pos_df



# Main

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Building lag history from %s ...", STATS_ALL)
    lag_df = build_lag_history(STATS_ALL)

    log.info("Loading model table from %s ...", MODEL_TABLE)
    df = load_model_table(MODEL_TABLE, lag_df)


    baselines = {
        "last_season":    predict_last_season,
        "three_year_avg": predict_three_year_avg,
    }

    all_overall, all_pos = [], []

    for name, fn in baselines.items():
        df[f"pred_{name}"] = fn(df)
        overall_rows, pos_df = evaluate(df, f"pred_{name}", name)
        all_overall.extend(overall_rows)
        all_pos.append(pos_df)
        for row in overall_rows:
            log.info(
                "%-16s  split=%-4s  n=%4d  RMSE=%.4f  MAE=%.4f  bias=%+.4f",
                name, row["split"], row["n"],
                row["rmse"], row["mae"], row["bias"],
            )


    overall_df = pd.DataFrame(all_overall)
    pos_df_all = pd.concat(all_pos, ignore_index=True)

    overall_path = RESULTS_DIR / "baseline_results.csv"
    pos_path     = RESULTS_DIR / "baseline_by_pos.csv"

    overall_df.to_csv(overall_path, index=False)
    pos_df_all.to_csv(pos_path, index=False)

    log.info("Wrote %s", overall_path)
    log.info("Wrote %s", pos_path)

    print("\n=== Overall baseline results ===")
    print(overall_df.to_string(index=False))
    print("\n=== Test-split results by position ===")
    print(pos_df_all.to_string(index=False))


if __name__ == "__main__":
    main()