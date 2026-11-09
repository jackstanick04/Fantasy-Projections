"""Builds/filters raw data into actionable X and y -- ran after data_import.py"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
IN_PATH = DATA_DIR / "player_stats_all.csv"

POSITIONS = ["RB", "WR", "TE"]
MIN_GAMES = 12        # for both target and previous
Y_PER_GAME = True     # predicts ppg v. total
SEED = 0
TRAIN_FRAC, VAL_FRAC = 0.8, 0.1   # the remaining 0.1 is test

COUNT_COLS = ["rushing_yards", "rushing_tds", "carries", "targets",
              "receptions", "receiving_yards", "receiving_tds"]
FEATURE_COLS = COUNT_COLS + ["target_share"] # target share doesn't depend on per game
META_COLS = ["player_id", "position", "season", "games", "recent_team"]
Y_COL = "y_ppr_per_game" if Y_PER_GAME else "y_ppr_total"


"""merges rows between player/year and aggregates all data into one table"""
def build_table(df: pd.DataFrame) -> pd.DataFrame:

    # target year: valid player and their points scored (y)
    t = df[df["position"].isin(POSITIONS) & (df["games"] >= MIN_GAMES)].copy()
    t[Y_COL] = t["fantasy_points_ppr"] / t["games"] if Y_PER_GAME else t["fantasy_points_ppr"]
    t = t[META_COLS + [Y_COL]]

    # prior year: prior year stats per game
    p = df[df["games"] >= MIN_GAMES].copy()
    p[COUNT_COLS] = p[COUNT_COLS].div(p["games"], axis=0)
    p = p[["player_id", "season"] + FEATURE_COLS]
    p = p.assign(season=p["season"] + 1) # "incrementing" the season so that it aligns with the target data above in the merge

    # inner join between player/year (must appear in both)
    table = t.merge(p, on=["player_id", "season"], how="inner").reset_index(drop=True)
    table = table.dropna(subset=FEATURE_COLS + [Y_COL]).reset_index(drop=True)
    return table

"""Random 80/10/10 train validation test split"""
def data_split(table: pd.DataFrame) -> pd.DataFrame:
    train, temp = train_test_split(table, test_size=0.2, random_state=SEED)
    val, test = train_test_split(temp, test_size=0.5, random_state=SEED)
    table = table.copy()

    # makes a new column that has each row's identity as train, val, or test
    table.loc[train.index, "split"] = "train"
    table.loc[val.index, "split"] = "val"
    table.loc[test.index, "split"] = "test"
    return table


if __name__ == "__main__":
    df = pd.read_csv(IN_PATH)
    table = data_split(build_table(df))

    model_cols = FEATURE_COLS + [Y_COL]   # the only columns the model ever sees
    for name in ["train", "val", "test"]: # each data set as a csv
        table[table["split"] == name][model_cols].to_csv(DATA_DIR / f"{name}.csv", index=False)
    table.to_csv(DATA_DIR / "model_table.csv", index=False)  # all data as a csv

    """clauded output just to see data stats and verify it works"""

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)

    print("\nsplit sizes:")
    print(table["split"].value_counts().reindex(["train", "val", "test"]).to_string())
    print("\nrows per target season:")
    print(table["season"].value_counts().sort_index().to_string())
    print("\nrows per position:")
    print(table["position"].value_counts().to_string())

    train = table[table["split"] == "train"]
    print(f"\nmodel-ready table (train, first 15 rows). Features = prior season per game; "
          f"last column = {Y_COL} (current season):")
    print(train[model_cols].head(15).round(3).to_string(index=False))

    print("\nsummary of model columns (all splits):")
    print(table[model_cols].describe().round(3).T.to_string())