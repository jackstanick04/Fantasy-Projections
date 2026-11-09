"""fetch 2005-2024 datasets from nflverse, download locally, and create output file"""
from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd

BASE = "https://github.com/nflverse/nflverse-data/releases/download/player_stats/"
YEARS = range(2005, 2025)  
RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/player_stats_all.csv")


"""Columns for inputs, filtering, output/comparison"""

ID_COLS = ["player_id", "position", "season", "games"] #  player/season merging, position/injury filtering
FEATURE_COLS = [
    "recent_team", # not used -- would be for SOS
    "rushing_yards",
    "rushing_tds",
    "carries",
    "targets",
    "receptions",    
    "receiving_yards",  
    "receiving_tds",
    "target_share",
]

TARGET_COL = ["fantasy_points_ppr"]
KEEP = ID_COLS + FEATURE_COLS + TARGET_COL


"""Download/retrieve single season's csv"""
def download_year(year: int) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    path = RAW_DIR / f"player_stats_season_{year}.csv"
    if not path.exists():
        urlretrieve(f"{BASE}player_stats_season_{year}.csv", path)
    return path

"""stores every season's regular season as a df, and concatenates all df into one"""
def load_all() -> pd.DataFrame:
    frames = []
    for year in YEARS:
        df = pd.read_csv(download_year(year))
        df = df[df["season_type"] == "REG"] # only want regular season stats
        frames.append(df[KEEP])
    return pd.concat(frames, ignore_index=True)


"""store aggregated dataframe to a csv for use by build_data file"""
if __name__ == "__main__":
    all_stats = load_all()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_stats.to_csv(OUT_PATH, index=False)