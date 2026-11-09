"""
data_pull.py
------------
Pulls raw data from nflverse (via nfl_data_py) and saves one CSV per
season to data/raw/nflverse/.

Outputs
-------
data/raw/nflverse/
    player_stats_{year}.csv     weekly player stats (offense only)
    snap_counts_{year}.csv      snap counts + snap pct by player/week
    depth_charts_{year}.csv     pre-season depth chart rank
    schedules_{year}.csv        game schedule (used for SOS)
    sos_{year}.csv              derived: opponent avg win-pct per team

data/raw/pull_log.csv           append-only record of every pull

Usage
-----
    python src/data_pull.py                     # pull all years
    python src/data_pull.py --years 2023 2024   # pull specific years

Requirements
------------
    pip install nfl_data_py pandas
"""

import argparse
import csv
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import nfl_data_py as nfl


SEASONS = list(range(2005, 2025))          # 2005–2024 inclusive
POSITIONS = ["RB", "WR", "TE"]  

RAW_DIR = Path("data/raw/nflverse")
LOG_PATH = Path("data/raw/pull_log.csv")

LOG_FIELDS = ["source", "table", "year", "rows", "pulled_by", "pull_date", "notes"]


# Helpers

def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log(source: str, table: str, year: int | str, rows: int, notes: str = ""):
    """Append one row to the pull log."""
    write_header = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "source":     source,
            "table":      table,
            "year":       year,
            "rows":       rows,
            "pulled_by":  os.environ.get("USER", "unknown"),
            "pull_date":  date.today().isoformat(),
            "notes":      notes,
        })


def save(df: pd.DataFrame, name: str, year: int, source: str, notes: str = ""):
    path = RAW_DIR / f"{name}_{year}.csv"
    df.to_csv(path, index=False)
    print(f"  saved {path}  ({len(df)} rows)")
    log(source, name, year, len(df), notes)


# ---------------------------------------------------------------------------
# Pull functions
# ---------------------------------------------------------------------------

def pull_player_stats(years: list[int]):
    """
    Weekly offensive player stats from nfl_data_py.import_weekly_data().
    We keep skill-position players and the columns needed for feature
    engineering.  One file per season.
    """
    print("\n[1/4] Player stats")

    keep_cols = [
        "player_id", "player_name", "position", "recent_team",
        "season", "week", "season_type",
        # receiving
        "targets", "receptions", "receiving_yards", "receiving_tds",
        "receiving_fumbles", "receiving_fumbles_lost",
        # rushing
        "carries", "rushing_yards", "rushing_tds",
        "rushing_fumbles", "rushing_fumbles_lost",
        # passing
        "completions", "attempts", "passing_yards", "passing_tds",
        "interceptions", "sacks",
        # fantasy
        "fantasy_points", "fantasy_points_ppr",
    ]

    for year in years:
        print(f"  {year}...")
        try:
            df = nfl.import_weekly_data(years=[year])
        except Exception as e:
            print(f"  ERROR pulling player_stats {year}: {e}", file=sys.stderr)
            continue

        # Filter to skill positions and regular season only
        df = df[
            df["position"].isin(POSITIONS) &
            (df["season_type"] == "REG")
        ].copy()

        # Keep only columns we need (drop missing gracefully)
        cols = [c for c in keep_cols if c in df.columns]
        df = df[cols]

        save(df, "player_stats", year, "nflverse/import_weekly_data")


def pull_snap_counts(years: list[int]):
    """
    Snap counts from nfl_data_py.import_snap_counts().
    Snap percentage is the key feature — it captures role stability.
    """
    print("\n[2/4] Snap counts")

    keep_cols = [
        "player_id", "player", "position", "team",
        "season", "week",
        "offense_snaps", "offense_pct",
        "defense_snaps", "defense_pct",   # keep for TE dual-role cases
    ]

    for year in years:
        print(f"  {year}...")
        try:
            df = nfl.import_snap_counts(years=[year])
        except Exception as e:
            print(f"  ERROR pulling snap_counts {year}: {e}", file=sys.stderr)
            continue

        df = df[df["position"].isin(POSITIONS)].copy()
        cols = [c for c in keep_cols if c in df.columns]
        df = df[cols]

        save(df, "snap_counts", year, "nflverse/import_snap_counts")


def pull_depth_charts(years: list[int]):
    """
    Pre-season depth chart rankings from nfl_data_py.import_depth_charts().
    We keep only Week 1 (the last depth chart before the season starts)
    as a proxy for pre-season role.
    """
    print("\n[3/4] Depth charts")

    keep_cols = [
        "season", "club_code", "week",
        "gsis_id",           # maps to player_id in other tables
        "full_name", "position",
        "depth_team",        # 1 = starter, 2 = backup, etc.
        "depth_position",    # e.g. WR1, WR2 — finer than depth_team
    ]

    for year in years:
        print(f"  {year}...")
        try:
            df = nfl.import_depth_charts(years=[year])
        except Exception as e:
            print(f"  ERROR pulling depth_charts {year}: {e}", file=sys.stderr)
            continue

        df = df[df["position"].isin(POSITIONS)].copy()

        # Keep only pre-season Week 1 depth chart
        # (lowest week number available; avoids in-season shuffling leaking in)
        if "week" in df.columns:
            min_week = df["week"].min()
            df = df[df["week"] == min_week]

        cols = [c for c in keep_cols if c in df.columns]
        df = df[cols]

        save(
            df, "depth_charts", year, "nflverse/import_depth_charts",
            notes=f"week={df['week'].iloc[0] if 'week' in df.columns else 'unknown'}"
        )


def pull_schedules_and_sos(years: list[int]):
    """
    Game schedules from nfl_data_py.import_schedules().
    Saves raw schedules, then derives a Strength of Schedule (SOS) table
    per team per season.

    SOS method
    ----------
    For each team T in season Y:
        1. Collect all opponents O_1 … O_17 (or 18 in 2021+).
        2. For each opponent O_i, compute their win percentage in season Y
           using ONLY games NOT against T (avoids double-counting).
        3. SOS = mean of opponent win percentages.

    This is the standard NFL SOS metric and is computable entirely from
    the schedule table, with no external data needed.
    """
    print("\n[4/4] Schedules + Strength of Schedule")

    keep_schedule_cols = [
        "game_id", "season", "week", "game_type",
        "home_team", "away_team",
        "home_score", "away_score",
        "result",           # home_score - away_score; negative = away win
    ]

    for year in years:
        print(f"  {year}...")
        try:
            df = nfl.import_schedules(years=[year])
        except Exception as e:
            print(f"  ERROR pulling schedules {year}: {e}", file=sys.stderr)
            continue

        # Regular season only
        df = df[df["game_type"] == "REG"].copy()

        cols = [c for c in keep_schedule_cols if c in df.columns]
        sched = df[cols].copy()

        save(sched, "schedules", year, "nflverse/import_schedules")

        # --- Derive SOS ---
        sos = _compute_sos(sched, year)
        if sos is not None:
            save(sos, "sos", year, "derived/sos",
                 notes="opponent avg win-pct excl. head-to-head games")


def _compute_sos(sched: pd.DataFrame, year: int) -> pd.DataFrame | None:
    """
    Given a regular-season schedule DataFrame with columns
    [home_team, away_team, home_score, away_score],
    return a DataFrame with columns [team, season, sos, opp_games_played].

    SOS = mean win-pct of all opponents, computed excluding
    the matchup against the team itself.
    """
    if sched.empty:
        return None

    # Build a flat game-result table: one row per team per game
    # result_for = 1 if that team won, 0 if lost, 0.5 if tied
    records = []
    for _, row in sched.iterrows():
        h, a = row["home_team"], row["away_team"]
        hs, as_ = row.get("home_score"), row.get("away_score")

        if pd.isna(hs) or pd.isna(as_):
            continue  # game not yet played / postponed

        if hs > as_:
            h_win, a_win = 1, 0
        elif as_ > hs:
            h_win, a_win = 0, 1
        else:
            h_win = a_win = 0.5

        records.append({"team": h, "opponent": a, "win": h_win})
        records.append({"team": a, "opponent": h, "win": a_win})

    if not records:
        return None

    games = pd.DataFrame(records)

    # Win pct per team (full season)
    win_pct = (
        games.groupby("team")["win"]
        .agg(win_pct="mean", games_played="count")
        .reset_index()
    )
    wp_map = win_pct.set_index("team")["win_pct"].to_dict()

    # For each team, compute mean opponent win-pct (excl. head-to-head games)
    sos_rows = []
    for team in games["team"].unique():
        team_games = games[games["team"] == team]
        opponents = team_games["opponent"].tolist()

        opp_wp = []
        for opp in opponents:
            # Opponent's win-pct in games NOT against this team
            opp_games = games[
                (games["team"] == opp) & (games["opponent"] != team)
            ]
            if opp_games.empty:
                # Fallback: use full win-pct if no other games exist
                opp_wp.append(wp_map.get(opp, 0.5))
            else:
                opp_wp.append(opp_games["win"].mean())

        sos_rows.append({
            "team":             team,
            "season":           year,
            "sos":              round(sum(opp_wp) / len(opp_wp), 4) if opp_wp else None,
            "opp_games_played": len(opp_wp),
        })

    return pd.DataFrame(sos_rows).sort_values("sos", ascending=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Pull nflverse data to data/raw/")
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=SEASONS,
        help=f"Seasons to pull (default: {SEASONS[0]}–{SEASONS[-1]})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    years = sorted(set(args.years))

    print(f"Pulling seasons: {years}")
    print(f"Output dir:      {RAW_DIR.resolve()}")

    ensure_dirs()

    pull_player_stats(years)
    pull_snap_counts(years)
    pull_depth_charts(years)
    pull_schedules_and_sos(years)

    print(f"\nDone. Pull log at {LOG_PATH.resolve()}")


if __name__ == "__main__":
    main()