import datetime as dt
from typing import List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# -------------------------------------------------
# Streamlit page config
# -------------------------------------------------
st.set_page_config(
    page_title="NFL Penalty Impact Dashboard",
    page_icon="🏈",
    layout="wide",
)


# -------------------------------------------------
# FASTR API LOADER (REPLACES nfl_data_py)
# -------------------------------------------------

@st.cache_data(show_spinner=True)
def load_fastr_pbp(years):
    """
    Load NFL play-by-play data directly from NFLverse FastR CSV URLs.
    This avoids nfl_data_py and works on Streamlit Cloud.
    """
    frames = []
    for y in years:
        url = f"https://github.com/nflverse/nflfastR-data/raw/master/data/play_by_play/{y}.csv.gz"
        try:
            df = pd.read_csv(url, compression="gzip", low_memory=False)
            frames.append(df)
        except Exception as e:
            st.error(f"Error loading {y}: {e}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=True)
def load_fastr_schedule(years):
    """
    Load NFL schedule data using FastR schedule files.
    """
    frames = []
    for y in years:
        url = f"https://github.com/nflverse/nflfastR-data/raw/master/data/schedules/sched_{y}.csv"
        try:
            df = pd.read_csv(url, low_memory=False)
            frames.append(df)
        except Exception as e:
            st.error(f"Error loading schedule for {y}: {e}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def _get_year_range(start_year: int, end_year: int):
    return list(range(start_year, end_year + 1))


@st.cache_data(show_spinner=True)
def load_pbp_penalties(start_year: int, end_year: int):
    """
    Load PBP and extract penalties.
    """
    years = _get_year_range(start_year, end_year)
    pbp = load_fastr_pbp(years)

    if pbp.empty:
        return pd.DataFrame()

    penalty_cols = [
        "game_id", "season", "week", "posteam", "defteam",
        "penalty", "penalty_team", "penalty_yards", "penalty_type",
        "penalty_player_name"
    ]

    pbp = pbp[penalty_cols]

    penalties = pbp[pbp["penalty"] == 1].copy()
    penalties["penalty_yards"] = penalties["penalty_yards"].fillna(0).astype(int)
    penalties["penalty_type"] = penalties["penalty_type"].fillna("Unknown")
    penalties["penalty_team"] = penalties["penalty_team"].fillna(penalties["posteam"])
    penalties["penalty_player_name"] = penalties["penalty_player_name"].fillna("Unknown Player")

    return penalties


@st.cache_data(show_spinner=True)
def load_team_records(start_year: int, end_year: int):
    """
    Load team records from FastR schedule data.
    """
    years = _get_year_range(start_year, end_year)
    sched = load_fastr_schedule(years)

    if sched.empty:
        return pd.DataFrame()

    sched["home_win"] = (sched["home_score"] > sched["away_score"]).astype(int)
    sched["away_win"] = (sched["away_score"] > sched["home_score"]).astype(int)

    home = (
        sched.groupby(["season", "home_team"])
        .agg(
            games_home=("game_id", "count"),
            wins_home=("home_win", "sum"),
        )
        .reset_index().rename(columns={"home_team": "team"})
    )

    away = (
        sched.groupby(["season", "away_team"])
        .agg(
            games_away=("game_id", "count"),
            wins_away=("away_win", "sum"),
        )
        .reset_index().rename(columns={"away_team": "team"})
    )

    team_season = pd.merge(home, away, on=["season", "team"], how="outer").fillna(0)
    team_season["games"] = team_season["games_home"] + team_season["games_away"]
    team_season["wins"] = team_season["wins_home"] + team_season["wins_away"]
    team_season["win_pct"] = team_season["wins"] / team_season["games"]

    return team_season[["season", "team", "games", "wins", "win_pct"]]


def build_team_penalty_summary(penalties, team_records):
    """
    Same logic as before.
    """
    team_pen = (
        penalties.groupby(["season", "penalty_team"])
        .agg(
            total_penalties=("penalty", "size"),
            total_penalty_yards=("penalty_yards", "sum"),
            unique_games=("game_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"penalty_team": "team"})
    )

    merged = pd.merge(team_pen, team_records, on=["season", "team"], how="left")
    merged["penalties_per_game"] = merged["total_penalties"] / merged["games"]
    merged["penalty_yards_per_game"] = merged["total_penalty_yards"] / merged["games"]

    return merged


def build_player_penalty_summary(penalties):
    """
    Same logic as before.
    """
    player = (
        penalties.groupby("penalty_player_name")
        .agg(
            penalty_count=("penalty", "size"),
            penalty_yards=("penalty_yards", "sum"),
            games_flagged=("game_id", "nunique"),
        )
        .reset_index()
    )

    player["penalties_per_game"] = player["penalty_count"] / player["games_flagged"]
    player["penalty_impact_score"] = (
        player["penalty_yards"] * 0.6 + player["penalty_count"] * 0.4
    )

    return player.sort_values("penalty_impact_score")


def correlation_summary(team_summary):
    corr_cols = ["total_penalties", "total_penalty_yards", "win_pct"]
    corr = team_summary[corr_cols].corr()

    corr = corr.reset_index().melt(id_vars="index")
    corr.rename(columns={"index": "metric", "variable": "variable", "value": "correlation"}, inplace=True)
    return corr


# -------------------------------------------------
# Sidebar UI
# -------------------------------------------------
st.sidebar.title("Filters")

current_year = dt.datetime.now().year
start_year, end_year = st.sidebar.slider(
    "Select Year Range",
    min_value=2000,
    max_value=current_year,
    value=(2015, current_year),
)

player_filter_mode = st.sidebar.radio(
    "Player list",
    ["Top 200 by Penalty Impact Score", "All Players"],
)


# -------------------------------------------------
# Load Data
# -------------------------------------------------
with st.spinner("Loading NFL data..."):
    penalties_df = load_pbp_penalties(start_year, end_year)
    team_records_df = load_team_records(start_year, end_year)
    team_summary_df = build_team_penalty_summary(penalties_df, team_records_df)
    player_summary_df = build_player_penalty_summary(penalties_df)
    corr_long_df = correlation_summary(team_summary_df)

# -------------------------------------------------
# UI (unchanged)
# -------------------------------------------------
# [Your whole visualization code stays EXACTLY the same below here]

st.title("🏈 NFL Penalty Impact Dashboard")
st.caption(f"Analyzing penalties vs winning from {start_year}–{end_year}.")

# Existing visualization logic continues...
