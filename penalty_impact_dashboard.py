import datetime as dt
from typing import List
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# -------------------------------------------------
# Streamlit Config
# -------------------------------------------------
st.set_page_config(
    page_title="hello",
    page_icon="",
    layout="wide",
)

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------

def _get_year_range(start_year: int, end_year: int):
    return list(range(start_year, end_year + 1))


# -------------------------------------------------
# Load Play-by-Play Data (FASTR API)
# -------------------------------------------------

@st.cache_data(show_spinner=True)
def load_fastr_pbp(years):
    """
    Loads NFL play-by-play from NFLverse FastR CSV files.
    Handles:
      - 1999–2022 from /play_by_play/{year}.csv.gz
      - 2023 from /play_by_play/regular/2023.csv.gz
    Skips missing years (2024+).
    """
    frames = []

    for y in years:
        url = None

        # Standard years: 1999–2022
        if 1999 <= y <= 2022:
            url = f"https://github.com/nflverse/nflfastR-data/raw/master/data/play_by_play/{y}.csv.gz"

        # Special year: 2023
        elif y == 2023:
            url = "https://github.com/nflverse/nflfastR-data/raw/master/data/play_by_play/regular/2023.csv.gz"

        else:
            st.warning(f"⚠️ Play-by-play data for {y} is not available. Skipping...")
            continue

        try:
            df = pd.read_csv(url, compression="gzip", low_memory=False)
            frames.append(df)
        except Exception as e:
            st.error(f"Error loading PBP for {y}: {e}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# -------------------------------------------------
# Load Schedule Data
# -------------------------------------------------

@st.cache_data(show_spinner=True)
def load_fastr_schedule(years):
    """
    Loads NFL schedules from nflverse.
    Schedules available for 1999–2024.
    """
    frames = []
    for y in years:

        if y > 2024:
            st.warning(f"⚠️ Schedule for {y} does not exist. Skipping...")
            continue

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
# Penalty Extraction
# -------------------------------------------------

@st.cache_data(show_spinner=True)
def load_pbp_penalties(start_year, end_year):
    years = _get_year_range(start_year, end_year)
    pbp = load_fastr_pbp(years)

    if pbp.empty:
        return pd.DataFrame()

    columns = [
        "game_id", "season", "week",
        "posteam", "defteam",
        "penalty", "penalty_team",
        "penalty_yards", "penalty_type",
        "penalty_player_name"
    ]

    pbp = pbp[columns]
    penalties = pbp[pbp["penalty"] == 1].copy()

    penalties["penalty_yards"] = penalties["penalty_yards"].fillna(0).astype(int)
    penalties["penalty_type"] = penalties["penalty_type"].fillna("Unknown")
    penalties["penalty_team"] = penalties["penalty_team"].fillna(penalties["posteam"])
    penalties["penalty_player_name"] = penalties["penalty_player_name"].fillna("Unknown Player")

    return penalties


# -------------------------------------------------
# Build Team Win %
# -------------------------------------------------

@st.cache_data(show_spinner=True)
def load_team_records(start_year, end_year):
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
        .reset_index()
        .rename(columns={"home_team": "team"})
    )

    away = (
        sched.groupby(["season", "away_team"])
        .agg(
            games_away=("game_id", "count"),
            wins_away=("away_win", "sum"),
        )
        .reset_index()
        .rename(columns={"away_team": "team"})
    )

    team_season = pd.merge(home, away, on=["season", "team"], how="outer").fillna(0)
    team_season["games"] = team_season["games_home"] + team_season["games_away"]
    team_season["wins"] = team_season["wins_home"] + team_season["wins_away"]
    team_season["win_pct"] = np.where(team_season["games"] > 0,
                                      team_season["wins"] / team_season["games"],
                                      np.nan)

    return team_season[["season", "team", "games", "wins", "win_pct"]]


# -------------------------------------------------
# Build Summaries
# -------------------------------------------------

def build_team_penalty_summary(penalties, records):
    if penalties.empty or records.empty:
        return pd.DataFrame()

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

    merged = pd.merge(team_pen, records, on=["season", "team"], how="left")
    merged["penalties_per_game"] = merged["total_penalties"] / merged["games"]
    merged["penalty_yards_per_game"] = merged["total_penalty_yards"] / merged["games"]

    return merged


def build_player_penalty_summary(penalties):
    if penalties.empty:
        return pd.DataFrame()

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
    player["penalty_impact_score"] = player["penalty_yards"] * 0.6 + player["penalty_count"] * 0.4

    return player.sort_values("penalty_impact_score")


def correlation_summary(team_summary):
    if team_summary.empty:
        return pd.DataFrame()

    corr = team_summary[["total_penalties", "total_penalty_yards", "win_pct"]].corr()

    corr = corr.reset_index().melt(id_vars="index")
    corr.columns = ["metric", "variable", "correlation"]
    return corr


# =========================
# Sidebar UI
# =========================
st.sidebar.title("Filters")

start_year, end_year = st.sidebar.slider(
    "Select Year Range",
    min_value=1999,
    max_value=2023,  # ONLY available through 2023!!
    value=(2015, 2023),
)

player_filter_mode = st.sidebar.radio(
    "Player list",
    ["Top 200 by Penalty Impact Score", "All Players"],
)

# =========================
# Load Data
# =========================
with st.spinner("Loading NFL data..."):
    penalties_df = load_pbp_penalties(start_year, end_year)
    team_records_df = load_team_records(start_year, end_year)
    team_summary_df = build_team_penalty_summary(penalties_df, team_records_df)
    player_summary_df = build_player_penalty_summary(penalties_df)
    corr_long_df = correlation_summary(team_summary_df)

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("🏈 NFL Penalty Impact Dashboard")
st.caption(f"Analyzing penalties vs winning from {start_year}–{end_year}.")

# KPIs
total_penalties = int(penalties_df.shape[0])
total_penalty_yards = int(penalties_df["penalty_yards"].sum())

team_totals = team_summary_df.groupby("team")["total_penalties"].sum().reset_index()
most_pen = team_totals.sort_values("total_penalties", ascending=False).head(1)
least_pen = team_totals.sort_values("total_penalties", ascending=True).head(1)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Penalties", f"{total_penalties:,}")
col2.metric("Penalty Yards", f"{total_penalty_yards:,} yds")
col3.metric("Most Penalized Team", most_pen["team"].iloc[0] if not most_pen.empty else "-", "")
col4.metric("Least Penalized Team", least_pen["team"].iloc[0] if not least_pen.empty else "-", "")

# Tabs
tab_team, tab_player, tab_corr = st.tabs(["📊 Team Analysis", "👤 Player Analysis", "🔗 Correlations"])

# =========================
# TEAM TAB
# =========================
with tab_team:
    st.header("Team-Level Penalty Impact")

    if team_summary_df.empty:
        st.warning("No data loaded.")
    else:
        agg_team = (
            team_summary_df.groupby("team")
            .agg(
                total_penalties=("total_penalties", "sum"),
                total_penalty_yards=("total_penalty_yards", "sum"),
                games=("games", "sum"),
                wins=("wins", "sum"),
            )
            .reset_index()
        )
        agg_team["win_pct"] = agg_team["wins"] / agg_team["games"]
        agg_team["penalties_per_game"] = agg_team["total_penalties"] / agg_team["games"]

        scatter = (
            alt.Chart(agg_team)
            .mark_circle(size=120)
            .encode(
                x="total_penalties:Q",
                y="win_pct:Q",
                color="team:N",
                tooltip=["team", "total_penalties", "win_pct"],
            )
            .interactive()
        )
        st.altair_chart(scatter, use_container_width=True)

# =========================
# PLAYER TAB
# =========================
with tab_player:
    st.header("Player Discipline Rankings")

    if player_summary_df.empty:
        st.warning("No player data available.")
    else:
        st.dataframe(player_summary_df, use_container_width=True)

# =========================
# CORRELATION TAB
# =========================
with tab_corr:
    st.header("Correlations")

    if corr_long_df.empty:
        st.warning("No correlation data available.")
    else:
        heatmap = (
            alt.Chart(corr_long_df)
            .mark_rect()
            .encode(
                x="metric:N",
                y="variable:N",
                color="correlation:Q",
                tooltip=["metric", "variable", "correlation"],
            )
        )
        st.altair_chart(heatmap, use_container_width=True)
