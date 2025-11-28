import datetime as dt
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# NFL data package (make sure it's in requirements.txt)
import nfl_data_py as nfl


# -------------------------------------------------
# Streamlit page config
# -------------------------------------------------
st.set_page_config(
    page_title="NFL Penalty Impact Dashboard",
    page_icon="🏈",
    layout="wide",
)


# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def _get_year_range(start_year: int, end_year: int) -> List[int]:
    return list(range(start_year, end_year + 1))


@st.cache_data(show_spinner=True)
def load_pbp_penalties(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Load NFL play-by-play data using nfl_data_py and filter to penalty plays.
    """
    years = _get_year_range(start_year, end_year)

    # Only pull columns we actually need to keep it lighter
    pbp_cols = [
        "game_id",
        "season",
        "week",
        "posteam",
        "defteam",
        "penalty",
        "penalty_team",
        "penalty_yards",
        "penalty_type",
        "penalty_player_name",
        "game_seconds_remaining",
    ]

    pbp = nfl.import_pbp_data(
        years=years,
        columns=pbp_cols,
        downcast=True,
        cache=False,
    )

    # Filter to penalties only
    penalties = pbp[pbp["penalty"] == 1].copy()

    # Clean / standardize some fields
    penalties["penalty_yards"] = penalties["penalty_yards"].fillna(0).astype(int)
    penalties["penalty_type"] = penalties["penalty_type"].fillna("Unknown")
    penalties["penalty_team"] = penalties["penalty_team"].fillna(penalties["posteam"])
    penalties["penalty_player_name"] = penalties["penalty_player_name"].fillna(
        "Unknown Player"
    )

    return penalties


@st.cache_data(show_spinner=True)
def load_team_records(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Use nfl_data_py schedules to compute team wins, games, and win percentage.
    """
    years = _get_year_range(start_year, end_year)
    sched = nfl.import_schedules(years)

    # Some schedules already have result flags, but we compute to be safe.
    # Home win = 1 if home_score > away_score; same for away.
    sched["home_win"] = (sched["home_score"] > sched["away_score"]).astype(int)
    sched["away_win"] = (sched["away_score"] > sched["home_score"]).astype(int)

    # Aggregate home and away separately
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

    # Merge home and away stats
    team_season = pd.merge(
        home,
        away,
        on=["season", "team"],
        how="outer",
    ).fillna(0)

    team_season["games"] = team_season["games_home"] + team_season["games_away"]
    team_season["wins"] = team_season["wins_home"] + team_season["wins_away"]
    team_season["win_pct"] = np.where(
        team_season["games"] > 0,
        team_season["wins"] / team_season["games"],
        np.nan,
    )

    return team_season[["season", "team", "games", "wins", "win_pct"]]


def build_team_penalty_summary(
    penalties: pd.DataFrame,
    team_records: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate penalties to season-team level and join to win %.
    """
    # Aggregate penalties by season + team (offense side)
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

    # Merge with team records on season + team
    merged = pd.merge(
        team_pen,
        team_records,
        on=["season", "team"],
        how="left",
    )

    # Rate stats
    merged["penalties_per_game"] = np.where(
        merged["games"] > 0,
        merged["total_penalties"] / merged["games"],
        np.nan,
    )
    merged["penalty_yards_per_game"] = np.where(
        merged["games"] > 0,
        merged["total_penalty_yards"] / merged["games"],
        np.nan,
    )

    return merged


def build_player_penalty_summary(penalties: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-player penalty impact metrics and a 'Penalty Impact Score'.
    Lower score = more disciplined.
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

    player["penalties_per_game"] = np.where(
        player["games_flagged"] > 0,
        player["penalty_count"] / player["games_flagged"],
        np.nan,
    )

    # Custom metric
    player["penalty_impact_score"] = (
        player["penalty_yards"] * 0.6 + player["penalty_count"] * 0.4
    )

    # Lower is better discipline; sort ascending
    player = player.sort_values("penalty_impact_score", ascending=True)

    return player


def correlation_summary(team_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlations between wins, penalties, and penalty yards.
    """
    corr_cols = ["total_penalties", "total_penalty_yards", "win_pct"]
    available = [c for c in corr_cols if c in team_summary.columns]
    if len(available) < 2:
        return pd.DataFrame()

    corr = team_summary[available].corr()
    corr = corr.reset_index().melt(
        id_vars="index",
        var_name="variable",
        value_name="correlation",
    )
    corr = corr.rename(columns={"index": "metric"})
    return corr


# -------------------------------------------------
# Sidebar Controls
# -------------------------------------------------
st.sidebar.title("Filters")

current_year = dt.datetime.now().year
start_year, end_year = st.sidebar.slider(
    "Select Year Range",
    min_value=2000,
    max_value=current_year,
    value=(2015, current_year),
)

st.sidebar.markdown("---")

player_filter_mode = st.sidebar.radio(
    "Player list",
    options=["Top 200 by Penalty Impact Score", "All Players"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data source: nflfastR via `nfl_data_py`. Lower Penalty Impact Score = more disciplined."
)


# -------------------------------------------------
# Load Data
# -------------------------------------------------
with st.spinner("Loading NFL penalty and team data..."):
    penalties_df = load_pbp_penalties(start_year, end_year)
    team_records_df = load_team_records(start_year, end_year)
    team_summary_df = build_team_penalty_summary(penalties_df, team_records_df)
    player_summary_df = build_player_penalty_summary(penalties_df)
    corr_long_df = correlation_summary(team_summary_df)


# -------------------------------------------------
# Top-level KPIs
# -------------------------------------------------
st.title("🏈 NFL Penalty Impact Dashboard")

st.caption(
    f"Analyzing the relationship between **penalties** and **winning** from "
    f"**{start_year}–{end_year}**."
)

total_penalties = int(penalties_df.shape[0])
total_penalty_yards = int(penalties_df["penalty_yards"].sum())

team_penalty_totals = (
    team_summary_df.groupby("team")["total_penalties"].sum().reset_index()
)
most_penalized_team = team_penalty_totals.sort_values(
    "total_penalties", ascending=False
).head(1)
least_penalized_team = team_penalty_totals.sort_values(
    "total_penalties", ascending=True
).head(1)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Penalties", f"{total_penalties:,}")
with col2:
    st.metric("Total Penalty Yards", f"{total_penalty_yards:,} yds")
with col3:
    if not most_penalized_team.empty:
        st.metric(
            "Most Penalized Team",
            most_penalized_team["team"].iloc[0],
            f'{int(most_penalized_team["total_penalties"].iloc[0]):,} flags',
        )
with col4:
    if not least_penalized_team.empty:
        st.metric(
            "Least Penalized Team",
            least_penalized_team["team"].iloc[0],
            f'{int(least_penalized_team["total_penalties"].iloc[0]):,} flags',
        )

st.markdown("---")


# -------------------------------------------------
# Tabs: Team Analysis | Player Analysis | Correlations
# -------------------------------------------------
tab_team, tab_player, tab_corr = st.tabs(
    ["📊 Team Analysis", "👤 Player Analysis", "🔗 Correlations & Penalty Types"]
)

# =========================
# TEAM ANALYSIS TAB
# =========================
with tab_team:
    st.subheader("Team-Level Relationship Between Wins and Penalties")

    # Aggregate across selected years at team level
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
    agg_team["win_pct"] = np.where(
        agg_team["games"] > 0, agg_team["wins"] / agg_team["games"], np.nan
    )
    agg_team["penalties_per_game"] = np.where(
        agg_team["games"] > 0,
        agg_team["total_penalties"] / agg_team["games"],
        np.nan,
    )

    # Optional team selection filter
    all_teams = sorted(agg_team["team"].unique())
    selected_teams = st.multiselect(
        "Filter Teams (optional)",
        options=all_teams,
        default=all_teams,
    )

    if selected_teams:
        agg_team_filtered = agg_team[agg_team["team"].isin(selected_teams)]
    else:
        agg_team_filtered = agg_team

    col_a, col_b = st.columns(2)

    # Scatter: penalties vs win %
    with col_a:
        st.markdown("**Scatter: Total Penalties vs Win Percentage**")
        scatter = (
            alt.Chart(agg_team_filtered)
            .mark_circle(size=120)
            .encode(
                x=alt.X("total_penalties:Q", title="Total Penalties"),
                y=alt.Y("win_pct:Q", title="Win Percentage"),
                color=alt.Color("team:N", legend=None),
                tooltip=[
                    alt.Tooltip("team:N"),
                    alt.Tooltip("total_penalties:Q", format=","),
                    alt.Tooltip("total_penalty_yards:Q", format=","),
                    alt.Tooltip("games:Q", format=","),
                    alt.Tooltip("wins:Q", format=","),
                    alt.Tooltip("win_pct:Q", format=".3f"),
                ],
            )
            .interactive()
        )
        st.altair_chart(scatter, use_container_width=True)

    # Bar: penalties per game vs win %
    with col_b:
        st.markdown("**Bar: Penalties Per Game vs Win Percentage**")

        bar_source = agg_team_filtered.sort_values("penalties_per_game", ascending=False)

        bar_chart = (
            alt.Chart(bar_source)
            .mark_bar()
            .encode(
                x=alt.X("team:N", sort="-y", title="Team"),
                y=alt.Y("penalties_per_game:Q", title="Penalties Per Game"),
                tooltip=[
                    "team",
                    alt.Tooltip("penalties_per_game:Q", format=".2f"),
                    alt.Tooltip("win_pct:Q", format=".3f", title="Win %"),
                ],
            )
        )

        win_line = (
            alt.Chart(bar_source)
            .mark_line(point=True)
            .encode(
                x="team:N",
                y=alt.Y("win_pct:Q", axis=alt.Axis(title="Win % (secondary)")),
            )
        )

        st.altair_chart(bar_chart + win_line, use_container_width=True)

    st.markdown("---")
    st.markdown("### Team Penalty & Win Summary (Aggregated)")

    st.dataframe(
        agg_team_filtered.sort_values("total_penalties", ascending=False),
        use_container_width=True,
    )


# =========================
# PLAYER ANALYSIS TAB
# =========================
with tab_player:
    st.subheader("Player Penalty Impact & Discipline Metric")

    if player_filter_mode.startswith("Top 200"):
        player_options_df = player_summary_df.head(200)
    else:
        player_options_df = player_summary_df.copy()

    player_names = player_options_df["penalty_player_name"].tolist()
    if not player_names:
        st.info("No player penalty data available for the selected years.")
    else:
        selected_player = st.selectbox(
            "Select Player",
            options=player_names,
        )

        player_row = player_summary_df[
            player_summary_df["penalty_player_name"] == selected_player
        ].iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Penalties", int(player_row["penalty_count"]))
        with col2:
            st.metric("Penalty Yards", f"{int(player_row['penalty_yards']):,} yds")
        with col3:
            st.metric("Games with a Penalty", int(player_row["games_flagged"]))
        with col4:
            st.metric(
                "Penalty Impact Score",
                f"{player_row['penalty_impact_score']:.1f}",
                help="Lower score = more disciplined.",
            )

        # Penalty type breakdown for this player
        player_penalties = penalties_df[
            penalties_df["penalty_player_name"] == selected_player
        ]
        player_type_counts = (
            player_penalties.groupby("penalty_type")
            .agg(
                count=("penalty", "size"),
                yards=("penalty_yards", "sum"),
            )
            .reset_index()
        )

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Penalty Types for Selected Player**")
            type_bar = (
                alt.Chart(player_type_counts)
                .mark_bar()
                .encode(
                    x=alt.X("penalty_type:N", sort="-y", title="Penalty Type"),
                    y=alt.Y("count:Q", title="Number of Flags"),
                    tooltip=[
                        "penalty_type",
                        alt.Tooltip("count:Q", title="Flags"),
                        alt.Tooltip("yards:Q", title="Yards", format=","),
                    ],
                )
            )
            st.altair_chart(type_bar, use_container_width=True)

        with col_b:
            st.markdown("**Penalty Yards by Type**")
            yards_bar = (
                alt.Chart(player_type_counts)
                .mark_bar()
                .encode(
                    x=alt.X("penalty_type:N", sort="-y", title="Penalty Type"),
                    y=alt.Y("yards:Q", title="Penalty Yards"),
                    tooltip=[
                        "penalty_type",
                        alt.Tooltip("count:Q", title="Flags"),
                        alt.Tooltip("yards:Q", title="Yards", format=","),
                    ],
                )
            )
            st.altair_chart(yards_bar, use_container_width=True)

        st.markdown("---")
        st.markdown("### Player Discipline Rankings (Lower PIS = Better)")

        st.dataframe(
            player_summary_df[
                [
                    "penalty_player_name",
                    "penalty_count",
                    "penalty_yards",
                    "games_flagged",
                    "penalties_per_game",
                    "penalty_impact_score",
                ]
            ]
            .sort_values("penalty_impact_score", ascending=True)
            .reset_index(drop=True),
            use_container_width=True,
        )


# =========================
# CORRELATIONS & PENALTY TYPES TAB
# =========================
with tab_corr:
    st.subheader("Penalty Yards, Types, and Win Correlations")

    col1, col2 = st.columns(2)

    # Correlation heatmap-like chart
    with col1:
        st.markdown("**Correlation of Wins vs Penalties**")

        if corr_long_df.empty:
            st.info("Not enough data to compute correlations.")
        else:
            corr_chart = (
                alt.Chart(corr_long_df)
                .mark_rect()
                .encode(
                    x=alt.X("metric:N", title="Metric"),
                    y=alt.Y("variable:N", title="Metric"),
                    color=alt.Color(
                        "correlation:Q",
                        scale=alt.Scale(scheme="redblue", domain=(-1, 1)),
                    ),
                    tooltip=[
                        "metric",
                        "variable",
                        alt.Tooltip("correlation:Q", format=".3f"),
                    ],
                )
            )
            st.altair_chart(corr_chart, use_container_width=True)

    # Penalty yards by team (total, across years)
    with col2:
        st.markdown("**Total Penalty Yards by Team**")

        team_yards = (
            team_summary_df.groupby("team")["total_penalty_yards"]
            .sum()
            .reset_index()
            .sort_values("total_penalty_yards", ascending=False)
        )
        yards_chart = (
            alt.Chart(team_yards)
            .mark_bar()
            .encode(
                x=alt.X("team:N", sort="-y", title="Team"),
                y=alt.Y("total_penalty_yards:Q", title="Total Penalty Yards"),
                tooltip=[
                    "team",
                    alt.Tooltip("total_penalty_yards:Q", format=","),
                ],
            )
        )
        st.altair_chart(yards_chart, use_container_width=True)

    st.markdown("---")

    # League-wide breakdown of penalties by type
    st.markdown("### League-Wide Penalty Breakdown by Type")

    league_types = (
        penalties_df.groupby("penalty_type")
        .agg(
            count=("penalty", "size"),
            yards=("penalty_yards", "sum"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )

    type_chart = (
        alt.Chart(league_types.head(30))  # limit to top 30 types
        .mark_bar()
        .encode(
            x=alt.X("penalty_type:N", sort="-y", title="Penalty Type"),
            y=alt.Y("count:Q", title="Number of Flags"),
            tooltip=[
                "penalty_type",
                alt.Tooltip("count:Q", title="Flags", format=","),
                alt.Tooltip("yards:Q", title="Yards", format=","),
            ],
        )
    )

    st.altair_chart(type_chart, use_container_width=True)
    st.dataframe(league_types, use_container_width=True)
