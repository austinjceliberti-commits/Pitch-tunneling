from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from pybaseball import statcast_pitcher, playerid_lookup, playerid_reverse_lookup

# -------------------------------------------------
# Streamlit page config
# -------------------------------------------------
st.set_page_config(
    page_title="Pitch Tunneling Dashboard",
    page_icon="",
    layout="wide",
)

# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def _ensure_column(df: pd.DataFrame, target: str, source_candidates: Iterable[str]) -> None:
    """Create ``target`` from the first existing column in ``source_candidates``.

    The new column is only filled when the target is missing. This utility lets us
    gracefully handle Statcast schema variants (e.g., ``release_spin_rate`` vs
    ``spin_rate``).
    """

    if target in df.columns:
        return

    for col in source_candidates:
        if col in df.columns:
            df[target] = df[col]
            return


def _compute_perceived_velocity(release_speed: pd.Series, extension: pd.Series) -> pd.Series:
    """Estimate perceived velocity using extension.

    Formula: velo * (60.5 / (60.5 - extension)) where distances are in feet.
    Falls back to the raw release speed when extension is missing.
    """

    release_speed = pd.to_numeric(release_speed, errors="coerce")
    extension = pd.to_numeric(extension, errors="coerce")

    perceived = release_speed * (60.5 / (60.5 - extension.replace(0, np.nan)))
    perceived = perceived.replace([np.inf, -np.inf], np.nan).fillna(release_speed)
    return perceived


def _compute_approach_angles(
    release_x: pd.Series,
    release_z: pd.Series,
    plate_x: pd.Series,
    plate_z: pd.Series,
    extension: Optional[pd.Series] = None,
) -> tuple[pd.Series, pd.Series]:
    """Approximate horizontal and vertical approach angles in degrees.

    Uses a simple geometry assumption from release point to plate over the remaining
    distance (60.5 ft minus extension when available).
    """

    release_x = pd.to_numeric(release_x, errors="coerce")
    release_z = pd.to_numeric(release_z, errors="coerce")
    plate_x = pd.to_numeric(plate_x, errors="coerce")
    plate_z = pd.to_numeric(plate_z, errors="coerce")

    if extension is not None:
        dist = 60.5 - pd.to_numeric(extension, errors="coerce").fillna(0.0)
    else:
        dist = pd.Series(60.5, index=release_x.index)

    delta_x = plate_x - release_x
    delta_z = plate_z - release_z

    horiz_angle = np.degrees(np.arctan2(delta_x, dist))
    vert_angle = np.degrees(np.arctan2(delta_z, dist))

    return horiz_angle, vert_angle


def get_pitcher_id(name: str) -> int:
    """
    Look up the MLBAM id for a pitcher using pybaseball's playerid_lookup.
    Assumes full name 'First Last'. Falls back to a hard-coded dict.
    """
    try:
        first, last = name.split(" ", 1)
        lookup = playerid_lookup(last, first)
        if not lookup.empty:
            return int(lookup.iloc[0]["key_mlbam"])
    except Exception:
        pass

    # Fallback mapping so things always work
    NAME_TO_ID: Dict[str, int] = {
        "Zack Wheeler": 554430,
    }
    return NAME_TO_ID.get(name, 554430)


@st.cache_data(show_spinner=False)
def load_statcast_data(pitcher_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load Statcast data for a pitcher between start_date and end_date,
    attach batter names, and normalize advanced tunneling fields.

    The loader backfills common Statcast variants (e.g., release_spin_rate,
    release_extension) and derives perceived velocity when possible so the
    dashboard's advanced charts have data even when source schemas vary.
    """
    pitcher_id = get_pitcher_id(pitcher_name)

    df = statcast_pitcher(start_date, end_date, pitcher_id)
    if df is None or df.empty:
        return pd.DataFrame()

    _ensure_column(df, "spin_rate", ["release_spin_rate"])
    _ensure_column(df, "extension", ["release_extension"])

    if "release_speed" in df.columns and "extension" in df.columns:
        df["perceived_velocity"] = _compute_perceived_velocity(
            df["release_speed"], df["extension"]
        )

    # Basic fields we care about
    advanced_keep_cols = [
        "spin_rate",
        "spin_axis",
        "perceived_velocity",
        "vertical_approach_angle",
        "horizontal_approach_angle",
        "extension",
        "release_diff_from_avg",
        "movement_diff_from_fastball",
        "movement_diff_from_fastball_horizontal",
        "movement_diff_from_fastball_vertical",
        "velocity_diff_from_fastball",
    ]

    keep_cols = [
        "game_date",
        "game_pk",
        "at_bat_number",
        "pitch_number",
        "pitch_type",
        "release_pos_x",
        "release_pos_z",
        "release_speed",
        "pfx_x",
        "pfx_z",
        "plate_x",
        "plate_z",
        "launch_speed",
        "events",
        "batter",
        "bat_score",
    ] + advanced_keep_cols

    existing_cols = [c for c in keep_cols if c in df.columns]
    df = df[existing_cols].copy()

    # Convert date to datetime for easier filtering/plotting
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])

    # Outcome-level columns
    if "launch_speed" in df.columns:
        df["outcome_velocity"] = df["launch_speed"]
    if "events" in df.columns:
        df["outcome_result"] = df["events"]

    # Map batter IDs to names
    if "batter" in df.columns:
        batter_ids = df["batter"].dropna().unique().tolist()
        if batter_ids:
            try:
                reverse = playerid_reverse_lookup(batter_ids, key_type="mlbam")
                reverse["batter_name"] = (
                    reverse["name_first"].str.strip() + " " + reverse["name_last"].str.strip()
                )
                reverse = reverse[["key_mlbam", "batter_name"]].drop_duplicates()
                df = df.merge(
                    reverse,
                    left_on="batter",
                    right_on="key_mlbam",
                    how="left",
                )
                df.drop(columns=["key_mlbam"], inplace=True)
            except Exception:
                df["batter_name"] = df["batter"].astype(str)
        else:
            df["batter_name"] = df["batter"].astype(str)

    return df

def compute_tunneling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add tunneling features for each pitch based on the previous pitch in the at-bat.
    """
    if df.empty:
        return df

    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).copy()

    # Align schema variations so downstream displays remain populated
    _ensure_column(df, "spin_rate", ["release_spin_rate"])
    _ensure_column(df, "extension", ["release_extension"])

    if "perceived_velocity" not in df.columns and {"release_speed", "extension"} <= set(df.columns):
        df["perceived_velocity"] = _compute_perceived_velocity(
            df["release_speed"], df["extension"]
        )

    if "vertical_approach_angle" not in df.columns or "horizontal_approach_angle" not in df.columns:
        if {"release_pos_x", "release_pos_z", "plate_x", "plate_z"} <= set(df.columns):
            h_angle, v_angle = _compute_approach_angles(
                df.get("release_pos_x"),
                df.get("release_pos_z"),
                df.get("plate_x"),
                df.get("plate_z"),
                df.get("extension") if "extension" in df.columns else None,
            )
            if "horizontal_approach_angle" not in df.columns:
                df["horizontal_approach_angle"] = h_angle
            if "vertical_approach_angle" not in df.columns:
                df["vertical_approach_angle"] = v_angle

    group_cols = ["game_pk", "at_bat_number"]
    cols_to_shift = [
        "pitch_type",
        "release_pos_x",
        "release_pos_z",
        "release_speed",
        "pfx_x",
        "pfx_z",
        "plate_x",
        "plate_z",
    ]
    for col in cols_to_shift:
        if col in df.columns:
            df[f"prev_{col}"] = df.groupby(group_cols)[col].shift(1)

    # Keep rows with a prior pitch context while allowing missing numeric data
    if "prev_pitch_type" in df.columns:
        df = df.dropna(subset=["prev_pitch_type"])

    def _safe(arr):
        return pd.to_numeric(arr, errors="coerce").fillna(0.0)

    # 1) Release consistency
    df["release_diff"] = np.sqrt(
        (_safe(df["release_pos_x"]) - _safe(df["prev_release_pos_x"])) ** 2
        + (_safe(df["release_pos_z"]) - _safe(df["prev_release_pos_z"])) ** 2
    )

    # 2) Trajectory consistency
    df["traj_diff"] = np.sqrt(
        (_safe(df["plate_x"]) - _safe(df["prev_plate_x"])) ** 2
        + (_safe(df["plate_z"]) - _safe(df["prev_plate_z"])) ** 2
    )

    # 3) Movement deception
    df["movement_diff"] = np.sqrt(
        (_safe(df["pfx_x"]) - _safe(df["prev_pfx_x"])) ** 2
        + (_safe(df["pfx_z"]) - _safe(df["prev_pfx_z"])) ** 2
    )

    # 4) Velocity separation
    if "release_speed" in df.columns and "prev_release_speed" in df.columns:
        df["velo_diff"] = (
            _safe(df["release_speed"]) - _safe(df["prev_release_speed"])
        ).abs()
    else:
        df["velo_diff"] = 0.0

    def _norm(series: pd.Series) -> pd.Series:
        s = series.astype(float)
        rng = s.max() - s.min()
        if rng == 0 or pd.isna(rng):
            return pd.Series(0.5, index=s.index)
        return (s - s.min()) / rng

    df["release_consistency"] = 1.0 - _norm(df["release_diff"])
    df["traj_consistency"] = 1.0 - _norm(df["traj_diff"])
    df["movement_deception"] = _norm(df["movement_diff"])
    df["velo_separation"] = _norm(df["velo_diff"])

    df["tunnel_score"] = (
        0.30 * df["release_consistency"]
        + 0.30 * df["traj_consistency"]
        + 0.20 * df["movement_deception"]
        + 0.20 * df["velo_separation"]
    )

    df["tunnel_score_norm"] = _norm(df["tunnel_score"])

    # Baseline-based feature gaps to populate dashboard visuals
    if {"pitch_type", "release_pos_x", "release_pos_z"} <= set(df.columns):
        release_avg = df.groupby("pitch_type")[
            ["release_pos_x", "release_pos_z"]
        ].transform("mean")
        df["release_diff_from_avg"] = np.sqrt(
            (pd.to_numeric(df["release_pos_x"], errors="coerce") - release_avg["release_pos_x"]) ** 2
            + (pd.to_numeric(df["release_pos_z"], errors="coerce") - release_avg["release_pos_z"]) ** 2
        )

    fastball_labels = {"FF", "FA", "SI", "FT"}
    if {"pitch_type", "pfx_x", "pfx_z"} <= set(df.columns):
        fb_mask = df["pitch_type"].isin(fastball_labels)
        if fb_mask.any():
            fb_means = {
                "pfx_x": pd.to_numeric(df.loc[fb_mask, "pfx_x"], errors="coerce").mean(),
                "pfx_z": pd.to_numeric(df.loc[fb_mask, "pfx_z"], errors="coerce").mean(),
                "release_speed": pd.to_numeric(df.loc[fb_mask, "release_speed"], errors="coerce").mean(),
            }

            df["movement_diff_from_fastball_horizontal"] = (
                pd.to_numeric(df["pfx_x"], errors="coerce") - fb_means["pfx_x"]
            )
            df["movement_diff_from_fastball_vertical"] = (
                pd.to_numeric(df["pfx_z"], errors="coerce") - fb_means["pfx_z"]
            )
            df["movement_diff_from_fastball"] = np.sqrt(
                df["movement_diff_from_fastball_horizontal"] ** 2
                + df["movement_diff_from_fastball_vertical"] ** 2
            )

            df["velocity_diff_from_fastball"] = (
                pd.to_numeric(df["release_speed"], errors="coerce") - fb_means["release_speed"]
            )

    return df


def summarize_games(df_tunnel: pd.DataFrame) -> pd.DataFrame:
    """
    Per-game summary: average tunnel score and runs allowed.
    Runs allowed ~= max(bat_score) - min(bat_score) while pitcher is on mound.
    """
    if df_tunnel.empty:
        return pd.DataFrame()

    df_g = df_tunnel.copy()
    df_g["game_date"] = pd.to_datetime(df_g["game_date"])

    agg_dict = {
        "avg_tunnel_score": ("tunnel_score_norm", "mean"),
        "num_pitches": ("pitch_type", "count"),
    }

    if "bat_score" in df_g.columns:
        agg_dict["runs_allowed"] = ("bat_score", lambda s: float(s.max() - s.min()))

    game_summary = (
        df_g.groupby(["game_pk", "game_date"])
        .agg(**agg_dict)
        .reset_index()
        .sort_values("game_date")
    )

    if "runs_allowed" not in game_summary.columns:
        game_summary["runs_allowed"] = np.nan

    return game_summary


# -------------------------------------------------
# Layout
# -------------------------------------------------
st.title("Tunnel Score")

# Sidebar controls
st.sidebar.header("Settings")

pitcher_name = st.sidebar.text_input(
    "Pitcher name",
    value="Zack Wheeler",
    help="Enter any MLB pitcher (First Last).",
).strip() or "Zack Wheeler"

season = st.sidebar.number_input(
    "Season", min_value=2015, max_value=2030, value=2024, step=1
)

# Date range picker
default_start = pd.to_datetime(f"{season}-03-01")
default_end = pd.to_datetime(f"{season}-11-30")

start_date, end_date = st.sidebar.date_input(
    "Date range",
    value=(default_start, default_end),
)

# Ensure proper order
if isinstance(start_date, list) or isinstance(start_date, tuple):
    start_date, end_date = start_date

if start_date > end_date:
    st.sidebar.error("Start date must be on or before end date.")
    st.stop()

# Number of pitches to show in the sample table
n_pitches = st.sidebar.slider(
    "Number of pitches to display",
    min_value=10,
    max_value=100,
    value=25,
    step=5,
)

show_raw = st.sidebar.checkbox("Show raw Statcast sample", value=False)
show_tunnel_sample = st.sidebar.checkbox(
    "Show tunneling feature sample", value=True
)
show_advanced_features = st.sidebar.checkbox(
    "Show advanced tunneling features", value=False
)

# Load data
with st.spinner(
    f"Pulling Statcast data for {pitcher_name} "
    f"from {start_date} to {end_date}..."
):
    df_raw = load_statcast_data(
        pitcher_name,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    )

if df_raw.empty:
    st.warning("No Statcast data returned for this pitcher / date range.")
    st.stop()

st.caption(
    f"Pulled Statcast data for **{pitcher_name}** from "
    f"**{start_date.strftime('%Y-%m-%d')}** to "
    f"**{end_date.strftime('%Y-%m-%d')}**."
)

# Optional: show raw Statcast sample
if show_raw:
    st.subheader("Raw Statcast Sample")
    st.dataframe(df_raw.head(n_pitches))

# Compute tunneling features
df_tunnel = compute_tunneling_features(df_raw)

if df_tunnel.empty:
    st.warning(
        "Not enough pitch sequences with previous-pitch context "
        "to compute tunneling features in this date range."
    )
else:
    # Tunneling feature sample
    if show_tunnel_sample:
        st.subheader("Tunneling Features (Sample)")

        desired_cols = [
            "game_date",
            "pitch_type",
            "tunnel_score_norm",
            "outcome_velocity",
            "outcome_result",
            "batter_name",
        ]
        advanced_cols = [
            "spin_rate",
            "spin_axis",
            "perceived_velocity",
            "vertical_approach_angle",
            "horizontal_approach_angle",
            "extension",
            "release_diff_from_avg",
            "movement_diff_from_fastball",
            "movement_diff_from_fastball_horizontal",
            "movement_diff_from_fastball_vertical",
            "velocity_diff_from_fastball",
        ]

        if show_advanced_features:
            desired_cols.extend(advanced_cols)

        existing_sample_cols = [c for c in desired_cols if c in df_tunnel.columns]

        if not existing_sample_cols:
            st.info(
                "Tunneling features computed, but requested display columns "
                "were not found."
            )
        else:
            st.dataframe(df_tunnel[existing_sample_cols].head(n_pitches))

# -------------------------------------------------
# Conceptual Summary
# -------------------------------------------------
st.markdown("## How This Tunnel Score Quantifies Pitch Tunneling")

st.markdown(
    """
This dashboard is built around a **Tunnel Score** – a way to measure how well a
pitcher disguises different pitch types within a sequence, eluding the batter to the best of the abilites.

**Pitch tunneling** is the idea that multiple pitch types can:

- Come out of almost the same release point, and
- Travel on nearly the same trajectory for as long as possible,

before separating late due to differences in spin and velocity. When a pitcher does this well,
the hitter has to commit to a swing before the pitches visibly separate, which is where
deception lives.
"""
)

st.markdown(
    """
### What goes into the Tunnel Score?

For each pitch, we compare it to the previous pitch in the at-bat and compute four components:

1. Release Consistency – How close are the two release points?
2. Trajectory Consistency – How similar are the locations as the ball
   travels toward the plate, how long is the tunnel duration?
3. Movement Deception – How different are the movement profiles once pitches leave the tunnel?
4. Velocity Separation – How different are the velocities off the same tunnel?

Each component is scaled to a 0–1 range and combined into a single **`tunnel_score_norm`**.
"""
)

# -------------------------------------------------
# Visualization: per-game avg tunnel score vs runs allowed
# -------------------------------------------------
st.markdown("## Per-Game Tunnel Score vs. Runs Allowed")

game_summary = summarize_games(df_tunnel)

if game_summary.empty:
    st.info(
        "Per-game summary could not be computed (no bat_score data for this range)."
    )
else:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Average Tunnel Score by Game Date**")
        line_chart_data = game_summary.set_index("game_date")[
            ["avg_tunnel_score"]
        ]
        st.line_chart(line_chart_data)

    with col2:
        st.markdown("**Tunnel Score vs. Runs Allowed (Regression View)**")

        base = alt.Chart(game_summary).encode(
            x=alt.X("avg_tunnel_score:Q", title="Average Tunnel Score (per game)"),
            y=alt.Y("runs_allowed:Q", title="Runs Allowed"),
            tooltip=[
                alt.Tooltip("game_date:T", title="Game Date"),
                alt.Tooltip("avg_tunnel_score:Q", title="Avg Tunnel Score"),
                alt.Tooltip("runs_allowed:Q", title="Runs Allowed"),
                alt.Tooltip("num_pitches:Q", title="Pitches"),
            ],
        )

        scatter = base.mark_circle(size=70)
        regression = base.transform_regression(
            "avg_tunnel_score", "runs_allowed"
        ).mark_line()

        st.altair_chart(scatter + regression, use_container_width=True)

    st.markdown("**Tunneling Features vs. Normalized Tunnel Score**")
    scatter_cols = st.columns(3)

    scatter_specs = [
        ("spin_rate", "Spin Rate"),
        ("extension", "Extension"),
        ("perceived_velocity", "Perceived Velocity"),
    ]

    for spec, container in zip(scatter_specs, scatter_cols):
        field, label = spec
        with container:
            if field in df_tunnel.columns:
                chart = (
                    alt.Chart(df_tunnel)
                    .mark_circle(size=60, opacity=0.7)
                    .encode(
                        x=alt.X(f"{field}:Q", title=label),
                        y=alt.Y("tunnel_score_norm:Q", title="Normalized Tunnel Score"),
                        tooltip=[
                            alt.Tooltip("game_date:T", title="Game Date"),
                            alt.Tooltip(f"{field}:Q", title=label),
                            alt.Tooltip("tunnel_score_norm:Q", title="Tunnel Score Norm"),
                        ],
                    )
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info(
                    f"{label} data not available in the current dataset to plot."
                )

    st.caption(
        "Each point is one game in the selected date range. "
        "The regression line shows the direction and strength of the relationship "
        "between tunneling quality and runs allowed."
    )
