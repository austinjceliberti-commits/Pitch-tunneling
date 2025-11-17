import math
import itertools
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
from pybaseball import statcast_pitcher, playerid_lookup
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# Streamlit page config
# -------------------------------------------------
st.set_page_config(
    page_title="Pitch Tunneling Dashboard",
    page_icon="",
    layout="wide",
)

st.title(" Pitch Recommendation Engine Based on Tunnel Score")


# -------------------------------------------------
# Utility: get MLBAM ID from pitcher name
# -------------------------------------------------
def get_mlbam_id(pitcher_name: str) -> int:
    """
    Look up MLBAM ID for a pitcher using pybaseball's playerid_lookup.
    Assumes 'First Last' format.
    """
    try:
        first, last = pitcher_name.strip().split(" ", 1)
    except ValueError:
        raise ValueError("Pitcher name must be in 'First Last' format.")

    id_df = playerid_lookup(last, first)
    if id_df.empty:
        raise ValueError(f"No MLBAM id found for pitcher: {pitcher_name}")

    return int(id_df.iloc[0]["key_mlbam"])


# -------------------------------------------------
# Data pipeline: pull Statcast data from Baseball Savant
# -------------------------------------------------
@st.cache_data(show_spinner=True)
def load_statcast_for_pitcher(
    pitcher_name: str,
    season: int = 2024,
) -> pd.DataFrame:
    """
    Pull Statcast data for a single pitcher from Baseball Savant using pybaseball.
    This replaces the old 'read local CSV' behavior.
    """
    mlbam_id = get_mlbam_id(pitcher_name)
    # Rough season bounds; adjust if needed
    start_date = f"{season}-03-01"
    end_date = f"{season}-11-30"

    st.info(f"Pulling Statcast data for {pitcher_name} ({mlbam_id}) from {start_date} to {end_date}…")
    df = statcast_pitcher(start_date, end_date, mlbam_id)

    # Basic safety check
    if df is None or df.empty:
        raise ValueError(f"No Statcast data returned for {pitcher_name} in {season}.")

    # Ensure consistent dtypes
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"])

    return df


# -------------------------------------------------
# Feature engineering: tunneling features + tunnel_score
# -------------------------------------------------
def minmax_norm(series: pd.Series) -> pd.Series:
    """Safe min-max normalization (returns 0 if constant)."""
    s = series.astype(float)
    s_min, s_max = s.min(), s.max()
    if pd.isna(s_min) or pd.isna(s_max) or s_min == s_max:
        return pd.Series(0.0, index=s.index)
    return (s - s_min) / (s_max - s_min)


def engineer_tunneling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given raw Statcast data for a single pitcher, add tunneling-related
    features and a composite tunnel_score.

    This is where you can iterate on the definition of tunneling:
      - Release consistency
      - Early trajectory overlap
      - Late movement separation
      - Velocity separation
    """
    # Work on a copy to avoid mutating cached data
    df = df.copy()

    # Sort properly
    df = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"])

    # Group by game & at-bat to keep "previous pitch" logic clean
    group_cols = ["game_pk", "at_bat_number"]

    # Columns we will compare to the previous pitch
    cols_to_shift = [
        "release_pos_x",
        "release_pos_z",
        "release_speed",
        "pfx_x",
        "pfx_z",
        "plate_x",
        "plate_z",
    ]

    for col in cols_to_shift:
        df[f"prev_{col}"] = df.groupby(group_cols)[col].shift(1)

    # Drop rows where we don't have a previous pitch context
    df = df.dropna(subset=[f"prev_{c}" for c in cols_to_shift])

    # 1) Release consistency: how similar is the release point to previous pitch?
    df["release_diff"] = np.sqrt(
        (df["release_pos_x"] - df["prev_release_pos_x"]) ** 2
        + (df["release_pos_z"] - df["prev_release_pos_z"]) ** 2
    )

    # 2) Early trajectory overlap approximated by plate_x/z similarity
    #    (proxy for tunnel duration; you can later upgrade to full trajectory)
    df["traj_diff"] = np.sqrt(
        (df["plate_x"] - df["prev_plate_x"]) ** 2
        + (df["plate_z"] - df["prev_plate_z"]) ** 2
    )

    # 3) Movement deception: how much do the movement profiles differ?
    df["movement_diff"] = np.sqrt(
        (df["pfx_x"] - df["prev_pfx_x"]) ** 2
        + (df["pfx_z"] - df["prev_pfx_z"]) ** 2
    )

    # 4) Velocity separation: how different are the velocities?
    df["velo_diff"] = (df["release_speed"] - df["prev_release_speed"]).abs()

    # Normalize components so they’re comparable in scale
    df["release_diff_norm"] = minmax_norm(df["release_diff"])
    df["traj_diff_norm"] = minmax_norm(df["traj_diff"])
    df["movement_diff_norm"] = minmax_norm(df["movement_diff"])
    df["velo_diff_norm"] = minmax_norm(df["velo_diff"])

    # -------------------------------------------------
    # Composite tunnel_score:
    #   - Lower release_diff is GOOD (harder for hitter to see difference early)
    #   - Higher movement_diff and velo_diff are GOOD (late separation)
    #   - You can choose whether higher or lower traj_diff is "good" for your theory.
    #
    # Here:
    #   tunnel_score = (1 - release_diff_norm) + movement_diff_norm + velo_diff_norm
    # You can add traj_diff_norm if you want to reward different end-locations.
    # -------------------------------------------------
    df["tunnel_score_component_release"] = 1.0 - df["release_diff_norm"]
    df["tunnel_score_component_movement"] = df["movement_diff_norm"]
    df["tunnel_score_component_velo"] = df["velo_diff_norm"]

    df["tunnel_score"] = (
        df["tunnel_score_component_release"]
        + df["tunnel_score_component_movement"]
        + df["tunnel_score_component_velo"]
    )

    # Optional final normalization of tunnel_score
    df["tunnel_score_norm"] = minmax_norm(df["tunnel_score"])

    return df


# -------------------------------------------------
# Sidebar controls
# -------------------------------------------------
with st.sidebar:
    st.header(" Settings")

    # You can replace this with your existing list/dict of pitchers
    default_pitchers = [
        "Zack Wheeler",
        "Aaron Nola",
        "Corbin Burnes",
        "Tarik Skubal",
        "Pablo López",
    ]
    pitcher_name = st.selectbox("Select Pitcher", options=default_pitchers)

    season = st.number_input("Season", min_value=2015, max_value=2025, value=2024, step=1)

    show_raw = st.checkbox("Show raw Statcast sample", value=False)
    show_tunnel_sample = st.checkbox("Show tunneling feature sample", value=True)


# -------------------------------------------------
# Main data & feature pipeline
# -------------------------------------------------
try:
    df_raw = load_statcast_for_pitcher(pitcher_name, season=season)
except Exception as e:
    st.error(f"Error loading Statcast data: {e}")
    st.stop()

if show_raw:
    st.subheader(" Raw Statcast Data (Sample)")
    st.dataframe(df_raw.head(25))

# Engineer tunneling features
df_tunnel = engineer_tunneling_features(df_raw)

if df_tunnel.empty:
    st.warning("No valid sequences with previous-pitch context were found for this pitcher/season.")
    st.stop()

if show_tunnel_sample:
    st.subheader("🧪 Tunneling Features (Sample)")
    st.dataframe(
        df_tunnel[
            [
                "game_date",
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
                "tunnel_score_norm",
            ]
        ].head(25)
    )

# -------------------------------------------------
#  Your existing model / recommendation logic
# -------------------------------------------------
# At this point, df_tunnel contains EVERYTHING you used to have from the CSVs,
# plus the new tunneling features:
#   - release_diff, traj_diff, movement_diff, velo_diff
#   - normalized versions
#   - tunnel_score, tunnel_score_norm
#
# You can keep your RandomForestClassifier (or any other model) exactly the same,
# just swap its input DataFrame from the old CSV-based one to df_tunnel.
#
# Below is an example stub showing how you might build a simple classifier
# using the new tunneling features. Replace this with your current logic.


st.subheader("🔮 Example: Simple Next-Pitch Classifier Using Tunneling Features")
st.caption(
    "Replace this section with your existing model code. "
    "Just make sure you use df_tunnel as your feature source."
)

# Example target: did the batter make weak contact or whiff?
# You likely already have your own target definition; this is just a placeholder
df_model = df_tunnel.copy()

# Example binary outcome: weak contact / strikeout vs everything else
# (You should plug in your own label logic here)
weak_events = ["strikeout", "swinging_strike", "foul_tip"]
if "des" in df_model.columns:
    df_model["target_good_outcome"] = df_model["des"].isin(weak_events).astype(int)
else:
    # Fallback dummy label so the code is runnable; replace with your own
    df_model["target_good_outcome"] = (df_model["tunnel_score_norm"] > 0.5).astype(int)

feature_cols = [
    "tunnel_score_norm",
    "release_diff_norm",
    "movement_diff_norm",
    "velo_diff_norm",
]

X = df_model[feature_cols]
y = df_model["target_good_outcome"]

# Guard against tiny datasets
if X.shape[0] < 200 or y.nunique() < 2:
    st.warning("⚠️ Not enough data or label variety to train an example model for this pitcher.")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Accuracy", f"{train_score:.3f}")
    with col2:
        st.metric("Test Accuracy", f"{test_score:.3f}")

    # Show feature importances to sanity-check tunneling impact
    importances = pd.DataFrame(
        {"feature": feature_cols, "importance": clf.feature_importances_}
    ).sort_values("importance", ascending=False)

    st.subheader("📊 Feature Importances (Example Model)")
    st.dataframe(importances)

    # You can then extend this to your full "Top 3 Recommended Pitches" logic
    # by:
    #   - Grouping df_tunnel by (prev_pitch_type, count, etc.)
    #   - Aggregating tunnel_score_norm and model-predicted success probabilities
    #   - Ranking pitch_type options
