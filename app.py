import streamlit as st
import pandas as pd

import engine


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Tennis League Simulator", layout="wide")
st.title("Tennis League Simulator (WTT Match + Season Mode + Lineups)")


# ----------------------------
# Load rosters
# ----------------------------
try:
    rosters = engine.load_rosters_csv("data/rosters.csv")
except Exception as e:
    st.error("Could not load data/rosters.csv")
    st.exception(e)
    st.stop()

teams = sorted(rosters["team"].dropna().unique().tolist())
if len(teams) < 2:
    st.error("Need at least 2 teams in rosters.csv")
    st.stop()


# ----------------------------
# Persistent season state
# ----------------------------
if "team_stats" not in st.session_state:
    st.session_state["team_stats"] = engine.init_team_stats(teams)

if "player_stats" not in st.session_state:
    st.session_state["player_stats"] = engine.init_player_stats()

if "match_history" not in st.session_state:
    st.session_state["match_history"] = []  # list of dicts (log)

if "result" not in st.session_state:
    st.session_state["result"] = None


# ----------------------------
# Helpers
# ----------------------------
def roster_names(team, gender):
    df = rosters[(rosters["team"] == team) & (rosters["gender"].str.upper() == gender)].copy()
    return sorted(df["name"].dropna().astype(str).str.strip().unique().tolist())


def standings_df_from_state():
    df = pd.DataFrame(st.session_state["team_stats"]).T.reset_index().rename(columns={"index": "Team"})
    df["Record"] = df.apply(lambda r: f"{int(r['W'])}-{int(r['L'])}-{int(r['OTL'])}", axis=1)
    df = df.sort_values(["PTS", "GD", "GF"], ascending=False)
    df = df[["Team", "MP", "Record", "PTS", "GF", "GA", "GD", "W", "L", "OTL"]]
    return df


def player_stats_df_from_state():
    df = pd.DataFrame(list(st.session_state["player_stats"].values()))
    if not df.empty:
        df = df.sort_values(["GamesWon", "SetsPlayed"], ascending=False)
    return df


def match_history_df():
    return pd.DataFrame(st.session_state["match_history"])


def safe_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Season Controls")

team_a = st.sidebar.selectbox("Team A", teams, index=0)
team_b = st.sidebar.selectbox("Team B", [t for t in teams if t != team_a], index=0)

mode = st.sidebar.radio("Lineup Mode", ["Auto", "Manual"], index=0)
target_games = st.sidebar.selectbox("Games to win each set", [5], index=0)

colA, colB = st.sidebar.columns(2)
run_button = colA.button("Simulate Match")
reset_button = colB.button("Reset Season")

st.sidebar.markdown("---")
st.sidebar.caption("No seed: every match/season is unique.")


# ----------------------------
# Reset season
# ----------------------------
if reset_button:
    st.session_state["team_stats"] = engine.init_team_stats(teams)
    st.session_state["player_stats"] = engine.init_player_stats()
    st.session_state["match_history"] = []
    st.session_state["result"] = None
    st.success("Season reset.")


# ----------------------------
# Manual lineup UI
# ----------------------------
manual_lineup_a = None
manual_lineup_b = None

if mode == "Manual":
    st.subheader("Manual Lineups")

    a_men = roster_names(team_a, "M")
    a_women = roster_names(team_a, "F")
    b_men = roster_names(team_b, "M")
    b_women = roster_names(team_b, "F")

    if len(a_men) < 3 or len(a_women) < 3 or len(b_men) < 3 or len(b_women) < 3:
        st.error("Each team must have 3 men and 3 women in rosters.csv.")
        st.stop()

    cA, cB = st.columns(2)

    with cA:
        st.write(f"### {team_a}")

        ms_a = st.selectbox("MS (Men Singles)", a_men, key="ms_a")
        ws_a = st.selectbox("WS (Women Singles)", a_women, key="ws_a")

        md_a_1 = st.selectbox("MD Player 1", a_men, key="md_a_1")
        md_a_2 = st.selectbox("MD Player 2", a_men, index=1 if len(a_men) > 1 else 0, key="md_a_2")

        wd_a_1 = st.selectbox("WD Player 1", a_women, key="wd_a_1")
        wd_a_2 = st.selectbox("WD Player 2", a_women, index=1 if len(a_women) > 1 else 0, key="wd_a_2")

        xd_a_m = st.selectbox("XD Man", a_men, key="xd_a_m")
        xd_a_w = st.selectbox("XD Woman", a_women, key="xd_a_w")

        manual_lineup_a = {
            "MS": ms_a,
            "WS": ws_a,
            "MD": (md_a_1, md_a_2),
            "WD": (wd_a_1, wd_a_2),
            "XD": (xd_a_m, xd_a_w),
        }

    with cB:
        st.write(f"### {team_b}")

        ms_b = st.selectbox("MS (Men Singles)", b_men, key="ms_b")
        ws_b = st.selectbox("WS (Women Singles)", b_women, key="ws_b")

        md_b_1 = st.selectbox("MD Player 1", b_men, key="md_b_1")
        md_b_2 = st.selectbox("MD Player 2", b_men, index=1 if len(b_men) > 1 else 0, key="md_b_2")

        wd_b_1 = st.selectbox("WD Player 1", b_women, key="wd_b_1")
        wd_b_2 = st.selectbox("WD Player 2", b_women, index=1 if len(b_women) > 1 else 0, key="wd_b_2")

        xd_b_m = st.selectbox("XD Man", b_men, key="xd_b_m")
        xd_b_w = st.selectbox("XD Woman", b_women, key="xd_b_w")

        manual_lineup_b = {
            "MS": ms_b,
            "WS": ws_b,
            "MD": (md_b_1, md_b_2),
            "WD": (wd_b_1, wd_b_2),
            "XD": (xd_b_m, xd_b_w),
        }

    st.caption("Lineup rules enforced in engine: gender correctness + max 2 sets per player + no duplicate doubles partner.")


# ----------------------------
# Run match (updates persistent stats)
# ----------------------------
if run_button:
    try:
        result = engine.simulate_match_wtt_with_stats(
            rosters_df=rosters,
            team_a=team_a,
            team_b=team_b,
            target_games=target_games,
            team_stats=st.session_state["team_stats"],
            player_stats=st.session_state["player_stats"],
            manual_lineup_a=manual_lineup_a if mode == "Manual" else None,
            manual_lineup_b=manual_lineup_b if mode == "Manual" else None,
        )

        st.session_state["result"] = result

        winner = result["team_a"] if result["total_a"] > result["total_b"] else result["team_b"]
        st.session_state["match_history"].append({
            "TeamA": result["team_a"],
            "A": result["total_a"],
            "TeamB": result["team_b"],
            "B": result["total_b"],
            "Winner": winner,
            "OT": result["decided_by_ot"],
            "Mode": mode,
        })

        st.success("Match simulated and season stats updated.")

    except Exception as e:
        st.error("Match simulation failed (often a lineup rule issue if Manual).")
        st.exception(e)


# ----------------------------
# SEASON DASHBOARD
# ----------------------------
st.subheader("Season Standings (Persistent)")

stand_df = standings_df_from_state()
st.dataframe(stand_df, use_container_width=True)

st.download_button(
    "Download Standings CSV",
    data=safe_csv_bytes(stand_df),
    file_name="standings.csv",
    mime="text/csv",
)


st.subheader("Season Player Stats (Persistent)")

ps_df = player_stats_df_from_state()
st.dataframe(ps_df, use_container_width=True)

st.download_button(
    "Download Player Stats CSV",
    data=safe_csv_bytes(ps_df),
    file_name="player_stats.csv",
    mime="text/csv",
)


st.subheader("Match History")

mh_df = match_history_df()
if mh_df.empty:
    st.info("No matches yet — simulate a match to start the season.")
else:
    st.dataframe(mh_df, use_container_width=True)
    st.download_button(
        "Download Match History CSV",
        data=safe_csv_bytes(mh_df),
        file_name="match_history.csv",
        mime="text/csv",
    )


# ----------------------------
# LAST MATCH DETAILS
# ----------------------------
st.subheader("Last Match Details")

result = st.session_state["result"]
if result is None:
    st.info("Simulate a match to see details here.")
else:
    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        st.metric(result["team_a"], result["total_a"])
    with c2:
        st.metric(result["team_b"], result["total_b"])
    with c3:
        winner = result["team_a"] if result["total_a"] > result["total_b"] else result["team_b"]
        st.write(f"Winner: **{winner}**")
        if result["decided_by_ot"]:
            st.warning("Overtime: Super Tiebreak (last set players) +1 game to winner")
            st.json(result["ot"])

    st.write("### Set-by-set")
    st.dataframe(pd.DataFrame(result["sets"]), use_container_width=True)

    st.write("### Lineups Used")
    lc1, lc2 = st.columns(2)
    with lc1:
        st.write(f"**{result['team_a']}**")
        st.json(result["lineup_a"])
    with lc2:
        st.write(f"**{result['team_b']}**")
        st.json(result["lineup_b"])


st.caption("Run: `python -m streamlit run app.py` (from tennis-ui with venv activated).")