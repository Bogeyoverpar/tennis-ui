import streamlit as st
import pandas as pd

import engine


st.set_page_config(page_title="Tennis League Simulator", layout="wide")
st.title("Tennis League Simulator (WTT Match + Season Stats + Lineups)")


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
    st.session_state["match_history"] = []

if "result" not in st.session_state:
    st.session_state["result"] = None


# ----------------------------
# Helpers
# ----------------------------
def roster_names(team, gender):
    df = rosters[(rosters["team"] == team) & (rosters["gender"].str.upper() == gender)]
    return sorted(df["name"].dropna().astype(str).unique().tolist())


def standings_df():
    df = pd.DataFrame(st.session_state["team_stats"]).T.reset_index().rename(columns={"index": "Team"})
    df["Record"] = df.apply(lambda r: f"{int(r['W'])}-{int(r['L'])}-{int(r['OTL'])}", axis=1)
    df = df.sort_values(["PTS", "GD", "GF"], ascending=False)
    return df[["Team", "MP", "Record", "PTS", "GF", "GA", "GD"]]


def player_stats_df():
    df = pd.DataFrame(list(st.session_state["player_stats"].values()))
    if not df.empty:
        df = df.sort_values(["GamesWon", "SetsPlayed"], ascending=False)
    return df


# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Match Settings")

team_a = st.sidebar.selectbox("Team A", teams, index=0)
team_b = st.sidebar.selectbox("Team B", [t for t in teams if t != team_a], index=0)

mode = st.sidebar.radio("Lineup Mode", ["Auto", "Manual"], index=0)

run_button = st.sidebar.button("Simulate Match")
reset_button = st.sidebar.button("Reset Season")


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

    cA, cB = st.columns(2)

    with cA:
        st.write(f"### {team_a}")
        manual_lineup_a = {
            "MS": st.selectbox("MS", a_men),
            "WS": st.selectbox("WS", a_women),
            "MD": (
                st.selectbox("MD Player 1", a_men),
                st.selectbox("MD Player 2", a_men, index=1 if len(a_men) > 1 else 0),
            ),
            "WD": (
                st.selectbox("WD Player 1", a_women),
                st.selectbox("WD Player 2", a_women, index=1 if len(a_women) > 1 else 0),
            ),
            "XD": (
                st.selectbox("XD Man", a_men),
                st.selectbox("XD Woman", a_women),
            ),
        }

    with cB:
        st.write(f"### {team_b}")
        manual_lineup_b = {
            "MS": st.selectbox("MS", b_men),
            "WS": st.selectbox("WS", b_women),
            "MD": (
                st.selectbox("MD Player 1", b_men),
                st.selectbox("MD Player 2", b_men, index=1 if len(b_men) > 1 else 0),
            ),
            "WD": (
                st.selectbox("WD Player 1", b_women),
                st.selectbox("WD Player 2", b_women, index=1 if len(b_women) > 1 else 0),
            ),
            "XD": (
                st.selectbox("XD Man", b_men),
                st.selectbox("XD Woman", b_women),
            ),
        }


# ----------------------------
# Run match
# ----------------------------
if run_button:
    try:
        result = engine.simulate_match_wtt_with_stats(
            rosters_df=rosters,
            team_a=team_a,
            team_b=team_b,
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
        })

    except Exception as e:
        st.error("Match simulation failed.")
        st.exception(e)


# ----------------------------
# Standings
# ----------------------------
st.subheader("Season Standings")
st.dataframe(standings_df(), use_container_width=True)

st.subheader("Player Stats")
st.dataframe(player_stats_df(), use_container_width=True)

st.subheader("Match History")
st.dataframe(pd.DataFrame(st.session_state["match_history"]), use_container_width=True)

# ----------------------------
# Last match
# ----------------------------
st.subheader("Last Match Details")

result = st.session_state["result"]

if result:
    st.metric(result["team_a"], result["total_a"])
    st.metric(result["team_b"], result["total_b"])

    if result["decided_by_ot"]:
        st.warning("Overtime: Super Tiebreak")
        st.json(result["ot"])

    st.write("### Sets")
    st.dataframe(pd.DataFrame(result["sets"]), use_container_width=True)

    st.write("### Lineups")
    c1, c2 = st.columns(2)
    with c1:
        st.json(result["lineup_a"])
    with c2:
        st.json(result["lineup_b"])