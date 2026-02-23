import streamlit as st
import pandas as pd

import engine


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Tennis League Simulator", layout="wide")
st.title("Tennis League Simulator (WTT Match + Season Stats)")


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
# Initialize persistent season state
# ----------------------------
if "team_stats" not in st.session_state:
    st.session_state["team_stats"] = engine.init_team_stats(teams)

if "player_stats" not in st.session_state:
    st.session_state["player_stats"] = engine.init_player_stats()

if "match_history" not in st.session_state:
    st.session_state["match_history"] = []  # list of match summaries

# Optional: keep a last result for display
if "result" not in st.session_state:
    st.session_state["result"] = None


# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Match Settings")

team_a = st.sidebar.selectbox("Team A", teams, index=0)
team_b_choices = [t for t in teams if t != team_a]
team_b = st.sidebar.selectbox("Team B", team_b_choices, index=0)

use_seed = st.sidebar.checkbox("Use seed (repeatable results)", value=False)
seed_val = st.sidebar.number_input("Seed", min_value=0, max_value=999999, value=123, step=1)

target_games = st.sidebar.selectbox("Games to win each set", [5], index=0)

col_a, col_b = st.sidebar.columns(2)
run_button = col_a.button("Simulate Match")
reset_button = col_b.button("Reset Season")


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
# Run match (updates persistent stats)
# ----------------------------
if run_button:
    try:
        # IMPORTANT: pass existing team_stats and player_stats so they accumulate
        result = engine.simulate_match_wtt_with_stats(
            rosters_df=rosters,
            team_a=team_a,
            team_b=team_b,
            seed=int(seed_val) if use_seed else None,
            target_games=target_games,
            team_stats=st.session_state["team_stats"],
            player_stats=st.session_state["player_stats"],
        )

        # Save last result for UI
        st.session_state["result"] = result

        # Add to match history (compact)
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
        st.error("Match simulation failed")
        st.exception(e)


# ----------------------------
# Helper: Dataframes for exports + display
# ----------------------------
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


# ----------------------------
# Top: Standings + exports
# ----------------------------
st.subheader("Season Standings (Persistent)")

stand_df = standings_df_from_state()
st.dataframe(stand_df, use_container_width=True)

csv_stand = stand_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Standings CSV",
    data=csv_stand,
    file_name="standings.csv",
    mime="text/csv"
)


# ----------------------------
# Player stats + export
# ----------------------------
st.subheader("Season Player Stats (Persistent)")

ps_df = player_stats_df_from_state()
st.dataframe(ps_df, use_container_width=True)

csv_ps = ps_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Player Stats CSV",
    data=csv_ps,
    file_name="player_stats.csv",
    mime="text/csv"
)


# ----------------------------
# Match history
# ----------------------------
st.subheader("Match History")

mh_df = match_history_df()
if mh_df.empty:
    st.info("No matches yet. Simulate a match to start the season.")
else:
    st.dataframe(mh_df, use_container_width=True)

csv_mh = mh_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Match History CSV",
    data=csv_mh,
    file_name="match_history.csv",
    mime="text/csv",
    disabled=mh_df.empty
)


# ----------------------------
# Last match details (lineups + sets)
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

    st.write("### Lineups")
    lc1, lc2 = st.columns(2)
    with lc1:
        st.write(f"**{result['team_a']}**")
        st.json(result["lineup_a"])
    with lc2:
        st.write(f"**{result['team_b']}**")
        st.json(result["lineup_b"])


st.caption("Run: `python -m streamlit run app.py` from the tennis-ui folder with your venv activated.")
# app.py
# Streamlit UI for:
# - Season Mode (7 stops, 14 rounds, double round robin) played ONE MATCH at a time
# - After season ends, Playoffs (NBA-style play-in with top-2 byes) played ONE MATCH at a time
#
# Requires your engine.py to include:
# - load_rosters_csv
# - SeasonRunner
# - PlayoffRunner
#
# Run:
#   streamlit run app.py

import streamlit as st
import pandas as pd

from engine import (
    load_rosters_csv,
    SeasonRunner,
    PlayoffRunner,
)

st.set_page_config(page_title="Tennis League (WTT) — Season Mode", layout="wide")


# ----------------------------
# Helpers
# ----------------------------

def _safe_df(rows):
    if rows is None:
        return pd.DataFrame()
    if isinstance(rows, pd.DataFrame):
        return rows
    return pd.DataFrame(rows)

def render_match_result(result: dict):
    """Pretty-ish match result display using your engine result schema."""
    if not result:
        st.info("No result yet.")
        return

    team_a = result["team_a"]
    team_b = result["team_b"]
    total_a = result["total_a"]
    total_b = result["total_b"]

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader(f"{team_a} vs {team_b}")
    with col2:
        st.metric(label=team_a, value=total_a)
    with col3:
        st.metric(label=team_b, value=total_b)

    if result.get("decided_by_ot"):
        ot = result.get("ot") or {}
        st.warning(f"Decided by super tiebreak in {ot.get('set','?')}: "
                   f"{ot.get('tb_a','?')}–{ot.get('tb_b','?')} (Winner: {ot.get('winner','?')})")

    # Sets table
    sets = result.get("sets", [])
    if sets:
        st.markdown("**Sets (WTT order)**")
        st.dataframe(_safe_df(sets), use_container_width=True, hide_index=True)

    # Lineups
    la = result.get("lineup_a", {})
    lb = result.get("lineup_b", {})
    if la and lb:
        st.markdown("**Lineups**")
        left, right = st.columns(2)
        with left:
            st.caption(f"{team_a}")
            st.json(la)
        with right:
            st.caption(f"{team_b}")
            st.json(lb)

def standings_table_from_team_stats(team_stats: dict):
    """Render a compact standings table from your stored team_stats dict."""
    rows = []
    for team, s in team_stats.items():
        rows.append({
            "Team": team,
            "MP": s.get("MP", 0),
            "W": s.get("W", 0),
            "L": s.get("L", 0),
            "OTL": s.get("OTL", 0),
            "PTS": s.get("PTS", 0),
            "GF": s.get("GF", 0),
            "GA": s.get("GA", 0),
            "GD": s.get("GD", 0),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["PTS", "W", "GD", "GF"], ascending=[False, False, False, False]).reset_index(drop=True)
    return df

def stop_round_badge(stop: int, rnd: int):
    st.markdown(
        f"<div style='padding:8px 12px; display:inline-block; border-radius:12px; "
        f"background:#111827; color:#F9FAFB; font-weight:600;'>STOP {stop} · ROUND {rnd}</div>",
        unsafe_allow_html=True
    )


# ----------------------------
# Session state bootstrap
# ----------------------------

if "rosters_df" not in st.session_state:
    st.session_state.rosters_df = None

if "teams" not in st.session_state:
    st.session_state.teams = []

if "season" not in st.session_state:
    st.session_state.season = None

if "season_last_played" not in st.session_state:
    st.session_state.season_last_played = None

if "playoffs" not in st.session_state:
    st.session_state.playoffs = None

if "playoffs_last_played" not in st.session_state:
    st.session_state.playoffs_last_played = None

if "mode" not in st.session_state:
    st.session_state.mode = "season"  # or "playoffs"

if "target_games" not in st.session_state:
    st.session_state.target_games = 5

if "seed_base" not in st.session_state:
    st.session_state.seed_base = 123


# ----------------------------
# Sidebar controls
# ----------------------------

st.sidebar.title("League Controls")

rosters_path = st.sidebar.text_input("Rosters CSV path (relative to engine.py)", value="data/rosters.csv")
st.session_state.target_games = st.sidebar.selectbox("Games per set (target_games)", options=[5, 6], index=0)
carry_fatigue_to_playoffs = st.sidebar.checkbox("Carry fatigue into playoffs", value=True)

st.session_state.seed_base = st.sidebar.number_input(
    "Seed base (optional, for reproducible randomness)",
    min_value=0, max_value=10_000_000, value=int(st.session_state.seed_base), step=1
)

col_a, col_b = st.sidebar.columns(2)
with col_a:
    if st.button("Load / Reload Rosters", use_container_width=True):
        try:
            df = load_rosters_csv(rosters_path)
            st.session_state.rosters_df = df
            st.session_state.teams = sorted(df["team"].unique().tolist())
            st.success(f"Loaded rosters for {len(st.session_state.teams)} teams.")
        except Exception as e:
            st.session_state.rosters_df = None
            st.session_state.teams = []
            st.error(str(e))

with col_b:
    if st.button("Reset League", use_container_width=True):
        st.session_state.season = None
        st.session_state.playoffs = None
        st.session_state.mode = "season"
        st.session_state.season_last_played = None
        st.session_state.playoffs_last_played = None
        st.success("Reset complete.")

st.sidebar.divider()

if st.session_state.rosters_df is not None and st.session_state.teams:
    st.sidebar.caption("Teams")
    st.sidebar.write(", ".join(st.session_state.teams))
else:
    st.sidebar.info("Load rosters to begin.")


# ----------------------------
# Main header
# ----------------------------

st.title("🎾 Tennis League Simulator — Season Mode (WTT)")
st.caption("Play one match at a time through 7 stops, then run playoffs one match at a time (NBA-style play-in).")

if st.session_state.rosters_df is None or len(st.session_state.teams) != 8:
    st.warning("You need exactly 8 teams loaded from rosters.csv to use this mode.")
    st.stop()


# ----------------------------
# Initialize season runner if missing
# ----------------------------

if st.session_state.season is None:
    st.session_state.season = SeasonRunner(
        rosters_df=st.session_state.rosters_df,
        teams=st.session_state.teams,
        target_games=int(st.session_state.target_games),
    )
    st.session_state.mode = "season"
    st.session_state.season_last_played = None
    st.session_state.playoffs = None
    st.session_state.playoffs_last_played = None


# ----------------------------
# Top-level mode switch display
# ----------------------------

season = st.session_state.season
playoffs = st.session_state.playoffs

top_left, top_right = st.columns([2, 1])

with top_left:
    if st.session_state.mode == "season":
        st.subheader("Season")
    else:
        st.subheader("Playoffs")

with top_right:
    # show progress
    if st.session_state.mode == "season":
        total = len(season.schedule)
        done = season.cursor
        st.progress(done / total if total else 0)
        st.caption(f"Season progress: {done}/{total} matches")
    else:
        total = len(playoffs.order)
        done = playoffs.cursor
        st.progress(done / total if total else 0)
        st.caption(f"Playoffs progress: {done}/{total} games")


st.divider()


# ----------------------------
# Season UI
# ----------------------------

if st.session_state.mode == "season":
    left, right = st.columns([1.2, 1])

    with left:
        # Next match preview
        nxt = season.peek_next_match()
        if nxt:
            stop_round_badge(nxt["stop"], nxt["round"])
            st.markdown(f"### Next match: **{nxt['home']}** vs **{nxt['away']}**")
        else:
            st.success("Season complete!")

        btn_row = st.columns(2)
        with btn_row[0]:
            if st.button("▶️ Play Next Season Match", use_container_width=True, disabled=not season.has_next_match()):
                played = season.play_next_match(seed=int(st.session_state.seed_base + season.cursor))
                st.session_state.season_last_played = played
        with btn_row[1]:
            if st.button("⏩ Play Remaining Stop", use_container_width=True, disabled=not season.has_next_match()):
                # play until stop changes or season ends
                nxt0 = season.peek_next_match()
                if nxt0:
                    current_stop = nxt0["stop"]
                    while season.has_next_match():
                        nxtx = season.peek_next_match()
                        if nxtx and nxtx["stop"] != current_stop:
                            break
                        played = season.play_next_match(seed=int(st.session_state.seed_base + season.cursor))
                        st.session_state.season_last_played = played

        st.divider()

        # Last played match
        st.markdown("## Last Played Match")
        last = st.session_state.season_last_played
        if last:
            stop_round_badge(last["stop"], last["round"])
            render_match_result(last["result"])
        else:
            st.info("No matches played yet.")

    with right:
        st.markdown("## Standings")
        st.dataframe(standings_table_from_team_stats(season.team_stats), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("## Stop Snapshot")
        # show the remaining matches in the current stop
        nxt = season.peek_next_match()
        if nxt:
            current_stop = nxt["stop"]
            remaining = [m for m in season.schedule[season.cursor:] if m["stop"] == current_stop]
            st.caption(f"Remaining in STOP {current_stop}: {len(remaining)} matches")
            st.dataframe(_safe_df(remaining), use_container_width=True, hide_index=True)
        else:
            st.caption("No remaining stop matches — season complete.")

        st.divider()
        st.markdown("## Season Controls")
        if season.season_complete():
            st.success("Ready for playoffs.")
            if st.button("🏆 Start Playoffs", use_container_width=True):
                final_standings = season.standings()  # already sorted by your standings_from_team_stats
                st.session_state.playoffs = PlayoffRunner(
                    rosters_df=st.session_state.rosters_df,
                    standings_sorted=final_standings,
                    target_games=int(st.session_state.target_games),
                    carry_fatigue=bool(carry_fatigue_to_playoffs),
                    fatigue_by_team=(season.fatigue_by_team if carry_fatigue_to_playoffs else None),
                )
                st.session_state.mode = "playoffs"
                st.session_state.playoffs_last_played = None
        else:
            st.info("Play the full season before starting playoffs.")


# ----------------------------
# Playoffs UI
# ----------------------------

else:
    if playoffs is None:
        st.error("Playoffs runner not initialized. Go back to season and click 'Start Playoffs'.")
        st.stop()

    left, right = st.columns([1.2, 1])

    with left:
        # Next playoff game preview
        nxt = playoffs.peek_next_match()
        if nxt:
            st.markdown(f"### Next playoff game: **{nxt.team_a}** vs **{nxt.team_b}**")
            st.caption(f"Game: {nxt.game_id} · Round: {nxt.round_name}")
        else:
            if playoffs.champion:
                st.success(f"CHAMPION: {playoffs.champion}")
            else:
                st.info("Next playoff game will be determined after earlier results.")

        btn_row = st.columns(2)
        with btn_row[0]:
            if st.button("▶️ Play Next Playoff Game", use_container_width=True, disabled=not playoffs.has_next_match()):
                played = playoffs.play_next_match(seed=int(st.session_state.seed_base + 10_000 + playoffs.cursor))
                st.session_state.playoffs_last_played = played
        with btn_row[1]:
            if st.button("🏁 Play Rest of Playoffs", use_container_width=True, disabled=not playoffs.has_next_match()):
                while playoffs.has_next_match():
                    played = playoffs.play_next_match(seed=int(st.session_state.seed_base + 10_000 + playoffs.cursor))
                    st.session_state.playoffs_last_played = played

        st.divider()

        st.markdown("## Last Played Playoff Game")
        last = st.session_state.playoffs_last_played
        if last:
            st.caption(f"{last['game_id']} · {last['round']}")
            render_match_result(last["result"])
        else:
            st.info("No playoff games played yet.")

        st.divider()
        st.markdown("## Playoff Stop Constraint Monitor")
        st.caption("Because playoffs are treated as one long stop (until FINAL), stop_used accumulates and can bench stars.")
        # show top stop_used counts for teams that have played so far
        if hasattr(playoffs, "stop_used_by_team"):
            rows = []
            for team, mp in playoffs.stop_used_by_team.items():
                if mp:
                    top = sorted(mp.items(), key=lambda kv: kv[1], reverse=True)[:6]
                    for name, cnt in top:
                        rows.append({"Team": team, "Player": name, "stop_used": cnt})
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("No accumulated stop_used yet (play some playoff games).")

    with right:
        st.markdown("## Bracket State")
        rep = playoffs.report()
        st.caption("Eliminations")
        st.write("Seeds 7–8 eliminated:", ", ".join(rep.get("eliminated_pre_playoffs", [])) or "—")
        st.write("Play-in eliminated:", ", ".join(rep.get("eliminated_play_in", [])) or "—")

        st.divider()
        st.markdown("## Completed Games")
        games_df = _safe_df(rep.get("games", []))
        if not games_df.empty:
            # keep it compact
            show_cols = [c for c in ["game_id", "round", "team_a", "team_b", "winner"] if c in games_df.columns]
            st.dataframe(games_df[show_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No playoff games completed yet.")

        st.divider()
        st.markdown("## Champion")
        if rep.get("champion"):
            st.success(rep["champion"])
        else:
            st.info("TBD")