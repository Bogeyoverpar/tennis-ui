# app.py
import pandas as pd
import streamlit as st

import engine
from engine import PlayoffRunner, SeasonRunner, load_rosters_csv

st.set_page_config(page_title="Tennis League (WTT)", layout="wide")


def _safe_df(rows):
    if rows is None:
        return pd.DataFrame()
    if isinstance(rows, pd.DataFrame):
        return rows
    return pd.DataFrame(rows)


def render_match_result(result: dict):
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
        st.warning(
            f"Decided by super tiebreak in {ot.get('set','?')}: "
            f"{ot.get('tb_a','?')}–{ot.get('tb_b','?')} (Winner: {ot.get('winner','?')})"
        )

    sets = result.get("sets", [])
    if sets:
        st.markdown("**Sets (WTT order)**")
        st.dataframe(_safe_df(sets), use_container_width=True, hide_index=True)

    la = result.get("lineup_a", {})
    lb = result.get("lineup_b", {})
    if la and lb:
        st.markdown("**Lineups**")
        left, right = st.columns(2)
        with left:
            st.caption(team_a)
            st.json(la)
        with right:
            st.caption(team_b)
            st.json(lb)


def standings_table_from_team_stats(team_stats: dict):
    rows = []
    for team, s in team_stats.items():
        rows.append(
            {
                "Team": team,
                "MP": s.get("MP", 0),
                "W": s.get("W", 0),
                "L": s.get("L", 0),
                "OTL": s.get("OTL", 0),
                "PTS": s.get("PTS", 0),
                "GF": s.get("GF", 0),
                "GA": s.get("GA", 0),
                "GD": s.get("GD", 0),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["PTS", "W", "GD", "GF"], ascending=[False, False, False, False]).reset_index(drop=True)
    return df


def stop_round_badge(stop: int, rnd: int):
    st.markdown(
        f"<div style='padding:8px 12px; display:inline-block; border-radius:12px; "
        f"background:#111827; color:#F9FAFB; font-weight:600;'>STOP {stop} · ROUND {rnd}</div>",
        unsafe_allow_html=True,
    )


def render_playoff_bracket(playoffs: PlayoffRunner):
    """Render a left-to-right playoff bracket with seeded teams."""
    seed_lookup = {row["team"]: i + 1 for i, row in enumerate(playoffs.standings_sorted)}

    def fmt_team(team: str) -> str:
        if not team or team == "TBD":
            return "TBD"
        seed = seed_lookup.get(team)
        return f"#{seed} {team}" if seed else team

    def matchup_label(game_id: str, default_a: str = "TBD", default_b: str = "TBD") -> str:
        g = playoffs.games.get(game_id)
        if not g:
            return f"{default_a} vs {default_b}"
        label = f"{fmt_team(g.team_a)}\nvs\n{fmt_team(g.team_b)}"
        if g.winner:
            label += f"\n🏅 {fmt_team(g.winner)}"
        return label

    st.markdown("## Playoff Bracket")
    c1, c2, c3, c4 = st.columns([1.5, 1.2, 1.1, 1.0])

    with c1:
        st.markdown("**Play-In**")
        st.text(matchup_label("PLAYIN_A"))
        st.text("↓")
        st.text(matchup_label("PLAYIN_B"))
        st.text("↓")
        st.text(matchup_label("PLAYIN_C"))

    with c2:
        st.markdown("**Semifinals**")
        st.text(matchup_label("SEMI_1"))
        st.text("↓")
        st.text(matchup_label("SEMI_2"))

    with c3:
        st.markdown("**Final**")
        st.text(matchup_label("FINAL"))

    with c4:
        st.markdown("**Champion**")
        if playoffs.champion:
            st.success(fmt_team(playoffs.champion))
        else:
            st.info("TBD")


def bootstrap_state():
    defaults = {
        "rosters_df": None,
        "teams": [],
        "season": None,
        "season_last_played": None,
        "playoffs": None,
        "playoffs_last_played": None,
        "mode": "season",
        "target_games": 5,
        "app_mode": "Season + Playoffs",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_classic_mode():
    st.title("Tennis League Simulator (Classic Match Mode)")
    rosters = st.session_state.rosters_df
    if rosters is None:
        st.warning("Load rosters from the sidebar to use classic mode.")
        return

    teams = sorted(rosters["team"].dropna().unique().tolist())
    if len(teams) < 2:
        st.error("Need at least 2 teams in rosters.csv")
        return

    if "classic_team_stats" not in st.session_state:
        st.session_state.classic_team_stats = engine.init_team_stats(teams)
    if "classic_player_stats" not in st.session_state:
        st.session_state.classic_player_stats = engine.init_player_stats()
    if "classic_match_history" not in st.session_state:
        st.session_state.classic_match_history = []
    if "classic_result" not in st.session_state:
        st.session_state.classic_result = None

    st.sidebar.subheader("Classic Match Settings")
    team_a = st.sidebar.selectbox("Team A", teams, key="classic_team_a")
    team_b = st.sidebar.selectbox("Team B", [t for t in teams if t != team_a], key="classic_team_b")
    ca, cb = st.sidebar.columns(2)
    run_button = ca.button("Simulate Match", key="classic_run")
    reset_button = cb.button("Reset Season", key="classic_reset")

    if reset_button:
        st.session_state.classic_team_stats = engine.init_team_stats(teams)
        st.session_state.classic_player_stats = engine.init_player_stats()
        st.session_state.classic_match_history = []
        st.session_state.classic_result = None
        st.success("Classic mode season reset.")

    if run_button:
        result = engine.simulate_match_wtt_with_stats(
            rosters_df=rosters,
            team_a=team_a,
            team_b=team_b,
            target_games=int(st.session_state.target_games),
            team_stats=st.session_state.classic_team_stats,
            player_stats=st.session_state.classic_player_stats,
        )
        st.session_state.classic_result = result
        winner = result["team_a"] if result["total_a"] > result["total_b"] else result["team_b"]
        st.session_state.classic_match_history.append(
            {
                "TeamA": result["team_a"],
                "A": result["total_a"],
                "TeamB": result["team_b"],
                "B": result["total_b"],
                "Winner": winner,
                "OT": result["decided_by_ot"],
            }
        )

    stand_df = pd.DataFrame(st.session_state.classic_team_stats).T.reset_index().rename(columns={"index": "Team"})
    stand_df["Record"] = stand_df.apply(lambda r: f"{int(r['W'])}-{int(r['L'])}-{int(r['OTL'])}", axis=1)
    stand_df = stand_df.sort_values(["PTS", "GD", "GF"], ascending=False)
    stand_df = stand_df[["Team", "MP", "Record", "PTS", "GF", "GA", "GD", "W", "L", "OTL"]]

    st.subheader("Season Standings (Classic)")
    st.dataframe(stand_df, use_container_width=True)

    ps_df = pd.DataFrame(list(st.session_state.classic_player_stats.values()))
    if not ps_df.empty:
        ps_df = ps_df.sort_values(["GamesWon", "SetsPlayed"], ascending=False)
    st.subheader("Season Player Stats (Classic)")
    st.dataframe(ps_df, use_container_width=True)

    mh_df = pd.DataFrame(st.session_state.classic_match_history)
    st.subheader("Match History (Classic)")
    if mh_df.empty:
        st.info("No matches yet.")
    else:
        st.dataframe(mh_df, use_container_width=True)

    st.subheader("Last Match Details")
    if st.session_state.classic_result is None:
        st.info("Simulate a match to see details here.")
    else:
        render_match_result(st.session_state.classic_result)


def render_season_playoffs_mode(carry_fatigue_to_playoffs: bool):
    st.title("🎾 Tennis League Simulator — Season Mode (WTT)")
    st.caption("Play one match at a time through 7 stops, then run playoffs one match at a time.")

    if st.session_state.rosters_df is None or len(st.session_state.teams) != 8:
        st.warning("You need exactly 8 teams loaded from rosters.csv to use this mode.")
        return

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

    season = st.session_state.season
    playoffs = st.session_state.playoffs

    top_left, top_right = st.columns([2, 1])
    with top_left:
        st.subheader("Season" if st.session_state.mode == "season" else "Playoffs")
    with top_right:
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

    if st.session_state.mode == "season":
        left, right = st.columns([1.2, 1])
        with left:
            nxt = season.peek_next_match()
            if nxt:
                stop_round_badge(nxt["stop"], nxt["round"])
                st.markdown(f"### Next match: **{nxt['home']}** vs **{nxt['away']}**")
            else:
                st.success("Season complete!")

            btn_row = st.columns(2)
            with btn_row[0]:
                if st.button("▶️ Play Next Season Match", use_container_width=True, disabled=not season.has_next_match()):
                    played = season.play_next_match()
                    st.session_state.season_last_played = played
            with btn_row[1]:
                if st.button("⏩ Play Remaining Stop", use_container_width=True, disabled=not season.has_next_match()):
                    nxt0 = season.peek_next_match()
                    if nxt0:
                        current_stop = nxt0["stop"]
                        while season.has_next_match():
                            nxtx = season.peek_next_match()
                            if nxtx and nxtx["stop"] != current_stop:
                                break
                            played = season.play_next_match()
                            st.session_state.season_last_played = played

            st.divider()
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
                    st.session_state.playoffs = PlayoffRunner(
                        rosters_df=st.session_state.rosters_df,
                        standings_sorted=season.standings(),
                        target_games=int(st.session_state.target_games),
                        carry_fatigue=bool(carry_fatigue_to_playoffs),
                        fatigue_by_team=(season.fatigue_by_team if carry_fatigue_to_playoffs else None),
                    )
                    st.session_state.mode = "playoffs"
                    st.session_state.playoffs_last_played = None
            else:
                st.info("Play the full season before starting playoffs.")
    else:
        if playoffs is None:
            st.error("Playoffs runner not initialized. Go back to season and click 'Start Playoffs'.")
            return

        left, right = st.columns([1.2, 1])
        with left:
            nxt = playoffs.peek_next_match()
            if nxt:
                st.markdown(f"### Next playoff game: **{nxt.team_a}** vs **{nxt.team_b}**")
                st.caption(f"Game: {nxt.game_id} · Round: {nxt.round_name}")
            elif playoffs.champion:
                st.success(f"CHAMPION: {playoffs.champion}")
            else:
                st.info("Next playoff game will be determined after earlier results.")

            btn_row = st.columns(2)
            with btn_row[0]:
                if st.button("▶️ Play Next Playoff Game", use_container_width=True, disabled=not playoffs.has_next_match()):
                    played = playoffs.play_next_match()
                    st.session_state.playoffs_last_played = played
            with btn_row[1]:
                if st.button("🏁 Play Rest of Playoffs", use_container_width=True, disabled=not playoffs.has_next_match()):
                    while playoffs.has_next_match():
                        played = playoffs.play_next_match()
                        st.session_state.playoffs_last_played = played

            st.divider()
            st.markdown("## Last Played Playoff Game")
            last = st.session_state.playoffs_last_played
            if last:
                st.caption(f"{last['game_id']} · {last['round']}")
                render_match_result(last["result"])
            else:
                st.info("No playoff games played yet.")

        with right:
            rep = playoffs.report()
            render_playoff_bracket(playoffs)

            st.divider()
            st.markdown("## Bracket State")
            st.caption("Eliminations")
            st.write("Seeds 7–8 eliminated:", ", ".join(rep.get("eliminated_pre_playoffs", [])) or "—")
            st.write("Play-in eliminated:", ", ".join(rep.get("eliminated_play_in", [])) or "—")

            st.divider()
            st.markdown("## Completed Games")
            games_df = _safe_df(rep.get("games", []))
            if not games_df.empty:
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


bootstrap_state()

st.sidebar.title("League Controls")
rosters_path = st.sidebar.text_input("Rosters CSV path (relative to engine.py)", value="data/rosters.csv")
st.session_state.target_games = st.sidebar.selectbox("Games per set (target_games)", options=[5, 6], index=0)
carry_fatigue_to_playoffs = st.sidebar.checkbox("Carry fatigue into playoffs", value=True)
ca, cb = st.sidebar.columns(2)
with ca:
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
with cb:
    if st.button("Reset League", use_container_width=True):
        st.session_state.season = None
        st.session_state.playoffs = None
        st.session_state.mode = "season"
        st.session_state.season_last_played = None
        st.session_state.playoffs_last_played = None
        st.success("Reset complete.")

if st.session_state.rosters_df is None:
    try:
        df = load_rosters_csv(rosters_path)
        st.session_state.rosters_df = df
        st.session_state.teams = sorted(df["team"].unique().tolist())
    except Exception:
        pass

st.sidebar.divider()
if st.session_state.teams:
    st.sidebar.caption("Teams")
    st.sidebar.write(", ".join(st.session_state.teams))
else:
    st.sidebar.info("Load rosters to begin.")

st.session_state.app_mode = st.sidebar.radio(
    "App Mode",
    options=["Season + Playoffs", "Classic Match Simulator"],
    index=0 if st.session_state.app_mode == "Season + Playoffs" else 1,
)

if st.session_state.app_mode == "Classic Match Simulator":
    render_classic_mode()
else:
    render_season_playoffs_mode(carry_fatigue_to_playoffs=carry_fatigue_to_playoffs)
