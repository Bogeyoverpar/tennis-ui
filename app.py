# app.py — upgraded UI: Bracket + Stats + Export
import io
import zipfile
from datetime import datetime

import streamlit as st
import pandas as pd

from engine import load_rosters_csv, SeasonRunner, PlayoffRunner

st.set_page_config(page_title="Tennis League (WTT) — Season + Playoffs", layout="wide")


# ----------------------------
# Helpers
# ----------------------------

def _df(x):
    if x is None:
        return pd.DataFrame()
    if isinstance(x, pd.DataFrame):
        return x
    return pd.DataFrame(x)

def standings_table_from_team_stats(team_stats: dict) -> pd.DataFrame:
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

def render_match_result(result: dict):
    if not result:
        st.info("No result.")
        return

    team_a = result["team_a"]
    team_b = result["team_b"]
    total_a = result["total_a"]
    total_b = result["total_b"]

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.subheader(f"{team_a} vs {team_b}")
    with c2:
        st.metric(team_a, total_a)
    with c3:
        st.metric(team_b, total_b)

    if result.get("decided_by_ot"):
        ot = result.get("ot") or {}
        st.warning(f"OT Super TB ({ot.get('set','?')}): {ot.get('tb_a','?')}–{ot.get('tb_b','?')} | Winner: {ot.get('winner','?')}")

    sets = result.get("sets", [])
    if sets:
        st.markdown("**Sets**")
        st.dataframe(_df(sets), use_container_width=True, hide_index=True)

    la = result.get("lineup_a", {})
    lb = result.get("lineup_b", {})
    if la and lb:
        st.markdown("**Lineups**")
        x, y = st.columns(2)
        with x:
            st.caption(team_a)
            st.json(la)
        with y:
            st.caption(team_b)
            st.json(lb)

def _flatten_season_results(season: SeasonRunner) -> pd.DataFrame:
    """
    Converts season.results (list of wrapped dicts) into a flat CSV-friendly table.
    """
    rows = []
    for w in getattr(season, "results", []):
        r = w.get("result", {})
        rows.append({
            "phase": "season",
            "stop": w.get("stop"),
            "round": w.get("round"),
            "match_index": w.get("match_index"),
            "team_a": r.get("team_a"),
            "team_b": r.get("team_b"),
            "total_a": r.get("total_a"),
            "total_b": r.get("total_b"),
            "decided_by_ot": r.get("decided_by_ot"),
            "ot_winner": (r.get("ot") or {}).get("winner") if r.get("ot") else None,
        })
    return pd.DataFrame(rows)

def _flatten_playoff_results(playoffs: PlayoffRunner) -> pd.DataFrame:
    rows = []
    rep = playoffs.report() if playoffs else {}
    for g in rep.get("games", []):
        r = g.get("result", {})
        rows.append({
            "phase": "playoffs",
            "game_id": g.get("game_id"),
            "round": g.get("round"),
            "team_a": g.get("team_a"),
            "team_b": g.get("team_b"),
            "winner": g.get("winner"),
            "total_a": r.get("total_a"),
            "total_b": r.get("total_b"),
            "decided_by_ot": r.get("decided_by_ot"),
        })
    return pd.DataFrame(rows)

def _player_stats_df(player_stats) -> pd.DataFrame:
    """
    Your engine uses defaultdict keyed by (team, player_name).
    Convert to a DataFrame.
    """
    rows = []
    for (_team, _player), s in player_stats.items():
        rows.append(dict(s))
    df = pd.DataFrame(rows)
    if not df.empty:
        # keep consistent column ordering (optional)
        preferred = [
            "Team", "Player", "MatchesPlayed", "SetsPlayed",
            "SinglesSets", "DoublesSets",
            "GamesWon", "GamesLost",
            "SetWins", "SetLosses",
            "OT_TB_PointsWon", "OT_TB_PointsLost",
        ]
        cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
        df = df[cols]
    return df

def _download_zip(files: dict) -> bytes:
    """
    files: { "filename.csv": bytes, ... }
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()

def bracket_card(title: str, a: str, b: str, winner: str | None = None, subtitle: str | None = None):
    wtxt = f"✅ {winner}" if winner else "TBD"
    sub = f"<div style='color:#6b7280;font-size:12px;margin-top:4px'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        f"""
        <div style="border:1px solid #e5e7eb;border-radius:14px;padding:12px 14px;margin-bottom:10px">
          <div style="font-weight:700">{title}</div>
          <div style="margin-top:8px">
            <div style="display:flex;justify-content:space-between"><span>{a}</span><span>vs</span><span>{b}</span></div>
          </div>
          {sub}
          <div style="margin-top:10px;font-weight:700;color:#111827">Winner: {wtxt}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ----------------------------
# Session state
# ----------------------------

for k, v in {
    "rosters_df": None,
    "teams": [],
    "season": None,
    "season_last_played": None,
    "playoffs": None,
    "playoffs_last_played": None,
    "mode": "season",
    "target_games": 5,
    "seed_base": 123,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ----------------------------
# Sidebar
# ----------------------------

st.sidebar.title("League Controls")
rosters_path = st.sidebar.text_input("Rosters CSV path", value="data/rosters.csv")
st.session_state.target_games = st.sidebar.selectbox("Games per set", options=[5, 6], index=0)
carry_fatigue_to_playoffs = st.sidebar.checkbox("Carry fatigue into playoffs", value=True)
st.session_state.seed_base = st.sidebar.number_input("Seed base", 0, 10_000_000, int(st.session_state.seed_base), 1)

c1, c2 = st.sidebar.columns(2)
with c1:
    if st.button("Load / Reload Rosters", use_container_width=True):
        try:
            df = load_rosters_csv(rosters_path)
            st.session_state.rosters_df = df
            st.session_state.teams = sorted(df["team"].unique().tolist())
            st.success(f"Loaded {len(st.session_state.teams)} teams.")
        except Exception as e:
            st.session_state.rosters_df = None
            st.session_state.teams = []
            st.error(str(e))
with c2:
    if st.button("Reset League", use_container_width=True):
        st.session_state.season = None
        st.session_state.playoffs = None
        st.session_state.mode = "season"
        st.session_state.season_last_played = None
        st.session_state.playoffs_last_played = None
        st.success("Reset complete.")

st.sidebar.divider()

page = st.sidebar.radio(
    "View",
    ["Season", "Playoffs", "Bracket", "Stats", "Export"],
    index=0
)


# ----------------------------
# Main header
# ----------------------------

st.title("🎾 Tennis League Simulator — Season + Playoffs (WTT)")
st.caption("Season: 7 stops / 14 rounds. Playoffs: NBA-style play-in with top-2 byes. One match at a time.")

if st.session_state.rosters_df is None or len(st.session_state.teams) != 8:
    st.warning("Load rosters with exactly 8 teams to continue.")
    st.stop()

# init season if missing
if st.session_state.season is None:
    st.session_state.season = SeasonRunner(
        rosters_df=st.session_state.rosters_df,
        teams=st.session_state.teams,
        target_games=int(st.session_state.target_games),
    )
    st.session_state.mode = "season"


season: SeasonRunner = st.session_state.season
playoffs: PlayoffRunner | None = st.session_state.playoffs


# ----------------------------
# Season page
# ----------------------------

if page == "Season":
    left, right = st.columns([1.2, 1])

    with left:
        nxt = season.peek_next_match()
        if nxt:
            st.markdown(f"### Next: **{nxt['home']}** vs **{nxt['away']}**")
            st.caption(f"STOP {nxt['stop']} · ROUND {nxt['round']} · Match {nxt['match_index']+1}/4")
        else:
            st.success("Season complete!")

        b1, b2 = st.columns(2)
        with b1:
            if st.button("▶️ Play Next Season Match", use_container_width=True, disabled=not season.has_next_match()):
                played = season.play_next_match(seed=int(st.session_state.seed_base + season.cursor))
                st.session_state.season_last_played = played
        with b2:
            if st.button("⏩ Play Remaining Stop", use_container_width=True, disabled=not season.has_next_match()):
                nxt0 = season.peek_next_match()
                if nxt0:
                    stop = nxt0["stop"]
                    while season.has_next_match():
                        nn = season.peek_next_match()
                        if nn and nn["stop"] != stop:
                            break
                        played = season.play_next_match(seed=int(st.session_state.seed_base + season.cursor))
                        st.session_state.season_last_played = played

        st.divider()
        st.markdown("## Last Played")
        if st.session_state.season_last_played:
            render_match_result(st.session_state.season_last_played["result"])
        else:
            st.info("No season matches played yet.")

    with right:
        st.markdown("## Standings")
        st.dataframe(standings_table_from_team_stats(season.team_stats), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("## Season Progress")
        total = len(season.schedule)
        done = season.cursor
        st.progress(done / total if total else 0)
        st.caption(f"{done}/{total} matches complete")

        st.divider()
        st.markdown("## Start Playoffs")
        if season.season_complete():
            if st.button("🏆 Start Playoffs", use_container_width=True):
                final_standings = season.standings()
                st.session_state.playoffs = PlayoffRunner(
                    rosters_df=st.session_state.rosters_df,
                    standings_sorted=final_standings,
                    target_games=int(st.session_state.target_games),
                    carry_fatigue=bool(carry_fatigue_to_playoffs),
                    fatigue_by_team=(season.fatigue_by_team if carry_fatigue_to_playoffs else None),
                )
                st.session_state.playoffs_last_played = None
                st.success("Playoffs initialized. Go to Playoffs/Bracket.")
        else:
            st.info("Finish the season to unlock playoffs.")


# ----------------------------
# Playoffs page
# ----------------------------

elif page == "Playoffs":
    if playoffs is None:
        st.warning("Playoffs not started yet. Finish season and click 'Start Playoffs'.")
        st.stop()

    left, right = st.columns([1.2, 1])

    with left:
        nxt = playoffs.peek_next_match()
        if nxt:
            st.markdown(f"### Next: **{nxt.team_a}** vs **{nxt.team_b}**")
            st.caption(f"{nxt.game_id} · {nxt.round_name}")
        else:
            if playoffs.champion:
                st.success(f"CHAMPION: {playoffs.champion}")
            else:
                st.info("Waiting for earlier results to determine next matchup.")

        b1, b2 = st.columns(2)
        with b1:
            if st.button("▶️ Play Next Playoff Game", use_container_width=True, disabled=not playoffs.has_next_match()):
                played = playoffs.play_next_match(seed=int(st.session_state.seed_base + 10_000 + playoffs.cursor))
                st.session_state.playoffs_last_played = played
        with b2:
            if st.button("🏁 Play Rest of Playoffs", use_container_width=True, disabled=not playoffs.has_next_match()):
                while playoffs.has_next_match():
                    played = playoffs.play_next_match(seed=int(st.session_state.seed_base + 10_000 + playoffs.cursor))
                    st.session_state.playoffs_last_played = played

        st.divider()
        st.markdown("## Last Played")
        if st.session_state.playoffs_last_played:
            render_match_result(st.session_state.playoffs_last_played["result"])
        else:
            st.info("No playoff games played yet.")

    with right:
        st.markdown("## Completed Playoff Games")
        rep = playoffs.report()
        games_df = _df(rep.get("games", []))
        if not games_df.empty:
            cols = [c for c in ["game_id", "round", "team_a", "team_b", "winner"] if c in games_df.columns]
            st.dataframe(games_df[cols], use_container_width=True, hide_index=True)
        else:
            st.info("None yet.")

        st.divider()
        st.markdown("## Champion")
        if rep.get("champion"):
            st.success(rep["champion"])
        else:
            st.info("TBD")


# ----------------------------
# Bracket page
# ----------------------------

elif page == "Bracket":
    if playoffs is None:
        st.warning("Playoffs not started.")
        st.stop()

    rep = playoffs.report()
    seeded = rep.get("seeded", {})  # if you included it in report; if not, we can infer from standings
    completed = {g["game_id"]: g for g in rep.get("games", [])}

    # Try to reconstruct teams even if report doesn't include seeded map
    # (Your PlayoffRunner stores games internally, so we'll use that)
    g = getattr(playoffs, "games", {})

    st.markdown("## Playoff Bracket")
    st.caption("Play-in → Semis → Final (playoffs treated as one long stop until the championship).")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### Play-In")
        for gid in ["PLAYIN_A", "PLAYIN_B", "PLAYIN_C"]:
            gg = g.get(gid)
            if gg is None:
                continue
            bracket_card(
                gid,
                gg.team_a,
                gg.team_b,
                winner=(completed.get(gid) or {}).get("winner"),
                subtitle="Single match"
            )

    with c2:
        st.markdown("### Semifinals")
        for gid in ["SEMI_1", "SEMI_2"]:
            gg = g.get(gid)
            if gg is None:
                continue
            bracket_card(
                gid,
                gg.team_a,
                gg.team_b,
                winner=(completed.get(gid) or {}).get("winner"),
                subtitle="Single match"
            )

    with c3:
        st.markdown("### Final")
        gg = g.get("FINAL")
        if gg:
            bracket_card(
                "FINAL",
                gg.team_a,
                gg.team_b,
                winner=(completed.get("FINAL") or {}).get("winner"),
                subtitle="Championship (stop_used reset here)"
            )

    if rep.get("champion"):
        st.success(f"Champion: {rep['champion']}")


# ----------------------------
# Stats page
# ----------------------------

elif page == "Stats":
    st.markdown("## Team Stats (Season)")
    st.dataframe(standings_table_from_team_stats(season.team_stats), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("## Player Leaders (Season)")

    ps_df = _player_stats_df(season.player_stats)
    if ps_df.empty:
        st.info("No player stats yet.")
    else:
        # Simple leaders
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Games Won**")
            st.dataframe(ps_df.sort_values("GamesWon", ascending=False).head(15), use_container_width=True, hide_index=True)

        with c2:
            st.markdown("**Set Wins**")
            st.dataframe(ps_df.sort_values("SetWins", ascending=False).head(15), use_container_width=True, hide_index=True)

        with c3:
            st.markdown("**Matches Played**")
            st.dataframe(ps_df.sort_values("MatchesPlayed", ascending=False).head(15), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("## Full Player Stat Table")
        st.dataframe(ps_df, use_container_width=True, hide_index=True)

    if playoffs is not None:
        st.divider()
        st.markdown("## Playoff Stats (optional separate accumulators)")
        # If your PlayoffRunner tracks its own team_stats/player_stats:
        if hasattr(playoffs, "team_stats"):
            st.markdown("**Team stats (playoffs-only accumulator)**")
            st.dataframe(standings_table_from_team_stats(playoffs.team_stats), use_container_width=True, hide_index=True)
        if hasattr(playoffs, "player_stats"):
            st.markdown("**Player stats (playoffs-only accumulator)**")
            st.dataframe(_player_stats_df(playoffs.player_stats), use_container_width=True, hide_index=True)


# ----------------------------
# Export page
# ----------------------------

elif page == "Export":
    st.markdown("## Export Season + Playoffs")

    season_matches_df = _flatten_season_results(season)
    season_team_df = standings_table_from_team_stats(season.team_stats)
    season_player_df = _player_stats_df(season.player_stats)

    playoff_games_df = pd.DataFrame()
    playoff_team_df = pd.DataFrame()
    playoff_player_df = pd.DataFrame()

    if playoffs is not None:
        playoff_games_df = _flatten_playoff_results(playoffs)
        if hasattr(playoffs, "team_stats"):
            playoff_team_df = standings_table_from_team_stats(playoffs.team_stats)
        if hasattr(playoffs, "player_stats"):
            playoff_player_df = _player_stats_df(playoffs.player_stats)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Download CSVs")
        st.download_button(
            "Download season_results.csv",
            data=season_matches_df.to_csv(index=False).encode("utf-8"),
            file_name="season_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Download season_team_stats.csv",
            data=season_team_df.to_csv(index=False).encode("utf-8"),
            file_name="season_team_stats.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Download season_player_stats.csv",
            data=season_player_df.to_csv(index=False).encode("utf-8"),
            file_name="season_player_stats.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if playoffs is not None:
            st.download_button(
                "Download playoff_results.csv",
                data=playoff_games_df.to_csv(index=False).encode("utf-8"),
                file_name="playoff_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
            if not playoff_team_df.empty:
                st.download_button(
                    "Download playoff_team_stats.csv",
                    data=playoff_team_df.to_csv(index=False).encode("utf-8"),
                    file_name="playoff_team_stats.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            if not playoff_player_df.empty:
                st.download_button(
                    "Download playoff_player_stats.csv",
                    data=playoff_player_df.to_csv(index=False).encode("utf-8"),
                    file_name="playoff_player_stats.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        else:
            st.info("Playoffs not started yet — playoff exports will appear once initialized.")

    with c2:
        st.markdown("### Download Everything (ZIP)")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        files = {
            "season_results.csv": season_matches_df.to_csv(index=False).encode("utf-8"),
            "season_team_stats.csv": season_team_df.to_csv(index=False).encode("utf-8"),
            "season_player_stats.csv": season_player_df.to_csv(index=False).encode("utf-8"),
        }
        if playoffs is not None:
            files["playoff_results.csv"] = playoff_games_df.to_csv(index=False).encode("utf-8")
            if not playoff_team_df.empty:
                files["playoff_team_stats.csv"] = playoff_team_df.to_csv(index=False).encode("utf-8")
            if not playoff_player_df.empty:
                files["playoff_player_stats.csv"] = playoff_player_df.to_csv(index=False).encode("utf-8")

        zip_bytes = _download_zip(files)
        st.download_button(
            "Download season_playoffs_export.zip",
            data=zip_bytes,
            file_name=f"season_playoffs_export_{stamp}.zip",
            mime="application/zip",
            use_container_width=True,
        )