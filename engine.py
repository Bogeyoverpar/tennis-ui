import random
import math
from itertools import combinations
from collections import defaultdict
from pathlib import Path

import pandas as pd


# ============================================================
# CSV LOADER
# ============================================================
def load_rosters_csv(path="data/rosters.csv") -> pd.DataFrame:
    """
    Loads roster CSV. Expected columns:
      team, gender, name, serve, ret, rally, clutch
    gender must be M or F (case-insensitive).
    """
    base = Path(__file__).resolve().parent
    full_path = base / path

    df = pd.read_csv(full_path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"team", "gender", "name", "serve", "ret", "rally", "clutch"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"rosters.csv missing columns: {sorted(missing)}")

    # normalize
    df["team"] = df["team"].astype(str).str.strip()
    df["gender"] = df["gender"].astype(str).str.strip().str.upper()
    df["name"] = df["name"].astype(str).str.strip()

    # numeric ratings (drop bad rows)
    for c in ["serve", "ret", "rally", "clutch"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["team", "gender", "name", "serve", "ret", "rally", "clutch"])
    df = df[df["gender"].isin(["M", "F"])]

    return df


# ============================================================
# RATINGS + PROBABILITY
# ============================================================
def apply_fatigue_value(player_row: dict, fp: int):
    """
    fp = fatigue points.
    Simple model:
      - serve, ret: -1 per fp
      - rally: -2 per fp
      - clutch unchanged
    """
    return {
        "serve": player_row["serve"] - 1 * fp,
        "ret": player_row["ret"] - 1 * fp,
        "rally": player_row["rally"] - 2 * fp,
        "clutch": player_row["clutch"],
    }


def singles_rating_row(player_row: dict, fp: int = 0) -> float:
    s = apply_fatigue_value(player_row, fp)
    return 0.35 * s["serve"] + 0.35 * s["ret"] + 0.20 * s["rally"] + 0.10 * s["clutch"]


def doubles_rating_rows(p1: dict, p2: dict, fp1: int = 0, fp2: int = 0) -> float:
    a = apply_fatigue_value(p1, fp1)
    b = apply_fatigue_value(p2, fp2)

    avg_serve = (a["serve"] + b["serve"]) / 2
    avg_ret = (a["ret"] + b["ret"]) / 2
    avg_rally = (a["rally"] + b["rally"]) / 2
    avg_clutch = (a["clutch"] + b["clutch"]) / 2

    return 0.25 * avg_serve + 0.25 * avg_ret + 0.30 * avg_rally + 0.20 * avg_clutch


def win_probability(rating_a, rating_b, scale=8):
    return 1 / (1 + math.exp(-(rating_a - rating_b) / scale))


# ============================================================
# SIMS (SETS + SUPER TB)
# ============================================================
def simulate_set(p, target_games=5):
    a = b = 0
    while a < target_games and b < target_games:
        if random.random() < p:
            a += 1
        else:
            b += 1
    return a, b


def simulate_super_tiebreak(p, target=10, win_by=2):
    a = b = 0
    while True:
        if random.random() < p:
            a += 1
        else:
            b += 1
        if (a >= target or b >= target) and abs(a - b) >= win_by:
            return a, b


# ============================================================
# ROSTER → PLAYER ROWS
# ============================================================
def _to_player_rows(rosters_df: pd.DataFrame, team_name: str, gender: str):
    gender = gender.upper()
    df = rosters_df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    team_df = df[(df["team"] == team_name) & (df["gender"].str.upper() == gender)].copy()
    players = []
    for _, r in team_df.iterrows():
        players.append({
            "name": str(r["name"]).strip(),
            "team": team_name,
            "gender": gender,
            "serve": float(r["serve"]),
            "ret": float(r["ret"]),
            "rally": float(r["rally"]),
            "clutch": float(r["clutch"]),
        })
    return players


# ============================================================
# STATS CONTAINERS
# ============================================================
def init_team_stats(team_names):
    return {
        t: {"MP": 0, "W": 0, "L": 0, "OTL": 0, "PTS": 0, "GF": 0, "GA": 0, "GD": 0}
        for t in team_names
    }


def init_player_stats():
    # key: (team, player_name)
    return defaultdict(lambda: {
        "Team": "",
        "Player": "",
        "SetsPlayed": 0,
        "MatchesPlayed": 0,
        "GamesWon": 0,
        "GamesLost": 0,
        "SinglesSets": 0,
        "DoublesSets": 0,
        "SetWins": 0,
        "SetLosses": 0,
        "OT_TB_PointsWon": 0,
        "OT_TB_PointsLost": 0,
    })


def record_match_participation(player_stats, team, players):
    # counts unique match appearances
    for p in set(players):
        key = (team, p)
        player_stats[key]["Team"] = team
        player_stats[key]["Player"] = p
        player_stats[key]["MatchesPlayed"] += 1


# ============================================================
# LINEUP (STAR + BALANCE) WITH LIMITS
# ============================================================
def eligible(players, match_used, stop_used, max_match=2, max_stop=3):
    return [
        p for p in players
        if match_used[p["name"]] < max_match and stop_used[p["name"]] < max_stop
    ]


def best_singles_player(players, fatigue, match_used, stop_used, prefer_name=None):
    cand = eligible(players, match_used, stop_used)
    if not cand:
        return None

    def score(p):
        base = singles_rating_row(p, fatigue.get(p["name"], 0))
        if prefer_name and p["name"] == prefer_name:
            base += 1.5
        base -= 0.8 * match_used[p["name"]]
        base -= 0.4 * stop_used[p["name"]]
        return base

    return max(cand, key=score)


def best_doubles_pair(players, fatigue, match_used, stop_used, avoid=set()):
    cand = eligible(players, match_used, stop_used)
    if len(cand) < 2:
        return None

    pairs = list(combinations(cand, 2))

    def score(pair):
        p1, p2 = pair
        base = doubles_rating_rows(
            p1, p2,
            fatigue.get(p1["name"], 0),
            fatigue.get(p2["name"], 0),
        )
        if p1["name"] in avoid:
            base -= 1.5
        if p2["name"] in avoid:
            base -= 1.5
        base -= 0.8 * (match_used[p1["name"]] + match_used[p2["name"]])
        base -= 0.4 * (stop_used[p1["name"]] + stop_used[p2["name"]])
        return base

    return max(pairs, key=score)


def best_mixed_pair(men, women, fatigue, match_used, stop_used, prefer=set()):
    m_cand = eligible(men, match_used, stop_used)
    w_cand = eligible(women, match_used, stop_used)
    if not m_cand or not w_cand:
        return None

    best = None
    best_score = -1e18
    for m in m_cand:
        for w in w_cand:
            base = doubles_rating_rows(m, w, fatigue.get(m["name"], 0), fatigue.get(w["name"], 0))
            if m["name"] in prefer:
                base += 0.8
            if w["name"] in prefer:
                base += 0.8
            base -= 0.8 * (match_used[m["name"]] + match_used[w["name"]])
            base -= 0.4 * (stop_used[m["name"]] + stop_used[w["name"]])
            if base > best_score:
                best_score = base
                best = (m, w)
    return best


def build_lineup_star_balance(team_men, team_women, fatigue, stop_used):
    all_players = team_men + team_women
    match_used = {p["name"]: 0 for p in all_players}

    star_m = max(team_men, key=lambda p: singles_rating_row(p, 0))
    star_w = max(team_women, key=lambda p: singles_rating_row(p, 0))

    ms = best_singles_player(team_men, fatigue, match_used, stop_used, prefer_name=star_m["name"])
    ws = best_singles_player(team_women, fatigue, match_used, stop_used, prefer_name=star_w["name"])
    if ms is None or ws is None:
        raise RuntimeError("Not enough eligible singles players")

    match_used[ms["name"]] += 1
    match_used[ws["name"]] += 1

    avoid = {ms["name"], ws["name"]}
    md = best_doubles_pair(team_men, fatigue, match_used, stop_used, avoid=avoid)
    wd = best_doubles_pair(team_women, fatigue, match_used, stop_used, avoid=avoid)
    if md is None or wd is None:
        raise RuntimeError("Not enough eligible doubles players")

    match_used[md[0]["name"]] += 1
    match_used[md[1]["name"]] += 1
    match_used[wd[0]["name"]] += 1
    match_used[wd[1]["name"]] += 1

    xd = best_mixed_pair(team_men, team_women, fatigue, match_used, stop_used, prefer={ws["name"]})
    if xd is None:
        raise RuntimeError("Not enough eligible mixed players")

    return {"MS": ms, "WS": ws, "MD": md, "WD": wd, "XD": xd}


# ============================================================
# MATCH SIM (LINEUP + STATS + OT)
# ============================================================
def simulate_match_wtt_with_stats(
    rosters_df,
    team_a,
    team_b,
    target_games=5,
    fatigue_a=None,
    fatigue_b=None,
    stop_used_a=None,
    stop_used_b=None,
    team_stats=None,
    player_stats=None,
):
    """
    Returns:
      result dict with lineups, set log, OT info, and updated team/player stats.
    Points:
      Win = 2 pts
      Loss = 0 pts
      OT Loss = 1 pt
    Record displayed as W-L-OTL.
    """

    if fatigue_a is None: fatigue_a = {}
    if fatigue_b is None: fatigue_b = {}
    if stop_used_a is None: stop_used_a = defaultdict(int)
    if stop_used_b is None: stop_used_b = defaultdict(int)

    if team_stats is None:
        team_stats = init_team_stats([team_a, team_b])
    if player_stats is None:
        player_stats = init_player_stats()

    # players
    a_m = _to_player_rows(rosters_df, team_a, "M")
    a_w = _to_player_rows(rosters_df, team_a, "F")
    b_m = _to_player_rows(rosters_df, team_b, "M")
    b_w = _to_player_rows(rosters_df, team_b, "F")

    if len(a_m) < 3 or len(a_w) < 3 or len(b_m) < 3 or len(b_w) < 3:
        raise ValueError("Each team must have at least 3 men and 3 women.")

    # initialize usage maps
    for p in a_m + a_w:
        fatigue_a.setdefault(p["name"], 0)
        stop_used_a.setdefault(p["name"], 0)
    for p in b_m + b_w:
        fatigue_b.setdefault(p["name"], 0)
        stop_used_b.setdefault(p["name"], 0)

    # build lineups
    la = build_lineup_star_balance(a_m, a_w, fatigue_a, stop_used_a)
    lb = build_lineup_star_balance(b_m, b_w, fatigue_b, stop_used_b)

    sets_order = ["MS", "WS", "MD", "WD", "XD"]
    set_log = []
    total_a = 0
    total_b = 0

    last_set = None
    match_players_a = []
    match_players_b = []

    for set_name in sets_order:
        if set_name in ["MS", "WS"]:
            pa = la[set_name]
            pb = lb[set_name]
            ra = singles_rating_row(pa, fatigue_a[pa["name"]])
            rb = singles_rating_row(pb, fatigue_b[pb["name"]])
            a_players = [pa["name"]]
            b_players = [pb["name"]]
            is_singles = True
        else:
            pa1, pa2 = la[set_name]
            pb1, pb2 = lb[set_name]
            ra = doubles_rating_rows(pa1, pa2, fatigue_a[pa1["name"]], fatigue_a[pa2["name"]])
            rb = doubles_rating_rows(pb1, pb2, fatigue_b[pb1["name"]], fatigue_b[pb2["name"]])
            a_players = [pa1["name"], pa2["name"]]
            b_players = [pb1["name"], pb2["name"]]
            is_singles = False

        p = win_probability(ra, rb)
        a_games, b_games = simulate_set(p, target_games=target_games)

        total_a += a_games
        total_b += b_games
        last_set = (set_name, a_players, b_players, p)

        # update fatigue + stop usage + match usage list
        for name in a_players:
            fatigue_a[name] += 1
            stop_used_a[name] += 1
            match_players_a.append(name)
        for name in b_players:
            fatigue_b[name] += 1
            stop_used_b[name] += 1
            match_players_b.append(name)

        # player stats
        for name in a_players:
            key = (team_a, name)
            ps = player_stats[key]
            ps["Team"] = team_a; ps["Player"] = name
            ps["SetsPlayed"] += 1
            ps["GamesWon"] += a_games
            ps["GamesLost"] += b_games
            ps["SinglesSets"] += 1 if is_singles else 0
            ps["DoublesSets"] += 0 if is_singles else 1
            ps["SetWins"] += 1 if a_games > b_games else 0
            ps["SetLosses"] += 1 if a_games < b_games else 0

        for name in b_players:
            key = (team_b, name)
            ps = player_stats[key]
            ps["Team"] = team_b; ps["Player"] = name
            ps["SetsPlayed"] += 1
            ps["GamesWon"] += b_games
            ps["GamesLost"] += a_games
            ps["SinglesSets"] += 1 if is_singles else 0
            ps["DoublesSets"] += 0 if is_singles else 1
            ps["SetWins"] += 1 if b_games > a_games else 0
            ps["SetLosses"] += 1 if b_games < a_games else 0

        set_log.append({
            "set": set_name,
            "a_games": a_games,
            "b_games": b_games,
            "a_players": ", ".join(a_players),
            "b_players": ", ".join(b_players),
            "p_a": round(p, 3),
        })

    decided_by_ot = False
    ot = None

    # OT if tied
    if total_a == total_b:
        decided_by_ot = True
        set_name, a_players, b_players, p_last = last_set
        tb_a, tb_b = simulate_super_tiebreak(p_last)

        if tb_a > tb_b:
            total_a += 1
            ot_winner = team_a
        else:
            total_b += 1
            ot_winner = team_b

        # TB points
        for name in a_players:
            key = (team_a, name)
            player_stats[key]["OT_TB_PointsWon"] += tb_a
            player_stats[key]["OT_TB_PointsLost"] += tb_b
        for name in b_players:
            key = (team_b, name)
            player_stats[key]["OT_TB_PointsWon"] += tb_b
            player_stats[key]["OT_TB_PointsLost"] += tb_a

        ot = {"tb_a": tb_a, "tb_b": tb_b, "winner": ot_winner, "set": set_name}

    # match participation
    record_match_participation(player_stats, team_a, match_players_a)
    record_match_participation(player_stats, team_b, match_players_b)

    # team stats
    team_stats[team_a]["MP"] += 1
    team_stats[team_b]["MP"] += 1

    team_stats[team_a]["GF"] += total_a
    team_stats[team_a]["GA"] += total_b
    team_stats[team_b]["GF"] += total_b
    team_stats[team_b]["GA"] += total_a

    # points + W/L/OTL
    if total_a > total_b:
        team_stats[team_a]["W"] += 1
        team_stats[team_a]["PTS"] += 2
        if decided_by_ot:
            team_stats[team_b]["OTL"] += 1
            team_stats[team_b]["PTS"] += 1
        else:
            team_stats[team_b]["L"] += 1
    else:
        team_stats[team_b]["W"] += 1
        team_stats[team_b]["PTS"] += 2
        if decided_by_ot:
            team_stats[team_a]["OTL"] += 1
            team_stats[team_a]["PTS"] += 1
        else:
            team_stats[team_a]["L"] += 1

    # GD update
    for t in [team_a, team_b]:
        team_stats[t]["GD"] = team_stats[t]["GF"] - team_stats[t]["GA"]

    return {
        "team_a": team_a,
        "team_b": team_b,
        "total_a": total_a,
        "total_b": total_b,
        "decided_by_ot": decided_by_ot,
        "ot": ot,
        "sets": set_log,
        "lineup_a": {
            "MS": la["MS"]["name"],
            "WS": la["WS"]["name"],
            "MD": (la["MD"][0]["name"], la["MD"][1]["name"]),
            "WD": (la["WD"][0]["name"], la["WD"][1]["name"]),
            "XD": (la["XD"][0]["name"], la["XD"][1]["name"]),
        },
        "lineup_b": {
            "MS": lb["MS"]["name"],
            "WS": lb["WS"]["name"],
            "MD": (lb["MD"][0]["name"], lb["MD"][1]["name"]),
            "WD": (lb["WD"][0]["name"], lb["WD"][1]["name"]),
            "XD": (lb["XD"][0]["name"], lb["XD"][1]["name"]),
        },
        "team_stats": team_stats,
        "player_stats": player_stats,
        "fatigue_a": dict(fatigue_a),
        "fatigue_b": dict(fatigue_b),
        "stop_used_a": dict(stop_used_a),
        "stop_used_b": dict(stop_used_b),
    }
# ============================================================
# SEASON MODE + PLAYOFF MODE (one match at a time)
# Works with your existing simulate_match_wtt_with_stats() (WTT rules)
# ============================================================
#
# What this adds:
# - 8-team double round robin (each opponent twice) across 7 stops
# - Each stop contains 2 rounds; each team plays 2 different opponents per stop
# - Fatigue persists across the whole season (you already do this via fatigue dicts)
# - "stop_used" resets at the start of each stop (so max_stop=3 makes sense per stop)
# - Plays season ONE MATCH at a time
# - When season ends, runs playoffs ONE MATCH at a time:
#     Seeds 1–2 bye
#     Seeds 7–8 eliminated
#     Play-in A: 3 vs 4
#     Play-in B: 5 vs 6
#     Play-in C: loser(A) vs winner(B)
#     Semis: 1 vs winner(C), 2 vs winner(A)
#     Final
#
# Paste this whole block at the BOTTOM of engine.py (below your existing code).
# ============================================================

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


# ----------------------------
# SCHEDULING (8 teams)
# ----------------------------

def _round_robin_pairs_even(teams: List[str]) -> List[List[Tuple[str, str]]]:
    """
    Circle method schedule for even # of teams.
    Returns N-1 rounds, each with N/2 matches.
    For 8 teams => 7 rounds x 4 matches.
    """
    if len(teams) % 2 != 0:
        raise ValueError("Even number of teams required for this scheduler.")
    n = len(teams)
    arr = teams[:]
    rounds: List[List[Tuple[str, str]]] = []

    for r in range(n - 1):
        pairs: List[Tuple[str, str]] = []
        for i in range(n // 2):
            a = arr[i]
            b = arr[n - 1 - i]
            # alternate home/away a bit
            pairs.append((a, b) if (r % 2 == 0) else (b, a))
        rounds.append(pairs)

        fixed = arr[0]
        rest = arr[1:]
        rest = [rest[-1]] + rest[:-1]
        arr = [fixed] + rest

    return rounds


def build_season_stops(teams: List[str]) -> List[Dict[str, Any]]:
    """
    8 teams, double round robin => 14 rounds.
    7 stops x 2 rounds/stop = 14 rounds.
    Each round has 4 matches => each stop has 8 matches total.
    """
    if len(teams) != 8:
        raise ValueError("This season format is built for exactly 8 teams.")

    first_leg = _round_robin_pairs_even(teams)                    # 7 rounds
    second_leg = [[(b, a) for (a, b) in rnd] for rnd in first_leg]  # swap home/away
    all_rounds = first_leg + second_leg                           # 14 rounds

    stops: List[Dict[str, Any]] = []
    round_num = 1
    for stop in range(1, 8):
        stop_rounds = []
        for _ in range(2):
            stop_rounds.append({
                "round": round_num,
                "matches": all_rounds[round_num - 1],  # list of (home, away)
            })
            round_num += 1
        stops.append({"stop": stop, "rounds": stop_rounds})
    return stops


def flatten_schedule(stops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Flattens to a single list of matches in the exact play order.
    """
    out: List[Dict[str, Any]] = []
    for stop_blob in stops:
        stop = stop_blob["stop"]
        for rnd_blob in stop_blob["rounds"]:
            rnd = rnd_blob["round"]
            for idx, (home, away) in enumerate(rnd_blob["matches"]):
                out.append({
                    "stop": stop,
                    "round": rnd,
                    "match_index": idx,
                    "home": home,
                    "away": away,
                })
    return out


# ----------------------------
# STANDINGS (use your team_stats)
# ----------------------------

def standings_from_team_stats(team_stats: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort order (typical for your points system):
      1) PTS desc
      2) W desc
      3) GD desc
      4) GF desc
      5) GA asc
      6) team name asc (stable)
    Returns list of dict rows with at least 'team'.
    """
    rows = []
    for team, s in team_stats.items():
        rows.append({
            "team": team,
            "PTS": int(s.get("PTS", 0)),
            "W": int(s.get("W", 0)),
            "L": int(s.get("L", 0)),
            "OTL": int(s.get("OTL", 0)),
            "GF": int(s.get("GF", 0)),
            "GA": int(s.get("GA", 0)),
            "GD": int(s.get("GD", 0)),
            "MP": int(s.get("MP", 0)),
        })

    rows.sort(
        key=lambda r: (
            r["PTS"],
            r["W"],
            r["GD"],
            r["GF"],
            -r["GA"],   # invert so lower GA ranks higher
            -ord(r["team"][0]) if r["team"] else 0  # deterministic fallback
        ),
        reverse=True
    )
    # That GA invert is a bit hacky with reverse=True; let's do it cleanly:
    rows.sort(
        key=lambda r: (
            -r["PTS"],
            -r["W"],
            -r["GD"],
            -r["GF"],
            r["GA"],
            r["team"].lower(),
        )
    )
    return rows


# ----------------------------
# SEASON RUNNER (one match at a time)
# ----------------------------

@dataclass
class SeasonRunner:
    rosters_df: "pd.DataFrame"
    teams: List[str]
    target_games: int = 5

    stops: List[Dict[str, Any]] = field(init=False)
    schedule: List[Dict[str, Any]] = field(init=False)
    cursor: int = 0
    last_stop: int = 0

    # shared season-long stats
    team_stats: Dict[str, Dict[str, Any]] = field(init=False)
    player_stats: Any = field(init=False)

    # fatigue persists across season
    fatigue_by_team: Dict[str, Dict[str, int]] = field(init=False)

    # stop_used resets each stop
    stop_used_by_team: Dict[str, Dict[str, int]] = field(init=False)

    def __post_init__(self):
        self.stops = build_season_stops(self.teams)
        self.schedule = flatten_schedule(self.stops)

        self.team_stats = init_team_stats(self.teams)
        self.player_stats = init_player_stats()
        self.results = []

        self.fatigue_by_team = {t: {} for t in self.teams}
        self.stop_used_by_team = {t: defaultdict(int) for t in self.teams}

    def has_next_match(self) -> bool:
        return self.cursor < len(self.schedule)

    def peek_next_match(self) -> Optional[Dict[str, Any]]:
        return self.schedule[self.cursor] if self.has_next_match() else None

    def _reset_stop_usage(self):
        self.stop_used_by_team = {t: defaultdict(int) for t in self.teams}

    def play_next_match(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Plays exactly ONE regular season match and advances cursor.
        Handles stop transition (resets stop_used) automatically.
        """
        if not self.has_next_match():
            raise RuntimeError("Season complete. No matches remaining.")

        sm = self.schedule[self.cursor]
        stop = sm["stop"]
        home = sm["home"]
        away = sm["away"]

        # if new stop, reset stop usage counts
        if self.last_stop and stop != self.last_stop:
            self._reset_stop_usage()

        res = simulate_match_wtt_with_stats(
            rosters_df=self.rosters_df,
            team_a=home,
            team_b=away,
            seed=seed,
            target_games=self.target_games,
            fatigue_a=self.fatigue_by_team[home],
            fatigue_b=self.fatigue_by_team[away],
            stop_used_a=self.stop_used_by_team[home],
            stop_used_b=self.stop_used_by_team[away],
            team_stats=self.team_stats,
            player_stats=self.player_stats,
        )

        # your simulate_match returns updated fatigue/stop_used dict snapshots; keep references updated
        self.fatigue_by_team[home] = res["fatigue_a"]
        self.fatigue_by_team[away] = res["fatigue_b"]
        # stop_used are per-stop counters; keep them too
        self.stop_used_by_team[home] = defaultdict(int, res["stop_used_a"])
        self.stop_used_by_team[away] = defaultdict(int, res["stop_used_b"])

        wrapped = {
            "phase": "season",
            "stop": stop,
            "round": sm["round"],
            "match_index": sm["match_index"],
            "home": home,
            "away": away,
            "result": res,
            "team_stats": self.team_stats,
            "player_stats": self.player_stats,
        }
        self.results.append(wrapped)

        self.cursor += 1
        self.last_stop = stop
        return wrapped

    def season_complete(self) -> bool:
        return not self.has_next_match()

    def standings(self) -> List[Dict[str, Any]]:
        return standings_from_team_stats(self.team_stats)


# ----------------------------
# PLAYOFF RUNNER (one match at a time)
# ----------------------------

@dataclass
class PlayoffGame:
    game_id: str
    round_name: str
    team_a: str
    team_b: str
    result: Optional[Dict[str, Any]] = None
    winner: Optional[str] = None


def _winner_from_match_result(match_result: Dict[str, Any]) -> str:
    # your simulate_match dict uses totals:
    if match_result["total_a"] > match_result["total_b"]:
        return match_result["team_a"]
    return match_result["team_b"]


@dataclass
class PlayoffRunner:
    rosters_df: "pd.DataFrame"
    standings_sorted: List[Dict[str, Any]]
    target_games: int = 5

    # carry fatigue from season into playoffs (set False to reset everyone)
    carry_fatigue: bool = True
    fatigue_by_team: Optional[Dict[str, Dict[str, int]]] = None

    cursor: int = 0
    champion: Optional[str] = None

    games: Dict[str, PlayoffGame] = field(init=False)
    order: List[str] = field(default_factory=lambda: ["PLAYIN_A", "PLAYIN_B", "PLAYIN_C", "SEMI_1", "SEMI_2", "FINAL"])

    # internal dependencies
    _winner_a: Optional[str] = None
    _loser_a: Optional[str] = None
    _winner_b: Optional[str] = None

    # per-match stop_used in playoffs (fresh each match)
    stop_used_by_team: Dict[str, Dict[str, int]] = field(init=False)

    # playoff stats can continue accumulating (optional)
    team_stats: Dict[str, Dict[str, Any]] = field(init=False)
    player_stats: Any = field(init=False)

    eliminated_pre: List[str] = field(default_factory=list)
    eliminated_playin: List[str] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        seeded = {i + 1: self.standings_sorted[i]["team"] for i in range(8)}

        self.eliminated_pre = [seeded[7], seeded[8]]

        self.games = {
            "PLAYIN_A": PlayoffGame("PLAYIN_A", "Play-In", seeded[3], seeded[4]),
            "PLAYIN_B": PlayoffGame("PLAYIN_B", "Play-In", seeded[5], seeded[6]),
            "PLAYIN_C": PlayoffGame("PLAYIN_C", "Play-In", "TBD", "TBD"),
            "SEMI_1":   PlayoffGame("SEMI_1", "Semifinal", seeded[1], "TBD"),
            "SEMI_2":   PlayoffGame("SEMI_2", "Semifinal", seeded[2], "TBD"),
            "FINAL":    PlayoffGame("FINAL", "Final", "TBD", "TBD"),
        }

        # stop_used fresh each match in playoffs
        self.stop_used_by_team = {t: defaultdict(int) for t in seeded.values()}

        # if you want playoff stats separate, re-init here; if you want to keep season stats, pass them in.
        self.team_stats = init_team_stats(list(seeded.values()))
        self.player_stats = init_player_stats()

        # fatigue handling
        if not self.carry_fatigue or self.fatigue_by_team is None:
            self.fatigue_by_team = {t: {} for t in seeded.values()}

    def has_next_match(self) -> bool:
        return self.champion is None and self.cursor < len(self.order)

    def peek_next_match(self) -> Optional[PlayoffGame]:
        if not self.has_next_match():
            return None
        gid = self.order[self.cursor]
        g = self.games[gid]
        if g.team_a == "TBD" or g.team_b == "TBD":
            return None
        return g

    def _fresh_stop_used(self, team: str) -> Dict[str, int]:
        # new dict each match so "max_stop=3" applies per playoff match
        return defaultdict(int)

    def play_next_match(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if not self.has_next_match():
            raise RuntimeError("Playoffs complete. No matches remaining.")

        gid = self.order[self.cursor]
        g = self.games[gid]
        if g.team_a == "TBD" or g.team_b == "TBD":
            raise RuntimeError(f"{gid} is not ready yet (teams TBD).")

        a = g.team_a
        b = g.team_b

        res = simulate_match_wtt_with_stats(
            rosters_df=self.rosters_df,
            team_a=a,
            team_b=b,
            seed=seed,
            target_games=self.target_games,
            fatigue_a=self.fatigue_by_team[a],
            fatigue_b=self.fatigue_by_team[b],
            stop_used_a=self._fresh_stop_used(a),
            stop_used_b=self._fresh_stop_used(b),
            team_stats=self.team_stats,
            player_stats=self.player_stats,
        )

        winner = _winner_from_match_result(res)
        loser = b if winner == a else a

        g.result = res
        g.winner = winner

        # keep fatigue updated if carrying through playoffs
        self.fatigue_by_team[a] = res["fatigue_a"]
        self.fatigue_by_team[b] = res["fatigue_b"]

        wrapped = {
            "phase": "playoffs",
            "game_id": gid,
            "round": g.round_name,
            "team_a": a,
            "team_b": b,
            "winner": winner,
            "result": res,
        }
        self.results.append(wrapped)

        # advance bracket
        if gid == "PLAYIN_A":
            self._winner_a = winner
            self._loser_a = loser
            self.games["SEMI_2"].team_b = self._winner_a
        elif gid == "PLAYIN_B":
            self._winner_b = winner
            self.eliminated_playin.append(loser)
        elif gid == "PLAYIN_C":
            # winner -> SEMI_1
            self.games["SEMI_1"].team_b = winner
            self.eliminated_playin.append(loser)
        elif gid == "SEMI_1":
            self.games["FINAL"].team_a = winner
        elif gid == "SEMI_2":
            self.games["FINAL"].team_b = winner
        elif gid == "FINAL":
            self.champion = winner

        # fill PLAYIN_C after A and B are done (order guarantees this)
        if self.cursor + 1 < len(self.order) and self.order[self.cursor + 1] == "PLAYIN_C":
            if self._loser_a is None or self._winner_b is None:
                raise RuntimeError("Play-In C dependencies missing.")
            self.games["PLAYIN_C"].team_a = self._loser_a
            self.games["PLAYIN_C"].team_b = self._winner_b

        self.cursor += 1
        return wrapped

    def report(self) -> Dict[str, Any]:
        return {
            "format": "8 teams | top 2 byes | seeds 3–6 play-in | seeds 7–8 eliminated",
            "eliminated_pre_playoffs": self.eliminated_pre,
            "eliminated_play_in": self.eliminated_playin,
            "games": self.results,
            "champion": self.champion,
        }


# ============================================================
# QUICK WIRING EXAMPLE (in app.py)
# ============================================================
#
# rosters = load_rosters_csv("data/rosters.csv")
# teams = sorted(rosters["team"].unique().tolist())
#
# season = SeasonRunner(rosters, teams)
#
# # click "Play Next Match"
# if season.has_next_match():
#     played = season.play_next_match()
#     st.json(played["result"])
#     st.table(season.standings())
#
# # after season complete:
# if season.season_complete():
#     final_standings = season.standings()
#     playoffs = PlayoffRunner(
#         rosters_df=rosters,
#         standings_sorted=final_standings,
#         carry_fatigue=True,
#         fatigue_by_team=season.fatigue_by_team,   # carry fatigue in
#     )
#
# # click "Play Next Playoff Match"
# if playoffs.has_next_match():
#     played_po = playoffs.play_next_match()
#     st.json(played_po["result"])
#
# if playoffs.champion:
#     st.success(f"Champion: {playoffs.champion}")
#
# ============================================================