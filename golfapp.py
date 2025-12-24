import streamlit as st
import random
import itertools
import copy
import pandas as pd
import plotly.express as px
from dataclasses import dataclass
from typing import Optional, Dict, Any  

# ---------------------------
# CONFIG & STATE
# ---------------------------
if 'seed' not in st.session_state:
    st.session_state.seed = 42

st.set_page_config(page_title="Golf Tourney Sim", layout="wide")

# Default Players
DEFAULT_PLAYERS = {
    "Trent": 10, "Jordan": 10, "Aaron": 15, "Tony": 18,
    "Frank": 20, "Cameron": 25, "Carlos": 30, "Jeff": 30
}
VOTER_MODES = ["Rational", "Reverse", "Chaotic", "Random"]
HOLES_PER_MATCH = 9
MAX_STROKES_OVER_PAR = 3  # Triple bogey cap
SCRAMBLE_VARIANCE = 0.6   # dampens scramble chaos
HOLE_PARS = [4, 5, 4, 3, 5, 3, 4, 4, 4]

# ---------------------------
# LOGIC FUNCTIONS (Ported from your script)
# ---------------------------

# ---------------------------
# GAME THEORY CORE (Step 1)
# ---------------------------

class Agent:
    def __init__(self, name, risk_aversion=0.5, utility_weights=None):
        self.name = name
        self.risk_aversion = risk_aversion
        self.utility_weights = utility_weights or {
            "win_prob": 1.0,
            "margin": 0.2,
            "variance": 0.5
        }

@dataclass
class StrategyProfile:
    """
    Explicit utility weights for decision-making.
    Utility is computed over simulated matchup outcomes (per-hole tournament points).
    """
    win_weight: float = 1.0          # weight on win probability of the matchup
    point_margin_weight: float = 1.0 # weight on expected tournament point margin
    variance_weight: float = 0.5     # penalty weight on variance of margin
    base_risk_aversion: float = 0.5  # baseline risk aversion (0 = risk neutral, 1 = highly averse)

    # Score awareness (leading/trailing effects)
    lead_risk_shift: float = 0.3     # increase risk aversion when leading
    trail_risk_shift: float = 0.3    # decrease risk aversion when trailing


def tournament_points_from_hole_scorecard(hole_scorecard):
    """
    Compute per-hole tournament points from an existing hole_scorecard.
    Returns: (pts_a, pts_b) where pts_a + pts_b == number_of_holes
    """
    a, b = 0.0, 0.0
    for h in hole_scorecard:
        pa, pb = points_from_winner(h["hole_winner"])
        a += pa
        b += pb
    return a, b


def adjusted_risk_aversion(profile: StrategyProfile, score_diff: float) -> float:
    """
    score_diff: positive means Team A is leading, negative means Team A is trailing.
    """
    ra = profile.base_risk_aversion
    if score_diff > 0:
        return min(1.0, ra + profile.lead_risk_shift)
    if score_diff < 0:
        return max(0.0, ra - profile.trail_risk_shift)
    return ra


def utility_from_margins(margins, agent):
    """
    margins: list of (Team A final - Team B final)
    """
    if len(margins) == 0:
        return 0

    s = pd.Series(margins)
    win_prob = (s > 0).mean()
    exp_margin = s.mean()
    variance = s.var() if len(s) > 1 else 0

    return (
        agent.utility_weights["win_prob"] * win_prob +
        agent.utility_weights["margin"] * exp_margin -
        agent.risk_aversion * agent.utility_weights["variance"] * variance
    )
def evaluate_draft_pick(
    candidate,
    team_self,
    team_opp,
    handicaps,
    agent
):
    """
    Utility proxy for draft decisions (non-recursive).
    """

    # Build hypothetical teams
    sim_team_self = team_self + [candidate]

    # Simple team strength metrics
    self_strength = sum(handicaps[p] for p in sim_team_self)
    opp_strength = sum(handicaps[p] for p in team_opp) if team_opp else self_strength

    # Margin proxy (lower handicap total = stronger)
    margin = opp_strength - self_strength

    # Variance proxy: higher spread = riskier
    self_variance = pd.Series([handicaps[p] for p in sim_team_self]).var()
    self_variance = 0 if pd.isna(self_variance) else self_variance

    margins = [margin]  # single-sample proxy

    return utility_from_margins(margins, agent) - agent.risk_aversion * self_variance

def adjust_risk_aversion(base_risk, score_diff):
    """
    score_diff: positive means this team is leading
    """
    if score_diff > 0:
        # Leading → reduce variance exposure
        return min(1.0, base_risk + 0.3)
    elif score_diff < 0:
        # Trailing → seek variance
        return max(0.0, base_risk - 0.3)
    return base_risk

def evaluate_singles_pick(
    candidate,
    self_pool,
    opp_pool,
    handicaps,
    agent,
    score_diff
):
    """
    Utility proxy for singles draft decisions.
    score_diff: positive means agent's team is leading
    """

    # Adjust risk based on score
    agent.risk_aversion = adjust_risk_aversion(
        agent.risk_aversion,
        score_diff
    )

    # --- SAFE opponent average ---
    if len(opp_pool) > 0:
        opp_avg = sum(handicaps[p] for p in opp_pool) / len(opp_pool)
    else:
        # No opponents left → neutral baseline
        opp_avg = handicaps[candidate]

    # Expected margin proxy
    margin = opp_avg - handicaps[candidate]

    # Variance proxy (singles are volatile)
    variance = abs(handicaps[candidate] - opp_avg)

    margins = [margin]

    return utility_from_margins(margins, agent) - agent.risk_aversion * variance
def evaluate_pairing_utility(
    pair_self,
    pair_opp,
    handicaps,
    agent,
    match_type,
    score_diff,
    profile: Optional[StrategyProfile] = None,
    n_sims: int = 15,
    return_components: bool = False
):
    """
    Explicit utility for a matchup based on *per-hole tournament points*.

    Utility components:
      - win probability of the matchup (by tournament points)
      - expected tournament-point margin
      - variance penalty (risk-adjusted)

    score_diff: positive means Team A is leading; negative means Team A is trailing.
    """
    profile = profile or StrategyProfile()

    # Use the explicit profile to set agent risk aversion
    agent.risk_aversion = adjusted_risk_aversion(profile, score_diff)
    ra = agent.risk_aversion

    margins = []
    wins = 0

    for _ in range(n_sims):
        _, _, hole_scorecard, _ = play_match(pair_self, pair_opp, handicaps, match_type)
        tp_self, tp_opp = tournament_points_from_hole_scorecard(hole_scorecard)
        margin = tp_self - tp_opp
        margins.append(margin)
        if margin > 0:
            wins += 1

    if len(margins) == 0:
        if return_components:
            return 0.0, {"win_prob": 0.0, "exp_margin": 0.0, "variance": 0.0, "risk_aversion": ra}
        return 0.0

    s = pd.Series(margins)
    win_prob = wins / len(margins)
    exp_margin = float(s.mean())
    variance = float(s.var()) if len(s) > 1 else 0.0

    utility = (
        profile.win_weight * win_prob +
        profile.point_margin_weight * exp_margin -
        ra * profile.variance_weight * variance
    )

    if return_components:
        return utility, {
            "win_prob": win_prob,
            "exp_margin": exp_margin,
            "variance": variance,
            "risk_aversion": ra,
            "weights": {
                "win_weight": profile.win_weight,
                "point_margin_weight": profile.point_margin_weight,
                "variance_weight": profile.variance_weight
            }
        }

    return utility

def assign_voter_modes(players):
    return {p: random.choice(VOTER_MODES) for p in players}

def vote_for_captains(modes, h):
    votes = {p: 0 for p in h}
    names = list(h.keys())
    for voter, mode in modes.items():
        if mode == "Rational": picks = sorted(names, key=lambda x: (h[x], x))[:2]
        elif mode == "Reverse": picks = sorted(names, key=lambda x: (-h[x], x))[:2]
        elif mode == "Chaotic":
            s = sorted(names, key=lambda x: (h[x], x))
            picks = [s[0], s[-1]]
        else: picks = random.sample(names, 2)
        for p in picks: votes[p] += 1
    ranked = sorted(votes.items(), key=lambda x: (-x[1], h[x[0]], x[0]))
    return [ranked[0][0], ranked[1][0]], votes

def snake_team_draft(captains, handicaps):
    pool = [p for p in handicaps if p not in captains]

    team_a = [captains[0]]
    team_b = [captains[1]]

    agent_a = Agent(captains[0])
    agent_b = Agent(captains[1])

    expected_score_diff = handicaps[captains[1]] - handicaps[captains[0]]

    # Snake order: A, B, B, A, A, B
    order = ["A", "B", "B", "A", "A", "B"]

    for turn in order:
        if turn == "A":
            best_pick = None
            best_u = -float("inf")

            agent_a.risk_aversion = adjust_risk_aversion(
                agent_a.risk_aversion,
                expected_score_diff
            )

            for candidate in pool:
                u = evaluate_draft_pick(
                    candidate,
                    team_a,
                    team_b,
                    handicaps,
                    agent_a
                )
                if u > best_u:
                    best_u = u
                    best_pick = candidate

            team_a.append(best_pick)
            pool.remove(best_pick)

        else:
            best_pick = None
            best_u = -float("inf")

            agent_b.risk_aversion = adjust_risk_aversion(
                agent_b.risk_aversion,
                -expected_score_diff
            )
            
            for candidate in pool:
                u = evaluate_draft_pick(
                    candidate,
                    team_b,
                    team_a,
                    handicaps,
                    agent_b
                )
                if u > best_u:
                    best_u = u
                    best_pick = candidate

            team_b.append(best_pick)
            pool.remove(best_pick)

    return [team_a, team_b], pool


def simulate_hole_score_scramble(pair, handicaps):
    base = min(handicaps[pair[0]], handicaps[pair[1]])
    variation = random.gauss(0, 1.5)
    return base + variation

def simulate_hole_score_alt(pair, handicaps):
    if len(pair) == 1: return handicaps[pair[0]]
    return (handicaps[pair[0]] + handicaps[pair[1]]) / 2

def generate_pars(num_holes):
    """
    Generate a realistic par distriubution for a 9-hole side. 
    Default: 2x Par 3 5x Par 4, 2x Par 5
    """
    if num_holes == 9:
        pars = [3,3,4,4,4,4,4,5,5]
        random.shuffle(pars)
        return pars
    pars = []
    n3 = round(num_holes * (2/9))
    n5 = round(num_holes * (2/9))
    n4 = max(0, num_holes - n3 - n5)
    pars = ([3] * n3) + ([4] * n4) + ([5] * n5)
    random.shuffle(pars)
    return pars[ :num_holes]

def play_match(pair_a, pair_b, handicaps, match_type="scramble"):
    pars = generate_pars(HOLES_PER_MATCH)

    match_score = 0  # + = Team A up, - = Team B up
    hole_scorecard = []

    for i, par in enumerate(pars, start=1):

        # --- Simulate strokes ---
        if match_type == "scramble":
            perf_a = simulate_hole_score_scramble(pair_a, handicaps)
            perf_b = simulate_hole_score_scramble(pair_b, handicaps)
        else:
            perf_a = simulate_hole_score_alt(pair_a, handicaps)
            perf_b = simulate_hole_score_alt(pair_b, handicaps)

        # Convert performance → strokes relative to par
        strokes_a = round(par + (perf_a - 10) * 0.3)
        strokes_b = round(par + (perf_b - 10) * 0.3)

        # Floor & ceiling (triple bogey cap)
        strokes_a = max(par - 2, min(par + MAX_STROKES_OVER_PAR, strokes_a))
        strokes_b = max(par - 2, min(par + MAX_STROKES_OVER_PAR, strokes_b))

        # --- Determine hole result ---
        if strokes_a < strokes_b:
            hole_winner = "A"
            match_score += 1
        elif strokes_b < strokes_a:
            hole_winner = "B"
            match_score -= 1
        else:
            hole_winner = "HALVE"

        hole_scorecard.append({
            "hole": i,
            "par": par,
            "strokes_a": strokes_a,
            "strokes_b": strokes_b,
            "hole_winner": hole_winner,
            "match_score": match_score
        })
    if match_score > 0:
        pts_a, pts_b = 1,0
    elif match_score < 0:
        pts_a, pts_b = 0,1
    else:
        pts_a, pts_b = 0.5, 0.5

    return pts_a, pts_b, hole_scorecard, None

def points_from_winner(w):
    if w == "A": 
        return 1, 0.0
    if w == "B": 
        return 0.0, 1.0
    return 0.5, 0.5

def sum_round_points(round_holes):
    a, b = 0.0, 0.0
    for match in round_holes:
        for h in match["hole_scorecard"]:
            pa, pb = points_from_winner(h["hole_winner"])
            a += pa
            b += pb
    return a, b
def compute_round_tournament_points(round_holes, expected_matches=None, expected_holes=HOLES_PER_MATCH):
    pts_a, pts_b = sum_round_points(round_holes)
    if expected_matches is not None:
        expected_total = expected_matches * expected_holes
        actual_total = pts_a + pts_b
        if abs(actual_total - expected_total) > 1e-9:
            st.warning(
                f"⚠️ Tournament points validation failed: expected {expected_total} total points, "
                f"got {actual_total:.1f}. (Matches={expected_matches}, holes/match={expected_holes})"
            )

    return pts_a, pts_b
def fmt_pair(p):
    if isinstance(p, (list, tuple)): return " & ".join(p)
    return str(p)
def format_debug_df(df):
    """
    Cleans and formats decision-debug tables for readability.
    """
    df = df.copy()

    # Rename columns to human-friendly labels
    rename_map = {
        "A_pair": "Team A Pair",
        "B_pair": "Team B Pair",
        "worst_case_B_response": "Worst B Response",
        "utility(worst_case)": "Decision Score (Worst Case)",
        "utility": "Decision Score",
        "win_prob": "Win %",
        "exp_margin": "Expected Margin",
        "variance": "Variance",
        "risk_aversion": "Risk Aversion"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Convert win probability to %
    if "Win %" in df.columns:
        df["Win %"] = (df["Win %"] * 100).round(1)

    # Round numeric columns
    for col in df.select_dtypes(include="number").columns:
        if col != "Win %":
            df[col] = df[col].round(2)

    return df

def snake_draft_singles_matchups(team_a, team_b, handicaps, leader="A"):
    """
    Captains snake-draft head-to-head singles matchups.
    Returns list of tuples: [(player_a, player_b), ...] length 4.

    team_a, team_b: lists of 4 players each
    leader: "A" or "B" (leader after Round 2)
    """

    pool_a = team_a.copy()
    pool_b = team_b.copy()

    matchups = []

    def best_player(pool):
        """Simple rational pick: lowest handicap available."""
        return min(pool, key=lambda p: handicaps[p])

    # Snake order for 4 matches
    # Leader initiates matches 1 and 3
    initiators = [
        leader,
        "B" if leader == "A" else "A",
        leader,
        "B" if leader == "A" else "A"
    ]

    for initiator in initiators:
        if initiator == "A":
            a_pick = best_player(pool_a)
            pool_a.remove(a_pick)

            b_pick = best_player(pool_b)
            pool_b.remove(b_pick)
        else:
            b_pick = best_player(pool_b)
            pool_b.remove(b_pick)

            a_pick = best_player(pool_a)
            pool_a.remove(a_pick)

        matchups.append((a_pick, b_pick))

    return matchups
def simulate_round3_singles_snake(team_a, team_b, handicaps, leader="A"):
    """
    Runs Round 3 singles using snake-drafted head-to-head matchups.

    Returns:
      r3_results: list of tuples (Player A, Player B, pts_a, pts_b)
      total_pts_a: float
      total_pts_b: float
      matchups: list of tuples (Player A, Player B)
      r3_holes: list of dicts, each dict has:
          {"pair_a": (a,), "pair_b": (b,), "hole_scorecard": [...]}
    """
    matchups = snake_draft_singles_matchups(team_a, team_b, handicaps, leader)

    total_a, total_b = 0.0, 0.0
    r3_results = []
    r3_holes = []
    
    for a_player, b_player in matchups:
        pts_a, pts_b, hole_scorecard, _ = play_match(
            (a_player,), (b_player,), handicaps, match_type="singles"
        )
        total_a += pts_a
        total_b += pts_b
        r3_results.append((a_player, b_player, pts_a, pts_b))
        r3_holes.append({
            "pair_a": (a_player,), "pair_b": (b_player,),
            "hole_scorecard": hole_scorecard
    
        })

    return r3_results, total_a, total_b, matchups, r3_holes
def total_points_from_round_holes(round_holes):
    a_total, b_total = 0, 0
    for match in round_holes:
        for h in match["hole_scorecard"]:
            pa, pb = points_from_winner(h["hole_winner"])
            a_total += pa
            b_total += pb

    return a_total, b_total

def total_points_from_r3_results(r3_results):
    a_total, b_total = 0.0, 0.0
    for _, _, _, _, hole_scorecard in r3_results:
        for h in hole_scorecard:
            pa, pb = points_from_winner(h["hole_winner"])
            a_total += pa
            b_total += pb
    return a_total, b_total



def play_locked_matchups(matchups, handicaps, match_type):
    total_a, total_b = 0, 0
    all_hole_cards = []
    for (pair_a, pair_b) in matchups:
        pts_a, pts_b, hole_scorecard, _ = play_match(
    pair_a, pair_b, handicaps, match_type)
        total_a += pts_a
        total_b += pts_b
        all_hole_cards.append({
            "pair_a": pair_a, "pair_b": pair_b,
            "hole_scorecard": hole_scorecard
        })
    return total_a, total_b, all_hole_cards
def solve_round_pairings(
    team_a,
    team_b,
    handicaps,
    match_type="scramble",
    initiative="A",
    score_diff=0,
    profile_a: Optional[StrategyProfile] = None,
    profile_b: Optional[StrategyProfile] = None,
    decision_debug: Optional[Dict[str, Any]] = None
):
    """
    Sequential, strategic pairing selection.
    """

    agent_a = Agent("Team A")
    agent_b = Agent("Team B")

    profile_a = profile_a or StrategyProfile()
    profile_b = profile_b or StrategyProfile()

    if decision_debug is not None:
        decision_debug.clear()
        decision_debug["match_type"] = match_type
        decision_debug["initiative"] = initiative
        decision_debug["score_diff"] = score_diff

    # --- Generate possible pairs ---
    a_pairs = list(itertools.combinations(team_a, 2))
    b_pairs = list(itertools.combinations(team_b, 2))

    matchups = []
    total_a, total_b = 0, 0

    # ---- FIRST MATCH ----
    if initiative == "A":
        best_pair_a = None
        best_u = -float("inf")

        candidate_rows = []  # used only if decision_debug is not None

        for pair_a in a_pairs:
            # Opponent best response (min utility)
            worst_case_u = float("inf")
            worst_pair_b = None
            worst_components = None

            for pair_b in b_pairs:
                u, comps = evaluate_pairing_utility(
                    pair_a, pair_b, handicaps,
                    agent_a, match_type, score_diff,
                    profile=profile_a,
                    return_components=True
                )
                if u < worst_case_u:
                    worst_case_u = u
                    worst_pair_b = pair_b
                    worst_components = comps

            if decision_debug is not None:
                candidate_rows.append({
                    "A_pair": " & ".join(pair_a),
                    "worst_case_B_response": " & ".join(worst_pair_b) if worst_pair_b else None,
                    "utility(worst_case)": worst_case_u,
                    "win_prob": worst_components["win_prob"] if worst_components else None,
                    "exp_margin": worst_components["exp_margin"] if worst_components else None,
                    "variance": worst_components["variance"] if worst_components else None,
                    "risk_aversion": worst_components["risk_aversion"] if worst_components else None
                })

            if worst_case_u > best_u:
                best_u = worst_case_u
                best_pair_a = pair_a

        # Opponent chooses best counter (max utility from B perspective vs chosen A pair)
        b_choice_rows = []
        best_pair_b = None
        best_b_u = -float("inf")

        for pair_b in b_pairs:
            u, comps = evaluate_pairing_utility(
                best_pair_a, pair_b, handicaps,
                agent_b, match_type, -score_diff,
                profile=profile_b,
                return_components=True
            )

            if decision_debug is not None:
                b_choice_rows.append({
                    "B_pair": " & ".join(pair_b),
                    "utility": u,
                    "win_prob": comps["win_prob"],
                    "exp_margin": comps["exp_margin"],
                    "variance": comps["variance"],
                    "risk_aversion": comps["risk_aversion"]
                })

            if u > best_b_u:
                best_b_u = u
                best_pair_b = pair_b

        if decision_debug is not None:
            top_a = sorted(candidate_rows, key=lambda r: r["utility(worst_case)"], reverse=True)[:3]
            top_b = sorted(b_choice_rows, key=lambda r: r["utility"], reverse=True)[:3]
            decision_debug["first_match"] = {
                "A_candidates_top3": top_a,
                "A_chosen": " & ".join(best_pair_a),
                "B_best_response_top3": top_b,
                "B_chosen": " & ".join(best_pair_b)
            }

    else:
        # Mirror logic if B has initiative (no debug instrumentation yet)
        best_pair_b = None
        best_u = -float("inf")

        for pair_b in b_pairs:
            worst_case_u = float("inf")
            for pair_a in a_pairs:
                u = evaluate_pairing_utility(
                    pair_b, pair_a, handicaps,
                    agent_b, match_type, -score_diff,
                    profile=profile_b
                )
                worst_case_u = min(worst_case_u, u)

            if worst_case_u > best_u:
                best_u = worst_case_u
                best_pair_b = pair_b

        best_pair_a = max(
            a_pairs,
            key=lambda p: evaluate_pairing_utility(
                p, best_pair_b, handicaps,
                agent_a, match_type, score_diff,
                profile=profile_a
            )
        )

    matchups.append((best_pair_a, best_pair_b))

    # ---- SECOND MATCH (remaining players) ----
    rem_a = [p for p in team_a if p not in best_pair_a]
    rem_b = [p for p in team_b if p not in best_pair_b]

    pair_a2 = tuple(rem_a)
    pair_b2 = tuple(rem_b)

    matchups.append((pair_a2, pair_b2))

    # ---- Play matches ----
    all_hole_scores = []
    for pair_a, pair_b in matchups:
        pts_a, pts_b, hole_scorecard, _ = play_match(
            pair_a, pair_b, handicaps, match_type
        )
        total_a += pts_a
        total_b += pts_b
        all_hole_scores.append({
            "pair_a": pair_a,
            "pair_b": pair_b,
            "hole_scorecard": hole_scorecard
        })

    return matchups, total_a, total_b, all_hole_scores



def simulate_singles_draft(team_a_players, team_b_players, handicaps, winning_captain_team):
    drafted_a, drafted_b = [], []
    pool_a = copy.deepcopy(team_a_players)
    pool_b = copy.deepcopy(team_b_players)
    agent_a = Agent("Team A")
    agent_b = Agent("Team B")

    if winning_captain_team == "A":
        score_diff = 1  # A is leading
    elif winning_captain_team == "B":
        score_diff = -1 # B is leading
    # Standard order
    draft_order_index = [0,1,1,0,0,1,1,0]
    # If B is winning, B picks second in first slot? (Logic from original script: index flip)
    if winning_captain_team == "B":
        draft_order_index[0] = 1
        draft_order_index[1] = 0

    for index in draft_order_index:
        if index == 0 and pool_a:
            pick = None
            best_u = -float("inf")
            for candidate in pool_a:
                u = evaluate_singles_pick(
                    candidate,
                    pool_a,
                    pool_b,
                    handicaps,
                    agent_a,
                    score_diff
                )
                if u > best_u:
                    best_u = u
                    pick = candidate
        
            drafted_a.append(pick)
            pool_a.remove(pick)
        elif index == 1 and pool_b:
            pick = None
            best_u = -float("inf")
            for candidate in pool_b:
                u = evaluate_singles_pick(
                    candidate,
                    pool_b,
                    pool_a,
                    handicaps,
                    agent_b,
                    -score_diff
                )
                if u > best_u:
                    best_u = u
                    pick = candidate
            drafted_b.append(pick)
            pool_b.remove(pick)

    total_r3_points_a, total_r3_points_b = 0, 0
    results = []
    for pa, pb in zip(drafted_a, drafted_b):
        p_a, p_b, _, _ = play_match((pa,), (pb,), handicaps, match_type="singles")
        total_r3_points_a += p_a
        total_r3_points_b += p_b
        results.append((pa, pb, p_a, p_b))

    return results, total_r3_points_a, total_r3_points_b

# ---------------------------
# WRAPPER FOR FULL TOURNAMENT
# ---------------------------
def run_full_tournament(handicaps):
    # 1. Vote
    modes = assign_voter_modes(list(handicaps.keys()))
    captains, votes = vote_for_captains(modes, handicaps)
    # TEMP: create agents for captains (not used yet)
    agent_a = Agent(captains[0])
    agent_b = Agent(captains[1])

    # 2. Draft
    teams, _ = snake_team_draft(captains, handicaps)
    team_a, team_b = teams[0], teams[1]

        # Utility profiles (hard-coded for now; UI dials come next)
    profile_a = StrategyProfile(
        win_weight=1.0,
        point_margin_weight=1.0,
        variance_weight=0.6,
        base_risk_aversion=0.5,
        lead_risk_shift=0.3,
        trail_risk_shift=0.3
    )
    profile_b = StrategyProfile(
        win_weight=1.0,
        point_margin_weight=1.0,
        variance_weight=0.6,
        base_risk_aversion=0.5,
        lead_risk_shift=0.3,
        trail_risk_shift=0.3
    )
    
    # 3. Round 1 (Scramble) — pairing selection uses explicit utility
    decision_debug_r1 = {}
    r1_matchups, r1_pts_a, r1_pts_b, r1_holes = solve_round_pairings(
        team_a, team_b, handicaps,
        match_type="scramble",
        initiative="A",
        score_diff=0,
        profile_a=profile_a,
        profile_b=profile_b,
        decision_debug=decision_debug_r1
    )


    # 4. Round 2 (Alt)
    r2_pts_a, r2_pts_b, r2_holes = play_locked_matchups(
    r1_matchups, handicaps,
    match_type="alt",
)


    # 5. Check winner for R3 draft advantage
    curr_a = r1_pts_a + r2_pts_a
    curr_b = r1_pts_b + r2_pts_b
    leader = "A" if curr_a >= curr_b else "B"

    # 6. Round 3 (Singles)
    r3_results, r3_pts_a, r3_pts_b, r3_matchups, r3_holes = simulate_round3_singles_snake(
        team_a, team_b, handicaps, leader
    )

    total_a = curr_a + r3_pts_a
    total_b = curr_b + r3_pts_b
    
        # --- Tournament Points (per-hole, must sum to 72 overall) ---
    r1_tp_a, r1_tp_b = compute_round_tournament_points(r1_holes, expected_matches=2)
    r2_tp_a, r2_tp_b = compute_round_tournament_points(r2_holes, expected_matches=2)
    r3_tp_a, r3_tp_b = compute_round_tournament_points(r3_holes, expected_matches=4)

    tournament_pts_a = r1_tp_a + r2_tp_a + r3_tp_a
    tournament_pts_b = r1_tp_b + r2_tp_b + r3_tp_b


    return {
        "team_a": team_a, "team_b": team_b,
        "captains": captains,
        "r1": (r1_matchups, r1_pts_a, r1_pts_b, r1_holes),
        "r2": (r1_matchups, r2_pts_a, r2_pts_b, r2_holes),
        "r3": (r3_results, r3_pts_a, r3_pts_b, r3_matchups, r3_holes),
        "tournament_points": (tournament_pts_a, tournament_pts_b),
        "decision_debug": {
            "r1": decision_debug_r1
        },
        "final": (total_a, total_b),
        "winner": "Team A" if total_a > total_b else "Team B"
        
    }

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("⛳ Golf Tournament Simulator")

# --- SIDEBAR CONFIG ---
st.sidebar.header("Configuration")
with st.sidebar.form("config_form"):
    st.write("### Edit Handicaps")
    custom_handicaps = {}
    for p, h in DEFAULT_PLAYERS.items():
        custom_handicaps[p] = st.number_input(f"{p}", value=h, step=1)
    
    run_btn = st.form_submit_button("Run Single Simulation")

st.sidebar.markdown("---")
st.sidebar.write("### Monte Carlo Mode")
num_sims = st.sidebar.number_input("Simulations", value=100, step=50)
run_multi_btn = st.sidebar.button("Run Monte Carlo")

# --- MAIN DISPLAY ---
tab1, tab2 = st.tabs(["Single Match View", "Monte Carlo Analysis"])

# TAB 1: SINGLE MATCH
with tab1:
    if run_btn:
        random.seed(random.randint(0, 10000)) # New seed every click
        data = run_full_tournament(custom_handicaps)
        
        r1_matchups, r1_pts_a, r1_pts_b, r1_holes = data['r1']
        r2_matches, r2_pts_a, r2_pts_b, r2_holes = data['r2']
        r3_res, r3a, r3b, r3_matchups, r3_holes = data['r3']
        r1_score_lookup = {}
        for match in r1_holes:
            pair_a = match["pair_a"]
            pair_b = match["pair_b"]

            final_match_score = match["hole_scorecard"][-1]["match_score"]

            r1_score_lookup[pair_a] = final_match_score
            r1_score_lookup[pair_b] = -final_match_score
        # Display Teams
        st.subheader(f"Captains: {data['captains'][0]} (Team A) vs {data['captains'][1]} (Team B)")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Team A**\n\n" + ", ".join(data['team_a']))
        with col2:
            st.success(f"**Team B**\n\n" + ", ".join(data['team_b']))


        # Round 1
        st.subheader("Round 1 – Scramble")
        # --- Decision Debug (Round 1) ---
        dbg = data.get("decision_debug", {}).get("r1", {})
        if dbg.get("first_match"):
            with st.expander("Decision Debug: Round 1 Pairing Selection (Match 1)"):
                fm = dbg["first_match"]

                st.markdown(
                    f"**Initiative:** {dbg.get('initiative')}  \n"
                    f"**Match type:** {dbg.get('match_type')}  \n"
                    f"**Score diff entering pairing:** {dbg.get('score_diff')}"
                )
                with st.popover("What is Decision Score?"):
                    st.markdown(
                        "- **Decision Score** is the model’s ranking for pairing options (higher = better).\n"
                        "- It’s based on simulated outcomes: **win chance** and **expected point margin**, with a penalty for **risk/volatility**.\n"
                        "- When a team is ahead, the model typically prefers **lower-risk** options; when behind, it tolerates **more risk**."
                    )
 
                st.markdown("#### Team A: Top 3 candidate pairs (maximin / worst-case utility)")
                df_a = format_debug_df(pd.DataFrame(fm["A_candidates_top3"]))
                cols_a = ["Team A Pair", "Worst B Response", "Decision Score (Worst Case)", "Win %", "Expected Margin", "Variance"]
                cols_a = [c for c in cols_a if c in df_a.columns]
                st.dataframe(df_a[cols_a], hide_index=True, use_container_width=True)
                st.success(f"Chosen Team A pair: {fm['A_chosen']}")

                st.markdown("#### Team B: Top 3 responses vs chosen Team A pair")
                df_b = format_debug_df(pd.DataFrame(fm["B_best_response_top3"]))

                cols_b = ["Team B Pair", "Decision Score", "Win %", "Expected Margin", "Variance"]
                cols_b = [c for c in cols_b if c in df_b.columns]
                st.dataframe(df_b[cols_b], hide_index=True, use_container_width=True)
                st.info(f"Chosen Team B pair: {fm['B_chosen']}")

                st.caption(
                    "How to read this: When picking pairings, captains weigh win odds and scoring potential. "
                    "If they’re ahead, they play it safe; if they’re behind, they’re willing to take risks."
                )

        for match_idx, match in enumerate(r1_holes):
            pair_a = match["pair_a"]
            pair_b = match["pair_b"]
            hole_scorecard = match["hole_scorecard"]
            st.markdown(
                f"**Match {match_idx + 1}: "
                f"{pair_a[0]} & {pair_a[1]} vs {pair_b[0]} & {pair_b[1]}**"
            )

            rows = []
            match_pts_a = 0.0
            match_pts_b = 0.0

            for h in hole_scorecard:
                pa, pb = points_from_winner(h["hole_winner"])
                match_pts_a += pa
                match_pts_b += pb
                rows.append({
                    "Hole": h["hole"],
                    "Par": h["par"],
                    f"({pair_a[0]} & {pair_a[1]})": h["strokes_a"],
                    f"({pair_b[0]} & {pair_b[1]})": h["strokes_b"],
                    "Result:": (
                        "½" if h["hole_winner"] == "HALVE"
                        else f"Team {h['hole_winner']}"),
                    
                    "Match": (
                            "AS" if h["match_score"] == 0
                            else f"A +{h['match_score']}" if h["match_score"] > 0
                            else f"B +{abs(h['match_score'])}"
                    )
            
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, hide_index=True, use_container_width=True)

        # Round 2
        st.markdown("### Round 2: Alt Shot")
        
        r2_data = []
        for (pair_a, pair_b) in r2_matches:
            r2_data.append({"Team A Pair": " & ".join(pair_a), "Team B Pair": " & ".join(pair_b)})
        st.dataframe(pd.DataFrame(r2_data), hide_index=True, use_container_width=True)

        total_score_a = r1_pts_a + r2_pts_a
        total_score_b = r1_pts_b + r2_pts_b
        st.caption(f"Score after R2: Team A {total_score_a} - {total_score_b} Team B")
        
        st.markdown("#### Round 2 Hole-by-Hole Scores")

        for match_idx, match in enumerate(r2_holes):
            pair_a = match["pair_a"]
            pair_b = match["pair_b"]
            hole_scorecard = match["hole_scorecard"]

            st.markdown(
                f"**Match {match_idx + 1}: "
                f"{pair_a[0]} & {pair_a[1]} vs {pair_b[0]} & {pair_b[1]}**"
            )
            col_a = f"{pair_a[0]} & {pair_a[1]}"
            col_b = f"{pair_b[0]} & {pair_b[1]}"
            df = pd.DataFrame(hole_scorecard)
            df = df.rename(columns={
                "hole": "Hole",
                "par": "Par",
                "strokes_a": col_a,
                "strokes_b": col_b,
                "hole_winner": "Result",
                "match_score": "Match"
            })

            def format_result(x):
                if x == "HALVE": return "½"
                if x == "A": return "A"
                if x == "B": return "B"
                return x
            df["Result"] = df["Result"].map({"A": "A", "B": "B", "HALVE": "½"}).fillna(df["Result"])

            def format_match(ms):
                if ms == 0: return "AS"
                if ms > 0: return f"A +{ms}"
                return f"B +{abs(ms)}"
            df["Match"] = df["Match"].apply(format_match)
            st.dataframe(
                df[["Hole", "Par", col_a, col_b,
                    "Result",
                    "Match"]],
                hide_index=True,
                use_container_width=True
            )
        
        # Round 3
        st.markdown("### Round 3: Singles")
        st.markdown("### Drafted Round 3 Matchups")
        st.dataframe(
            pd.DataFrame(
                [{"Match": i + 1, "Team A": a, "Team B": b} for i, (a, b) in enumerate(r3_matchups)]
            ),
            hide_index=True,
            use_container_width=True
        )
        st.markdown("### Round 3 Hole-by-Hole Scores")
        for i, match in enumerate(r3_holes):
            a_name = match["pair_a"][0]
            b_name = match["pair_b"][0] # Singles, so only one name per pair
            st.markdown(f"**Match {i + 1}: {a_name} vs {b_name}**")
            df = pd.DataFrame(match["hole_scorecard"]).rename(columns={
                "hole": "Hole",
                "par": "Par",
                "strokes_a": a_name,
                "strokes_b": b_name,
                "hole_winner": "Result",
                "match_score": "Match"
            })

            df["Result"] = df["Result"].map({"A": "A", "B": "B", "HALVE": "½"}).fillna(df["Result"])

            def format_match(ms):
                if ms == 0: return "AS"
                if ms > 0: return f"A +{ms}"
                return f"B +{abs(ms)}"
            df["Match"] = df["Match"].apply(format_match)

            st.dataframe(
                df[["Hole", "Par", a_name, b_name, "Result", "Match"]],
                hide_index=True,
                use_container_width=True
            )
       
        # Final
        fa, fb = data['final']
        tp_a, tp_b = data["tournament_points"]
        
        
        st.divider()
        st.subheader("Final Scores")
        m1, m2, m3 = st.columns(3)
        m1.metric("Team A Final", tp_a, f"+{fa}")
        m2.metric("Team B Final", tp_b, f"+{fb}")
        m3.write(f"### Winner: {data['winner']}")

# TAB 2: MONTE CARLO
with tab2:
    if run_multi_btn:
        st.write(f"Running {num_sims} simulations...")
        progress_bar = st.progress(0)
        
        results = []
        winners = []
        margin = []

        for i in range(num_sims):
            # No fixed seed here, we want variance
            res = run_full_tournament(custom_handicaps)
            diff = res['final'][0] - res['final'][1] # A - B
            results.append(res)
            winners.append(res['winner'])
            margin.append(diff)
            progress_bar.progress((i + 1) / num_sims)
        
        # Analytics
        df_res = pd.DataFrame({"Winner": winners, "Margin (A - B)": margin})
        
        st.subheader("Win Distribution")
        fig = px.histogram(df_res, x="Margin (A - B)", color="Winner", nbins=20, 
                           title="Distribution of Point Margins (Positive = A Wins)")
        st.plotly_chart(fig, use_container_width=True)
        
        win_counts = df_res['Winner'].value_counts()
        st.write("### Win Counts")
        st.bar_chart(win_counts)

        st.dataframe(win_counts)