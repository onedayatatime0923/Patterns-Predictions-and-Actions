import nflgame
from collections import defaultdict, deque

# --------- Config ----------
YEARS = [2019]       # can be a list, e.g., [2018, 2019]
WEEKS = range(1, 18) # regular season; adjust as needed
WINDOW = 10          # number of previous weeks to aggregate
TOP_N = 10           # how many players to print per week

# Which stats to roll up (keys must exist in nflgame PlayerStats)
STAT_KEYS = ["rushing_yds", "rushing_att", "rushing_tds"]
# --------------------------

def player_key(p):
    """
    Robust, stable key for a player across weeks.
    Falls back gracefully if some attrs aren't present in your nflgame build.
    """
    # Prefer gsis_id/playerid if present, else name+team
    pid = getattr(p, "gsis_id", None)
    if not pid:
        pid = getattr(p, "playerid", None)
    if not pid:
        pid = f"{getattr(p, 'name', str(p))}|{getattr(p, 'team', '')}"
    return pid

def player_name_team(p):
    name = getattr(p, "name", str(p))
    team = getattr(p, "team", "")
    return name, team

def sum_stats(deq):
    total = {k: 0 for k in STAT_KEYS}
    for s in deq:
        for k in STAT_KEYS:
            total[k] += s.get(k, 0)
    return total

# Rolling storage: pid -> deque of last WINDOW per-week dicts (excluding current week when reporting)
rolling = defaultdict(lambda: deque(maxlen=WINDOW))
# Keep a friendly map pid -> (name, team) updated as we see players
id_to_meta = {}

for year in YEARS:
    for week in WEEKS:
        # Gather games and stats for the current (year, week)
        games = nflgame.games(year, week=week)
        players = nflgame.combine_game_stats(games)

        # ---- REPORT FOR THIS WEEK USING *PRIOR* 10 WEEKS ----
        # Build rolling aggregates (from existing rolling deques that do NOT yet include this week)
        aggregates = []
        for pid, deq in rolling.items():
            if len(deq) == 0:
                continue
            totals = sum_stats(deq)
            # Only include players with nonzero yards (adjust as desired)
            if totals.get("rushing_yds", 0) > 0:
                name, team = id_to_meta.get(pid, ("<unknown>", ""))
                aggregates.append((pid, name, team, totals))
        print(aggregates)
        input()

        # Sort and print top N by rolling rushing yards
        aggregates.sort(key=lambda x: x[3]["rushing_yds"], reverse=True)
        print(f"\n==== {year} Week {week} — Top {TOP_N} rushers over PREVIOUS {WINDOW} weeks ====")
        for i, (_, name, team, t) in enumerate(aggregates[:TOP_N], 1):
            print(f"{i:>2}. {name} ({team}) — "
                  f"{t['rushing_att']} att, {t['rushing_yds']} yds, {t['rushing_tds']} TDs")

        # ---- NOW INGEST CURRENT WEEK INTO THE ROLLING WINDOW ----
        # For every player that appeared this week, append this week's stat line
        for p in players:
            pid = player_key(p)
            name, team = player_name_team(p)
            id_to_meta[pid] = (name, team)

            # Build a clean dict of this week's stats with 0 defaults
            stat_line = {k: int(getattr(p, k, 0) or 0) for k in STAT_KEYS}
            rolling[pid].append(stat_line)

        # Optional: pause like your original snippet
        # input()
