"""
Scrapes NCAA tournament bracket results from Sports-Reference.
Replaces Kaggle dependency for historical game results.

Each year yields ~67 games: 32(R64) + 16(R32) + 8(S16) + 4(E8) + 2(F4) + 1(NCG) + 4(FF play-in)
2020 had no tournament (COVID).
"""
import logging
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

_SR_BRACKET_URL = "https://www.sports-reference.com/cbb/postseason/men/{year}-ncaa.html"
_RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "sports_ref"
_DELAY = 4.0  # seconds between requests (SR rate limit)


def fetch_tournament_results(year: int, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch NCAA tournament bracket results for a given year.
    Returns DataFrame with columns:
        year, round, W_seed, W_team, W_score, L_seed, L_team, L_score
    Returns empty DataFrame if no tournament (e.g. 2020).
    """
    cache_path = _RAW_DIR / f"tourney_{year}.parquet"
    if cache_path.exists() and not force_refresh:
        return pd.read_parquet(cache_path)

    if year == 2020:
        log.info(f"  Skipping {year} — no NCAA tournament (COVID cancellation)")
        return pd.DataFrame()

    url = _SR_BRACKET_URL.format(year=year)
    log.info(f"Fetching tournament bracket for {year}: {url}")

    try:
        r = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (research project)"},
            timeout=20,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        log.error(f"  Failed to fetch {year} bracket: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(r.text, "html.parser")
    games = _parse_bracket(soup, year)

    if not games:
        log.warning(f"  No games parsed for {year} — bracket format may have changed")
        return pd.DataFrame()

    df = pd.DataFrame(games)
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    log.info(f"  Saved {len(df)} games → {cache_path.name}")
    return df


def fetch_all_tournament_results(
    years: list[int], force_refresh: bool = False
) -> pd.DataFrame:
    """Fetch and combine tournament results for multiple years."""
    frames = []
    for i, year in enumerate(years):
        df = fetch_tournament_results(year, force_refresh=force_refresh)
        if not df.empty:
            frames.append(df)
        if i < len(years) - 1:
            time.sleep(_DELAY)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _parse_bracket(soup: BeautifulSoup, year: int) -> list[dict]:
    """
    Page structure:
      - 4x div#bracket.team16  → regional brackets (R64, R32, S16, E8)
        each has 5 round divs; rounds 0-3 are real games, round[4] is an empty F4 placeholder
      - 1x div#bracket.team4   → Final Four bracket (F4, NCG)
        has 3 round divs; rounds 0-1 are real games, round[2] is an empty placeholder
    """
    all_brackets = soup.find_all("div", id="bracket")
    if not all_brackets:
        log.warning(f"  No #bracket divs found for {year}")
        return []

    games = []

    for bracket in all_brackets:
        classes = bracket.get("class") or []
        round_divs = bracket.find_all("div", class_="round", recursive=False)

        if "team16" in classes:
            # Regional bracket: rounds 0-3 = R64, R32, S16, E8
            regional_round_names = ["R64", "R32", "S16", "E8"]
            for round_idx, round_name in enumerate(regional_round_names):
                if round_idx >= len(round_divs):
                    break
                for game_div in round_divs[round_idx].find_all("div", recursive=False):
                    game = _parse_game(game_div, year, round_name)
                    if game:
                        games.append(game)

        elif "team4" in classes:
            # Final Four bracket: round 0 = F4, round 1 = NCG
            ff_round_names = ["F4", "NCG"]
            for round_idx, round_name in enumerate(ff_round_names):
                if round_idx >= len(round_divs):
                    break
                for game_div in round_divs[round_idx].find_all("div", recursive=False):
                    game = _parse_game(game_div, year, round_name)
                    if game:
                        games.append(game)

    return games


def _parse_game(game_div, year: int, round_name: str) -> dict | None:
    team_divs = game_div.find_all("div", recursive=False)
    if len(team_divs) < 2:
        return None

    winner = None
    loser = None
    for td in team_divs:
        classes = td.get("class") or []
        parsed = _parse_team_div(td)
        if parsed is None:
            continue
        if "winner" in classes:
            winner = parsed
        else:
            loser = parsed

    if winner is None or loser is None:
        return None

    return {
        "year": year,
        "round": round_name,
        "W_seed": winner["seed"],
        "W_team": winner["name"],
        "W_score": winner["score"],
        "L_seed": loser["seed"],
        "L_team": loser["name"],
        "L_score": loser["score"],
    }


def _parse_team_div(td) -> dict | None:
    try:
        seed_span = td.find("span")
        if not seed_span:
            return None
        seed_text = seed_span.text.strip()
        if not seed_text.isdigit():
            return None
        seed = int(seed_text)

        links = td.find_all("a")
        if len(links) < 2:
            return None

        name = links[0].text.strip()
        score_text = links[1].text.strip()
        score = int(score_text) if score_text.isdigit() else None

        if not name:
            return None

        return {"seed": seed, "name": name, "score": score}
    except (ValueError, AttributeError, IndexError):
        return None
