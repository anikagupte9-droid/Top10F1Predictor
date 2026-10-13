import argparse
import time
from pathlib import Path

import pandas as pd
import requests


BASE_URL = "https://api.openf1.org/v1"
DATA_DIR = Path("data")

REQUIRED_SESSIONS = [
    "Practice 1",
    "Practice 2",
    "Practice 3",
    "Qualifying",
    "Race",
]


def get(endpoint, params=None, retries=5):
    """GET request with basic retry handling for rate limits."""

    url = f"{BASE_URL}/{endpoint}"

    for attempt in range(retries):
        response = requests.get(
            url,
            params=params,
            timeout=30,
        )

        if response.status_code == 429:
            wait_time = 5 * (attempt + 1)
            print(
                f"  Rate limited. Waiting {wait_time}s "
                f"before retry..."
            )
            time.sleep(wait_time)
            continue

        response.raise_for_status()
        return response.json()

    raise RuntimeError(
        f"OpenF1 rate limit persisted after {retries} attempts."
    )


def get_race_sessions(season):
    """
    Get all normal Race sessions for a season.

    We filter by session_name == 'Race' because sprint sessions
    can have session_type == 'Race' in the OpenF1 API.
    """

    data = get(
        "sessions",
        params={"year": season}
    )

    sessions = pd.DataFrame(data)

    if sessions.empty:
        return sessions

    races = sessions[
        sessions["session_name"] == "Race"
    ].copy()

    # One normal race session per meeting.
    races = races.drop_duplicates(
        subset=["meeting_key"]
    )

    races = races.sort_values(
        "date_start"
    ).reset_index(drop=True)

    return races


def get_session_map(season, meeting_key):
    """Return required sessions for one Grand Prix meeting."""

    data = get(
        "sessions",
        params={
            "year": season,
            "meeting_key": meeting_key,
        }
    )

    sessions = {}

    for session in data:
        name = session.get("session_name")

        if name in REQUIRED_SESSIONS:
            sessions[name] = session

    return sessions


def get_session_results(session_key):
    """Fetch driver results for one session."""

    data = get(
        "session_result",
        params={"session_key": session_key}
    )

    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)


def build_race_dataset(season, race):
    """Build one driver's dataset for one Grand Prix."""

    meeting_key = race["meeting_key"]
    race_name = race["location"]

    print(
        f"\nProcessing {race_name} {season} - Race"
    )

    sessions = get_session_map(
        season,
        meeting_key
    )

    # We require all five sessions.
    missing = [
        name
        for name in REQUIRED_SESSIONS
        if name not in sessions
    ]

    if missing:
        print(
            f"  Missing {', '.join(missing)} — skipping race"
        )
        return None

    results = {}

    for session_name in REQUIRED_SESSIONS:

        print(f"  Fetching {session_name}...")

        session_key = sessions[session_name]["session_key"]

        df = get_session_results(session_key)

        if df.empty:
            print(
                f"  No results for {session_name} — "
                f"skipping race"
            )
            return None

        results[session_name] = df[
            ["driver_number", "position"]
        ].copy()

        results[session_name] = results[
            session_name
        ].rename(
            columns={
                "position": f"{session_name}_Position"
            }
        )

        time.sleep(0.5)

    # Start with FP1 drivers.
    # Start with drivers who actually participated in the race.
# Practice-only/reserve drivers are therefore excluded.
    merged = results["Race"]

    for session_name in [
        "Practice 1",
        "Practice 2",
        "Practice 3",
        "Qualifying",
   ]:
        merged = merged.merge(
            results[session_name],
            on="driver_number",
            how="left",
        )

    # Rename columns to our ML feature names.
    merged = merged.rename(
        columns={
            "Practice 1_Position": "FP1_Position",
            "Practice 2_Position": "FP2_Position",
            "Practice 3_Position": "FP3_Position",
            "Qualifying_Position": "Qualifying_Position",
            "Race_Position": "Race_Position",
        }
    )

    merged["Season"] = season
    merged["Race"] = race_name

    # Top 10 target.
    merged["Top10"] = (
        merged["Race_Position"] <= 10
    ).astype("Int64")

    # Keep consistent column order.
    columns = [
        "Season",
        "Race",
        "driver_number",
        "FP1_Position",
        "FP2_Position",
        "FP3_Position",
        "Qualifying_Position",
        "Race_Position",
        "Top10",
    ]

    return merged[columns]


def build_season_dataset(season):
    """Build the dataset for one season."""

    print("\n" + "=" * 60)
    print(f"BUILDING {season}")
    print("=" * 60)

    races = get_race_sessions(season)

    print(
        f"\nFound {len(races)} normal race weekends "
        f"for {season}"
    )

    season_data = []

    for _, race in races.iterrows():

        try:
            race_df = build_race_dataset(
                season,
                race
            )

            if race_df is not None:
                season_data.append(race_df)

        except requests.RequestException as e:
            print(
                f"  Request failed: {e}"
            )
            print("  Skipping this race.")

        except Exception as e:
            print(
                f"  Unexpected error: {e}"
            )
            print("  Skipping this race.")

    if not season_data:
        print(f"\nNo usable data collected for {season}.")
        return pd.DataFrame()

    df = pd.concat(
        season_data,
        ignore_index=True
    )

    # Race sessions were collected chronologically,
    # so preserve that order.
    df = df.reset_index(drop=True)

    season_path = DATA_DIR / f"f1_{season}.csv"

    df.to_csv(
        season_path,
        index=False
    )

    print(
        f"\n{season} complete: "
        f"{len(df)} rows"
    )

    print(
        f"Saved to: {season_path}"
    )

    return df


def combine_seasons(seasons):
    """Combine all successfully collected seasons."""

    datasets = []

    for season in seasons:

        path = DATA_DIR / f"f1_{season}.csv"

        if path.exists():
            df = pd.read_csv(path)
            datasets.append(df)

    if not datasets:
        raise RuntimeError(
            "No season datasets were found."
        )

    combined = pd.concat(
        datasets,
        ignore_index=True
    )

    combined = combined.reset_index(drop=True)

    output_path = DATA_DIR / "f1_training_data.csv"

    combined.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("COMBINED DATASET COMPLETE")
    print("=" * 60)

    print(f"Rows: {len(combined)}")
    print(
        f"Seasons: "
        f"{sorted(combined['Season'].unique())}"
    )

    print("\nRows by season:")

    print(
        combined.groupby("Season")
        .size()
        .to_string()
    )

    print(
        f"\nSaved to: {output_path}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        required=True,
        help="Seasons to collect, e.g. 2023 2024 2025",
    )

    args = parser.parse_args()

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for season in args.seasons:
        build_season_dataset(season)

    combine_seasons(args.seasons)


if __name__ == "__main__":
    main()