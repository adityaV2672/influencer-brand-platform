"""
The behavioural event log.

Why it exists
-------------
src/reco/ranker.py and src/reco/cf.py are trained on a SIMULATED interaction
history because there is no real one. This module is where a real one starts:
every shortlist, rejection, pitch and request the app actually performs is
recorded in the same shape as the simulated log, so the two can be concatenated
and the models retrained without changing a line of the training code.

Where it writes
---------------
Session state always, and a parquet on disk when the filesystem allows it. The
hosted deployment has an ephemeral disk, so the file survives a session and not
a restart - which is the honest limit of a prototype with no database. The
schema is what matters: it matches nectar_interactions.parquet exactly, so a
real deployment swaps the sink and keeps everything else.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

KEY = "_event_log"
SINK = Path(__file__).resolve().parents[2] / "app_data" / "nectar_event_log.parquet"

# Same vocabulary as the simulated log, so the two stack.
STAGES = ["viewed", "shortlisted", "contacted", "accepted", "declined", "completed"]

COLUMNS = ["ts", "actor", "actor_id", "stage", "brand_id", "campaign_id",
           "influencer_id", "surface", "note"]


def log(stage: str, *, actor: str, actor_id: str = "", brand_id: str = "",
        campaign_id: str = "", influencer_id: str = "", surface: str = "",
        note: str = "") -> None:
    """Record one event. Never raises - a failed write must not break a click."""
    row = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "actor": actor, "actor_id": str(actor_id), "stage": stage,
        "brand_id": str(brand_id), "campaign_id": str(campaign_id),
        "influencer_id": str(influencer_id), "surface": surface, "note": note,
    }
    st.session_state.setdefault(KEY, []).append(row)
    try:
        df = pd.DataFrame(st.session_state[KEY], columns=COLUMNS)
        df.to_parquet(SINK, index=False)
    except Exception:                                            # noqa: BLE001
        # Read-only filesystem, or parquet unavailable. The session copy is
        # still there and the UI reads that, so the click succeeds either way.
        pass


def recent(n: int = 25) -> pd.DataFrame:
    rows = st.session_state.get(KEY, [])
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.tail(n).iloc[::-1]


def count(stage: str | None = None) -> int:
    rows = st.session_state.get(KEY, [])
    if stage is None:
        return len(rows)
    return sum(1 for r in rows if r.get("stage") == stage)


def summary() -> dict:
    rows = st.session_state.get(KEY, [])
    out = {s: 0 for s in STAGES}
    for r in rows:
        if r.get("stage") in out:
            out[r["stage"]] += 1
    return out
