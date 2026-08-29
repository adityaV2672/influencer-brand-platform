"""
What a creator can actually deliver, and when.

Two components of Campaign Fit were missing entirely before this module, and
both are the kind of thing that sinks a campaign in practice rather than in a
model: a creator who does not make Reels, and a creator who is booked.

Deliverable capability
----------------------
A rate card entry is necessary but not sufficient. A creator who charges for
Carousels but whose Carousels reach a third of their Reel audience is not a
good Carousel buy. So capability carries both a hard part - do they offer this
format on this platform at all - and a soft part - how well that format
performs for them relative to their own baseline.

Availability
------------
The creator table carried a text label ("8-17 Sept", "Booked through Sept")
that no code could compare against a campaign's dates. Structured windows are
generated here and the display text is DERIVED from them, so the label a
creator reads and the window the matcher uses cannot drift apart.
"""
from __future__ import annotations

import hashlib
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.config import SEED

FORMATS = ["Reel", "Story", "Carousel"]

# Anchor date for the synthetic marketplace. Campaign dates in the existing
# campaign table sit in this window.
TODAY = date(2026, 9, 1)
HORIZON_DAYS = 120


def _unit(key: str, salt: str) -> float:
    h = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def build(creators: pd.DataFrame, latents: pd.DataFrame,
          seed: int = SEED) -> pd.DataFrame:
    """One row per creator: format capability, format strength, availability."""
    rng = np.random.default_rng(seed + 8181)
    c = creators.copy()
    c["influencer_id"] = c.influencer_id.astype(str)
    lat = latents.copy(); lat["influencer_id"] = lat.influencer_id.astype(str)
    c = c.merge(lat[["influencer_id", "content_quality", "consistency"]],
                on="influencer_id", how="left")

    rows = []
    for r in c.itertuples():
        iid = str(r.influencer_id)

        # ---- formats offered -------------------------------------------
        offers = {f: bool(getattr(r, f"rate_{f.lower()}", 0) or 0) > 0
                  for f in FORMATS}
        # Not every creator does every format even if a rate exists. Stories
        # are near-universal; Carousels are a deliberate choice.
        offers["Carousel"] = offers["Carousel"] and _unit(iid, "carousel") > 0.22
        offers["Reel"] = offers["Reel"] and _unit(iid, "reel") > 0.05

        # ---- strength per format ---------------------------------------
        # Centred on 1.0: above means this format outperforms the creator's own
        # baseline, below means it underperforms. A brand should read it as a
        # multiplier, not a score out of a hundred.
        q = float(r.content_quality or 0.5)
        strength = {}
        for f in FORMATS:
            base = 0.72 + 0.55 * _unit(iid, f"str_{f}")
            lift = {"Reel": 0.18, "Story": -0.05, "Carousel": 0.02}[f]
            strength[f] = round(float(np.clip(base + lift + 0.25 * (q - 0.5)
                                              + rng.normal(0, 0.05), 0.35, 1.75)), 3)
            if not offers[f]:
                strength[f] = 0.0

        # ---- availability ----------------------------------------------
        # Modelled as a BOOKED BLOCK, not as the single window in which the
        # creator is free. The first version had it backwards - a creator was
        # available only inside one window and blocked everywhere else - which
        # made 51% of every campaign pool ineligible on dates alone. Real
        # creators are free by default and busy in patches, so the gate should
        # be rare and meaningful rather than the dominant filter.
        # The block's START is spread across the whole horizon for every
        # status; only its LENGTH varies. Clustering the start by status
        # instead - Busy creators all booked in September, Available ones all
        # booked from late October - made availability a proxy for the status
        # label rather than a date calculation, and every Busy creator failed
        # every September campaign identically.
        status = str(r.availability)
        booked_start = TODAY + timedelta(
            days=int(HORIZON_DAYS * _unit(iid, "bk_s")) - 20)
        booked_len = {"Unavailable": int(55 + 45 * _unit(iid, "bk_l")),
                      "Busy": int(16 + 26 * _unit(iid, "bk_l")),
                      }.get(status, int(4 + 12 * _unit(iid, "bk_l")))
        booked_end = booked_start + timedelta(days=booked_len)

        # Lead time a creator needs before a campaign can start.
        lead = int(3 + 11 * _unit(iid, "lead"))

        rows.append({
            "influencer_id": iid,
            **{f"offers_{f.lower()}": offers[f] for f in FORMATS},
            **{f"strength_{f.lower()}": strength[f] for f in FORMATS},
            "n_formats": int(sum(offers.values())),
            "availability_status": status,
            "booked_from": booked_start.isoformat(),
            "booked_to": booked_end.isoformat(),
            "lead_time_days": lead,
            "available_window": ("Free from " + (booked_end + timedelta(days=1))
                                 .strftime("%-d %b")) if booked_start <= TODAY
            else f"Free until {booked_start.strftime('%-d %b')}",
        })
    return pd.DataFrame(rows)


def deliverable_fit(cap_row, deliverables) -> tuple[float, list[str], bool]:
    """(score 0-1, reasons, blocked).

    Blocked when the creator does not offer a required format at all - that is
    an operational impossibility, not a preference, so it belongs with the hard
    gates rather than as a deduction.
    """
    if deliverables is None or len(deliverables) == 0:
        return 1.0, [], False
    reasons, weighted, total, missing = [], 0.0, 0, []
    for d in deliverables:
        fmt = str(d.get("type", "")) if isinstance(d, dict) else str(d)
        qty = int(d.get("qty", 1)) if isinstance(d, dict) else 1
        offers = bool(getattr(cap_row, f"offers_{fmt.lower()}", False))
        strength = float(getattr(cap_row, f"strength_{fmt.lower()}", 0.0))
        total += qty
        if not offers:
            missing.append(fmt)
            continue
        weighted += qty * float(np.clip(strength / 1.3, 0.0, 1.0))
    if missing:
        return 0.0, [f"Does not produce {', '.join(sorted(set(missing)))}"], True
    score = weighted / max(total, 1)
    best = max(FORMATS, key=lambda f: float(getattr(cap_row, f"strength_{f.lower()}", 0)))
    if score >= 0.75:
        reasons.append(f"{best}s are among their strongest format")
    elif score < 0.45:
        reasons.append("Required formats are not where they perform best")
    return round(float(np.clip(score, 0, 1)), 4), reasons, False


def availability_fit(cap_row, start, end) -> tuple[float, list[str], bool]:
    """(score 0-1, reasons, blocked).

    Free days are campaign days that fall OUTSIDE the creator's booked block.
    Blocked only when the block swallows the campaign entirely, which is a
    genuine impossibility rather than an inconvenience.
    """
    try:
        bs = date.fromisoformat(str(cap_row.booked_from))
        be = date.fromisoformat(str(cap_row.booked_to))
        s = start if isinstance(start, date) else pd.Timestamp(start).date()
        e = end if isinstance(end, date) else pd.Timestamp(end).date()
    except Exception:                                            # noqa: BLE001
        return 0.5, [], False

    needed = (e - s).days + 1
    clash = max(0, (min(be, e) - max(bs, s)).days + 1)
    free = needed - clash
    if free <= 0:
        return 0.0, [f"Booked for the entire window, free from "
                     f"{(be + timedelta(days=1)).strftime('%-d %b')}"], True
    cover = float(np.clip(free / max(needed, 1), 0, 1))
    lead_ok = (s - TODAY).days >= int(cap_row.lead_time_days)
    score = cover * (1.0 if lead_ok else 0.88)
    if cover >= 0.99:
        reasons = ["Free for the whole campaign window"]
    else:
        reasons = [f"Free for {free} of the {needed} campaign days"]
    if not lead_ok:
        reasons.append(f"Needs {int(cap_row.lead_time_days)} days' notice")
    return round(score, 4), reasons, False
