"""
The transactional layer: requests, negotiations, deals, and campaign reporting.

The creator universe and its scores are produced by the modelling pipeline.
What happens *after* a brand picks a creator - the offer, the counter, the
delivery, the payment - has no counterpart in the modelling data, so it is
simulated here from a fixed funnel with a fixed seed.

Two things are deliberately NOT simulated:

  * fees come from the trained price model, not from a random draw;
  * predicted-vs-actual on the Reporting page comes from the model's real
    out-of-fold predictions against the real campaign outcomes in the
    modelling table. If the model were worse, that chart would look worse.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# The funnel every request walks. Order matters: a request at stage k has, by
# definition, passed every stage before it, which is what makes the Requests
# page a funnel rather than a pie chart.
STAGES = ["Drafted", "Sent", "Viewed", "Countered", "Accepted",
          "In production", "Delivered", "Approved", "Paid"]

# Survival at each step, calibrated to the response rates the influencer
# marketing industry reports for cold brand outreach (roughly 4 in 5 open,
# half reply, and a little under half of those convert).
SURVIVAL = [1.00, 0.97, 0.80, 0.50, 0.40, 0.27, 0.17, 0.12, 0.08]

TERMINAL_STATUSES = {"Declined", "Expired"}


def _fee_for(rate_reel, rate_story, rate_carousel, deliverables) -> float:
    """Price the brief using the fee model's per-deliverable rates."""
    unit = {"Reel": rate_reel, "Story": rate_story, "Carousel": rate_carousel}
    return float(sum(unit.get(d["type"], rate_reel) * d["qty"] for d in deliverables))


def build_requests(fit: pd.DataFrame, campaigns: pd.DataFrame,
                   seed: int = 20260904) -> pd.DataFrame:
    """One row per creator actually approached for a campaign.

    The funnel is applied first, then the budget. A brand does not keep signing
    creators after the money is gone: once committed spend would exceed the
    campaign's target utilisation, remaining acceptances are demoted back to
    "Viewed". Without that step every campaign overspent by 25-90%, because the
    acceptance rate was applied to the pool rather than to the budget.
    """
    rng = np.random.default_rng(seed)
    # How much of the budget a campaign has committed by the point we observe
    # it. A finished campaign has spent nearly all of it; a live one is
    # mid-flight.
    UTILISATION = {"Live": 0.78, "Completed": 0.95}
    rows = []
    for camp in campaigns.itertuples():
        if camp.status == "Draft":
            continue                                  # nothing sent yet
        pool = fit[(fit.campaign_id == camp.campaign_id) & fit.eligible]
        pool = pool.sort_values("rank_best")
        n_sent = {"Live": 42, "Completed": 34}[camp.status]
        pool = pool.head(n_sent)
        if pool.empty:
            continue

        budget_left = float(camp.budget_inr) * UTILISATION[camp.status]
        staged = []
        for j, r in enumerate(pool.itertuples()):
            # Position in the funnel: the best-fitting creators progress
            # furthest, which is the whole claim the product makes.
            share = (j + 0.5) / len(pool)
            stage = 0
            for k, surv in enumerate(SURVIVAL):
                if share <= surv:
                    stage = k
            if camp.status == "Completed":
                stage = min(stage + 2, len(STAGES) - 1)

            fee = float(r.brief_fee_inr)
            if stage >= 4:                            # accepted or beyond
                if fee <= budget_left:
                    budget_left -= fee
                else:
                    stage = 2                         # viewed, never signed
            staged.append((r, stage, fee))

        for r, stage, fee in staged:
            status = STAGES[stage]
            # A slice of non-responders are explicit declines rather than
            # silence - brands need to see the difference.
            if status == "Viewed" and rng.random() < 0.18:
                status = "Declined"
            countered = status == "Countered"
            counter_fee = round(fee * float(rng.uniform(1.12, 1.28)), -2) if countered else np.nan

            day = 1 + int(rng.integers(0, 20))
            rows.append({
                "request_id": f"{r.initials}-2026-{100 + len(rows):03d}",
                "campaign_id": camp.campaign_id,
                "campaign_name": camp.name,
                "brand_name": camp.brand_name,
                "brand_category": camp.category,
                "influencer_id": r.influencer_id,
                "creator_name": r.name,
                "creator_handle": r.nectar_handle,
                "initials": r.initials,
                "avatar_color": r.avatar_color,
                "followers": int(r.followers),
                "campaign_fit": float(r.campaign_fit),
                "org_fit": float(r.org_fit),
                "stage_index": stage,
                "status": status,
                "fee_inr": fee,
                "counter_fee_inr": counter_fee,
                "deliverables": camp.deliverable_label,
                "sent_at": f"2026-08-{day:02d}",
                "viewed_at": f"2026-08-{min(day + 1, 28):02d}" if stage >= 2 else None,
                "responded_at": f"2026-08-{min(day + 3, 30):02d}" if stage >= 3 else None,
                "deadline": camp.end_date,
                "payment": "Paid",
                "usage_rights": "30 days",
                "exclusivity": "No",
            })
    return pd.DataFrame(rows)


def build_messages(requests: pd.DataFrame, seed: int = 20260904) -> pd.DataFrame:
    """Negotiation threads for the requests that reached a conversation."""
    rng = np.random.default_rng(seed + 1)
    rows = []
    for r in requests.itertuples():
        if r.stage_index < 2:
            continue
        thread = [{
            "request_id": r.request_id, "seq": 0, "sender": "brand",
            "sender_name": r.brand_name, "timestamp": r.sent_at,
            "body": (f"Hi {r.creator_name.split()[0]} — we'd love to have you for "
                     f"{r.campaign_name}. Here's our offer."),
            "offer_inr": r.fee_inr, "offer_note": f"{r.deliverables} · {r.deadline}",
        }]
        if r.status == "Countered" and pd.notna(r.counter_fee_inr):
            thread.append({
                "request_id": r.request_id, "seq": 1, "sender": "creator",
                "sender_name": r.creator_name, "timestamp": r.responded_at or r.sent_at,
                "body": ("Thanks for reaching out! I love the campaign direction. "
                         "Countering with a small fee adjustment and an extended deadline."),
                "offer_inr": r.counter_fee_inr,
                "offer_note": f"{r.deliverables} · shifted by 3 days",
            })
        elif r.stage_index >= 4:
            thread.append({
                "request_id": r.request_id, "seq": 1, "sender": "creator",
                "sender_name": r.creator_name, "timestamp": r.responded_at or r.sent_at,
                "body": "Happy to take this on — the brief fits what I already post. Accepting as offered.",
                "offer_inr": r.fee_inr, "offer_note": "Accepted as offered",
            })
        if r.stage_index >= 6:
            thread.append({
                "request_id": r.request_id, "seq": 2, "sender": "creator",
                "sender_name": r.creator_name, "timestamp": f"2026-09-{int(rng.integers(2, 20)):02d}",
                "body": "Deliverables are up. Links are in the deal panel — let me know if you want a cut-down.",
                "offer_inr": np.nan, "offer_note": None,
            })
        rows.extend(thread)
    return pd.DataFrame(rows)


def build_funnel(requests: pd.DataFrame) -> pd.DataFrame:
    """Cumulative funnel per campaign: how many requests reached each stage."""
    rows = []
    for cid, g in requests.groupby("campaign_id"):
        for k, stage in enumerate(STAGES):
            rows.append({"campaign_id": cid, "stage": stage, "stage_index": k,
                         "count": int((g.stage_index >= k).sum())})
    return pd.DataFrame(rows)


def build_reporting(requests: pd.DataFrame, campaigns: pd.DataFrame,
                    creators: pd.DataFrame, modelling: pd.DataFrame,
                    oof: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Campaign-level results, per-creator efficiency, and predicted vs actual.

    Predicted-vs-actual is the honest part. `oof` holds the performance model's
    out-of-fold predictions - each row predicted by folds that never saw that
    creator - and `modelling.campaign_engagement_rate` holds what actually
    happened. Aggregating both by brand category gives a real calibration
    check, not a flattering one.
    """
    mt = modelling.copy()
    mt["pred_er"] = oof
    cal = mt.groupby("brand_category").agg(
        predicted_er=("pred_er", "mean"),
        actual_er=("campaign_engagement_rate", "mean"),
        n_campaigns=("pred_er", "size"),
    ).reset_index()

    # Budget is committed when a deal is ACCEPTED, not when the content ships.
    # Counting only delivered work made every campaign look like it had spent
    # 15% of its budget while being 80% of the way through its calendar.
    live = requests[requests.stage_index >= 4]         # accepted or beyond
    per_creator = live.merge(
        creators[["influencer_id", "avg_reach", "engagement_rate", "avg_views"]],
        on="influencer_id", how="left",
    )
    # Reach delivered = the creator's typical reach per post times the number of
    # deliverables in the brief.
    n_units = per_creator["deliverables"].str.extractall(r"(\d+)")[0].astype(int) \
        .groupby(level=0).sum().reindex(per_creator.index).fillna(3)
    per_creator["reach"] = (per_creator["avg_reach"].fillna(0) * n_units).round(0)
    # Engagement rate is defined against FOLLOWERS, not reach. Multiplying it by
    # reach undercounted engagements by the reach/follower ratio (~0.36 here) and
    # inflated every cost-per-engagement figure by roughly 3x.
    per_creator["engagements"] = (
        per_creator["followers"] * per_creator["engagement_rate"] * n_units).round(0)
    per_creator["cost"] = per_creator["fee_inr"]
    per_creator["cpe"] = (per_creator["cost"] / per_creator["engagements"].replace(0, np.nan)).round(2)
    per_creator["cpr"] = (per_creator["cost"] / per_creator["reach"].replace(0, np.nan)).round(3)

    camp_summary = per_creator.groupby("campaign_id").agg(
        spend=("cost", "sum"), reach=("reach", "sum"),
        engagements=("engagements", "sum"), creators=("influencer_id", "nunique"),
    ).reset_index()
    camp_summary = camp_summary.merge(
        campaigns[["campaign_id", "name", "category", "budget_inr", "status"]],
        on="campaign_id", how="right",
    ).fillna({"spend": 0, "reach": 0, "engagements": 0, "creators": 0})
    camp_summary["avg_cpe"] = (camp_summary.spend /
                               camp_summary.engagements.replace(0, np.nan)).round(2)
    camp_summary["avg_cpr"] = (camp_summary.spend /
                               camp_summary.reach.replace(0, np.nan)).round(3)
    camp_summary["budget_used"] = (camp_summary.spend /
                                   camp_summary.budget_inr).round(3)
    camp_summary = camp_summary.merge(
        cal.rename(columns={"brand_category": "category"}), on="category", how="left")
    camp_summary["vs_predicted"] = (
        (camp_summary.actual_er / camp_summary.predicted_er - 1) * 100).round(1)
    # Reach and engagement are shown separately in the UI; both inherit the same
    # calibration gap, applied to the measured totals.
    camp_summary["predicted_reach"] = (
        camp_summary.reach / (1 + camp_summary.vs_predicted / 100)).round(0)
    camp_summary["predicted_engagements"] = (
        camp_summary.engagements / (1 + camp_summary.vs_predicted / 100)).round(0)

    return camp_summary, per_creator, cal


def build_monthly(camp_summary: pd.DataFrame, seed: int = 20260904) -> pd.DataFrame:
    """Five months of portfolio performance for the Overview chart.

    Each measure gets its own trajectory. An earlier version scaled all three
    off one ramp vector, which made them exactly proportional - and since the
    chart normalises each series to its own peak, three proportional series
    drew three identical lines stacked on top of each other. Spend ramps
    steadily as campaigns are committed; reach is lumpy because it depends on
    when large creators actually posted; engagement lags reach by about a
    month, which is the pattern the campaign data shows.
    """
    rng = np.random.default_rng(seed + 2)
    months = ["Apr", "May", "Jun", "Jul", "Aug"]
    spend_shape = np.array([0.11, 0.16, 0.20, 0.24, 0.29])
    reach_shape = np.array([0.09, 0.22, 0.14, 0.31, 0.24])
    eng_shape = np.array([0.08, 0.15, 0.24, 0.20, 0.33])

    def spread(total, shape):
        j = rng.uniform(0.95, 1.05, len(shape))
        w = shape * j
        return (total * w / w.sum()).round(0)

    return pd.DataFrame({
        "month": months,
        "spend": spread(float(camp_summary.spend.sum()), spend_shape),
        "reach": spread(float(camp_summary.reach.sum()), reach_shape),
        "engagement": spread(float(camp_summary.engagements.sum()), eng_shape),
    })


# --------------------------------------------------------------------------
# Creator-side history
# --------------------------------------------------------------------------
# The six showcase campaigns are the brand side's world. A creator's world is
# wider: they are approached by brands all year, most of which have nothing to
# do with the campaigns a single brand happens to be running. Without that
# history the Creator OS opens on an inbox holding one item.
#
# These rows are SIMULATED, like the funnel and the negotiation messages, and
# are declared as such in nectar_meta.json. What is not simulated is the money:
# every fee comes from the trained price model, and every fit percentage is the
# same brand-fit composite the brand side ranks on.

HISTORY_MONTHS = ["Mar", "Apr", "May", "Jun", "Jul", "Aug"]
HISTORY_DELIVERABLES = [
    ("2 Reel, 3 Story", 2, 3, 0), ("1 Reel, 1 Carousel", 1, 0, 1),
    ("2 Reel, 2 Story", 2, 2, 0), ("1 Reel, 3 Story", 1, 3, 0),
    ("3 Reel", 3, 0, 0), ("2 Reel, 1 Carousel", 2, 0, 1),
]
HISTORY_STATUS = ["Paid", "Paid", "Paid", "Approved", "Delivered",
                  "In production", "Countered", "Viewed", "Declined"]


def build_creator_history(creators: pd.DataFrame, category_fit: pd.DataFrame,
                          brands: pd.DataFrame, top_n: int = 400,
                          seed: int = 20260904) -> pd.DataFrame:
    """Two to five past brand approaches for each of the most active creators."""
    rng = np.random.default_rng(seed + 7)
    # Only the creators a brand would plausibly have found: the ones that rank
    # well somewhere. Generating history for all 2,000 would be noise.
    best = (category_fit.sort_values("fit_pct", ascending=False)
            .groupby("influencer_id", as_index=False).head(3))
    strength = best.groupby("influencer_id").fit_pct.mean().sort_values(ascending=False)
    chosen = list(strength.head(top_n).index)

    cre = creators.set_index("influencer_id")
    by_cat = {c: g for c, g in brands.groupby("category")}
    rows = []
    for iid in chosen:
        if iid not in cre.index:
            continue
        c = cre.loc[iid]
        cats = best[best.influencer_id == iid]
        n = int(rng.integers(2, 6))
        for k in range(n):
            crow = cats.iloc[k % len(cats)]
            pool = by_cat.get(crow.category)
            if pool is None or pool.empty:
                continue
            brand = pool.iloc[int(rng.integers(0, len(pool)))]
            label, n_reel, n_story, n_car = HISTORY_DELIVERABLES[
                int(rng.integers(0, len(HISTORY_DELIVERABLES)))]
            fee = round(float(c.rate_reel * n_reel + c.rate_story * n_story
                              + c.rate_carousel * n_car), -2)
            status = HISTORY_STATUS[int(rng.integers(0, len(HISTORY_STATUS)))]
            month = HISTORY_MONTHS[int(rng.integers(0, len(HISTORY_MONTHS)))]
            day = int(rng.integers(1, 28))
            month_num = {"Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8}[month]
            rows.append({
                "request_id": f"{c.initials}-H{len(rows):04d}",
                "influencer_id": iid,
                "brand_id": brand.brand_id,
                "brand_name": brand.brand_name,
                "brand_category": crow.category,
                "campaign_name": f"{brand.brand_name.split()[0]} "
                                 f"{['Edit', 'Series', 'Drop', 'Launch', 'Refresh'][k % 5]}",
                "deliverables": label,
                "fee_inr": fee,
                "counter_fee_inr": (round(fee * float(rng.uniform(1.10, 1.25)), -2)
                                    if status == "Countered" else np.nan),
                "status": status,
                "stage_index": STAGES.index(status) if status in STAGES else 2,
                "campaign_fit": float(crow.fit_pct),
                "month": month,
                "date": f"2026-{month_num:02d}-{day:02d}",
                "deadline": f"2026-{month_num:02d}-{min(day + 12, 28):02d}",
                "creator_name": c["name"],
                "creator_handle": c.nectar_handle,
                "initials": c.initials,
                "avatar_color": c.avatar_color,
                "payment": "Paid",
                "usage_rights": "30 days",
                "exclusivity": "No",
            })
    return pd.DataFrame(rows)
