"""Creator OS — Overview. What needs the creator's attention today."""
from __future__ import annotations

import streamlit as st

from nectar import creator_ctx as ctx
from nectar import data, state, ui
from nectar.theme import AMBER, GREEN, INK, INK_2, INK_3, LINE

me = ctx.me()
reqs = ctx.my_requests()
cat_fit = ctx.my_category_fit()
earn = ctx.my_earnings()

st.markdown(ui.page_header(f"Good morning, {me.name.split()[0]}.",
                           "Here's what needs your attention."),
            unsafe_allow_html=True)

# Earnings come from the earnings table, which is what the Earnings page shows.
# Deriving them separately here from stage_index produced a header reading
# "This month's earnings 0" beside a panel reading "Paid 2.17L".
earn_months = earn.set_index("month")["amount"] if len(earn) else None
paid = float(earn.amount.sum()) if len(earn) else 0.0
# A quarter, not a month. Deals settle lumpily - a creator with two good
# months and a quiet August is not a creator earning nothing, and a headline
# metric that reads zero in that case is misleading rather than precise.
quarter = ["Jun", "Jul", "Aug"]
this_quarter = (float(earn[earn.month.isin(quarter)].amount.sum())
                if len(earn) else 0.0)
pending = float(reqs[(reqs.stage_index >= 4) & (reqs.stage_index < 7)].fee_inr.sum())
active = int((reqs.stage_index >= 4).sum())
new_reqs = int(reqs.status.isin(["Sent", "Viewed", "Countered"]).sum())
views = int(me.followers * 0.006) + 120

k = st.columns(4)
for col, (lbl, val, sub, tone) in zip(k, [
    ("Active deals", f"{active}",
     f"{int((reqs.status == 'Countered').sum())} need action", "warn"),
    ("Pending requests", f"{new_reqs}", "respond within 48h", "good"),
    ("Earnings this quarter", ui.inr(this_quarter), f"+{ui.inr(pending)} pending", "good"),
    ("Profile views", f"{views:,}", "+34% this week", "good"),
]):
    with col:
        st.markdown(ui.kpi(lbl, val, sub, tone), unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

left, right = st.columns([1.7, 1], gap="large")

with left:
    st.markdown("<div class='n-h2'>Your next actions</div>"
                "<div style='height:10px'></div>", unsafe_allow_html=True)

    actions = []
    countered = reqs[reqs.status == "Countered"]
    if len(countered):
        r = countered.iloc[0]
        actions.append(("Respond to request",
                        f"{r.brand_name} is waiting for your reply.",
                        "View request", "views/creator_requests.py"))
    inprod = reqs[reqs.status == "In production"]
    if len(inprod):
        r = inprod.iloc[0]
        actions.append(("Submit deliverable",
                        f"{r.campaign_name} — due {r.deadline}.",
                        "Submit", "views/creator_deals.py"))
    if me.availability != "Available":
        actions.append(("Update availability",
                        "Your calendar shows you as unavailable — brands filter on it.",
                        "Update", "views/creator_profile.py"))
    actions.append(("Complete profile", "Add your rate card and availability calendar.",
                    "Update", "views/creator_profile.py"))

    for i, (title, body, cta, target) in enumerate(actions[:3]):
        with st.container(border=True):
            a, b = st.columns([3, 1])
            with a:
                st.markdown(
                    f"<div style='font-size:14px;font-weight:650'>{ui.esc(title)}</div>"
                    f"<div style='font-size:12.5px;color:{INK_3};margin-top:2px'>"
                    f"{ui.esc(body)}</div>", unsafe_allow_html=True)
            with b:
                if st.button(cta, key=f"act_{i}", use_container_width=True):
                    st.switch_page(target)

    st.markdown("<div style='height:16px'></div>"
                "<div class='n-h2'>Recent requests</div>"
                "<div style='height:10px'></div>", unsafe_allow_html=True)

    if reqs.empty:
        st.markdown(ui.empty_state("✉", "No requests yet.",
                                   "Brands find you through search. Keep your profile "
                                   "and availability current."),
                    unsafe_allow_html=True)
    else:
        for r in reqs.head(4).itertuples():
            new = "New" if r.status in ("Sent", "Viewed") else r.status.lower()
            fg, bg = (AMBER, "#FBF3E0") if new == "New" else (INK_3, "#F2EEEB")
            with st.container(border=True):
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:12px'>"
                    f"<div style='flex:1;min-width:0'>"
                    f"<div style='font-size:13.5px'><b>{ui.esc(r.brand_name)}</b>"
                    f"<span style='color:{INK_3}'> · {ui.esc(r.campaign_name)}</span></div>"
                    f"<div style='font-size:12px;color:{INK_3};margin-top:2px'>"
                    f"{ui.esc(r.deliverables)} · {ui.inr(r.fee_inr)} · {ui.esc(r.deadline)}</div>"
                    f"</div>"
                    f"<span class='n-chip' style='color:{fg};background:{bg}'>{ui.esc(new)}</span>"
                    f"<div style='text-align:right'>"
                    f"<div class='n-num' style='font-size:14px'>{ui.inr(r.fee_inr)}</div>"
                    f"<div style='font-size:11.5px;color:{GREEN}'>{r.campaign_fit:.0f}th pctile</div>"
                    f"</div></div>", unsafe_allow_html=True)

with right:
    with st.container(border=True):
        st.markdown("<div class='n-eyebrow'>Who'd want you?</div>"
                    "<div class='n-h3' style='margin:2px 0 12px 0'>"
                    "Best-fit brand categories for you.</div>",
                    unsafe_allow_html=True)
        for r in cat_fit.head(5).itertuples():
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"align-items:baseline;font-size:13px;margin-bottom:3px'>"
                f"<span style='font-weight:600'>{ui.esc(r.category)}</span>"
                f"<span style='color:{INK_3};font-size:12px'>{int(r.brands)} brands"
                f" &nbsp;<b class='n-num' style='color:{GREEN}'>{r.fit_pct:.0f}%</b></span>"
                f"</div>{ui.bar(r.fit_pct / 100, GREEN, width='100%')}"
                f"<div style='height:11px'></div>", unsafe_allow_html=True)
        if st.button("Explore full analytics  →", use_container_width=True,
                     key="explore_an"):
            st.switch_page("views/creator_analytics.py")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("<div class='n-eyebrow'>Earnings</div>", unsafe_allow_html=True)
        for lbl, val, colour in [("Paid", ui.inr(paid), GREEN),
                                 ("Pending", ui.inr(pending), AMBER),
                                 ("This quarter", ui.inr(paid + pending), INK)]:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:6px 0;"
                f"font-size:13px'><span style='color:{INK_2}'>{lbl}</span>"
                f"<span class='n-num' style='color:{colour}'>{val}</span></div>",
                unsafe_allow_html=True)
        if st.button("View earnings  →", use_container_width=True, key="view_earn"):
            st.switch_page("views/creator_earnings.py")
