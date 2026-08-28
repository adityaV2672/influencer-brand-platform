"""
Brand OS — Reporting. What the campaign delivered, and how close the model got.

The predicted-vs-actual panels are the honest part of this page. `predicted`
is the performance model's out-of-fold prediction — each campaign row scored by
folds that never saw that creator — and `actual` is the outcome recorded in the
modelling table. The model is not marking its own homework.
"""
from __future__ import annotations

import streamlit as st

from nectar import charts, data, state, ui
from nectar.theme import AMBER, GREEN, INK, INK_2, INK_3, LINE

summary = data.campaign_summary()
perf = data.creator_perf()
funnel = data.funnel()
camps = data.campaigns()

ran = summary[summary.spend > 0]
if ran.empty:
    st.markdown(ui.empty_state("📈", "No results yet.",
                               "Reporting appears once a campaign has creators in production."),
                unsafe_allow_html=True)
    st.stop()

names = list(ran.name)
default = state.campaign().name
idx = names.index(default) if default in names else 0

# Title first, controls on the right - the campaign name is the subject of the
# page, so a selector above it reads as a filter bar rather than a report.
h1, h2, h3 = st.columns([2.4, 1.15, 0.95])
pick = st.session_state.get("rep_camp") or names[idx]
if pick not in names:
    pick = names[idx]
row = ran[ran.name == pick].iloc[0]
with h1:
    st.markdown(ui.page_header(row["name"],
                               f"Performance report · {row.category} · Aug 2026",
                               eyebrow="Campaign"), unsafe_allow_html=True)
with h2:
    st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
    chosen = st.selectbox("Campaign", names, index=names.index(pick),
                          label_visibility="collapsed", key="rep_camp")
    if chosen != pick:
        st.rerun()
with h3:
    st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
    st.download_button(
        "⤓  Export report",
        data=perf[perf.campaign_id == row.campaign_id].to_csv(index=False),
        file_name=f"nectar_report_{row.campaign_id}.csv",
        mime="text/csv", use_container_width=True)

# ---- KPI row --------------------------------------------------------------
k = st.columns(5)
vs = float(row.vs_predicted)
for col, (lbl, val, sub, tone) in zip(k, [
    ("Total spend", ui.inr(row.spend), f"{row.budget_used:.0%} of budget", "flat"),
    ("Total reach", ui.count(row.reach), f"{vs:+.0f}% vs predicted",
     "good" if vs >= 0 else "warn"),
    ("Total engagement", ui.count(row.engagements), f"{vs:+.0f}% vs predicted",
     "good" if vs >= 0 else "warn"),
    ("Avg CPE", ui.inr(row.avg_cpe, dp=2), "cost per engagement", "flat"),
    ("Avg CPR", ui.inr(row.avg_cpr, dp=3), "cost per person reached", "flat"),
]):
    with col:
        st.markdown(ui.kpi(lbl, val, sub, tone), unsafe_allow_html=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# ---- funnel ---------------------------------------------------------------
f = funnel[funnel.campaign_id == row.campaign_id].sort_values("stage_index")
with st.container(border=True):
    st.markdown(ui.section("Campaign funnel",
                           "How many of the creators approached reached each stage"),
                unsafe_allow_html=True)
    st.plotly_chart(charts.funnel_bars(f.stage, f["count"], height=250),
                    use_container_width=True, config=charts.CONFIG)
    top = int(f["count"].max()) or 1
    def rate(stage):
        v = f[f.stage == stage]["count"]
        return (int(v.iloc[0]) / top) if len(v) else 0
    st.markdown(
        f"<div style='display:flex;gap:36px;font-size:12.5px;color:{INK_2}'>"
        f"<span><b class='n-num'>{rate('Viewed'):.0%}</b> view rate</span>"
        f"<span><b class='n-num'>{rate('Countered'):.0%}</b> response rate</span>"
        f"<span><b class='n-num'>{rate('Accepted'):.0%}</b> acceptance rate</span>"
        f"<span><b class='n-num'>{rate('Paid'):.0%}</b> completion rate</span></div>",
        unsafe_allow_html=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# ---- predicted vs actual --------------------------------------------------
p1, p2 = st.columns(2, gap="medium")
for col, (title, blurb, pred, act, colour) in zip([p1, p2], [
    ("Predicted vs Actual — Reach", "How well did our predictions hold?",
     row.predicted_reach, row.reach, charts.SERIES["reach"]),
    ("Predicted vs Actual — Engagement", "Engagement performed above expectation."
     if vs >= 0 else "Engagement came in under expectation.",
     row.predicted_engagements, row.engagements, charts.SERIES["engagement"]),
]):
    with col:
        with st.container(border=True):
            st.markdown(ui.section(title, blurb), unsafe_allow_html=True)
            c1, c2 = st.columns([2.4, 1])
            with c1:
                st.plotly_chart(charts.compare_pair(float(pred), float(act), colour),
                                use_container_width=True, config=charts.CONFIG)
            with c2:
                st.markdown(
                    f"<div style='padding-top:52px'>"
                    f"<div class='n-num' style='font-size:26px;"
                    f"color:{GREEN if vs >= 0 else AMBER}'>{vs:+.0f}%</div>"
                    f"<div style='font-size:11.5px;color:{INK_3}'>"
                    f"{'over' if vs >= 0 else 'under'} prediction</div></div>",
                    unsafe_allow_html=True)

st.markdown(
    f"<div class='n-muted' style='margin:8px 2px 20px 2px;line-height:1.6'>"
    f"Predictions are the performance model's <b>out-of-fold</b> estimates: every "
    f"creator was scored by folds that never saw them. They are back-transformed "
    f"from log space with Duan's smearing estimator, because exp() of a log-space "
    f"prediction estimates the median and biased these forecasts about 10% low "
    f"before the correction was applied. What remains is genuine forecast error, "
    f"not an artefact of the transform."
    f"</div>", unsafe_allow_html=True)

# ---- creator performance --------------------------------------------------
st.markdown("<div class='n-h2'>Creator performance</div>"
            "<div style='height:10px'></div>", unsafe_allow_html=True)
pc = perf[perf.campaign_id == row.campaign_id].sort_values("engagements", ascending=False)
rows = []
for r in pc.itertuples():
    rows.append([
        ui.creator_cell(r.creator_name, r.creator_handle, r.initials, r.avatar_color),
        f"<span class='n-num'>{ui.count(r.reach)}</span>",
        f"<span class='n-num'>{ui.count(r.engagements)}</span>",
        f"<span class='n-num'>{ui.inr(r.cost)}</span>",
        f"<span class='n-num' style='color:{GREEN}'>{ui.inr(r.cpe, dp=2)}</span>",
        f"<span class='n-num' style='color:{INK_3}'>{ui.inr(r.cpr, dp=3)}</span>",
    ])
st.markdown(
    ui.table(["Creator", "Reach", "Engagement", "Cost", "CPE", "CPR"], rows,
             aligns=["left", "right", "right", "right", "right", "right"]),
    unsafe_allow_html=True)
