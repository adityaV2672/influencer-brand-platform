"""Brand OS — Requests. Every creator request across every campaign, as a funnel."""
from __future__ import annotations

import streamlit as st

from nectar import charts, data, state, ui
from nectar.theme import AMBER, GREEN, INK, INK_2, INK_3

reqs = data.requests()
funnel = data.funnel()
camps = data.campaigns()

st.markdown(ui.page_header("Requests",
                           "Track every creator request across your campaigns."),
            unsafe_allow_html=True)

names = ["All campaigns"] + list(camps[camps.status != "Draft"].name)
pick = st.selectbox("Campaign", names, label_visibility="collapsed", key="req_camp")
d = reqs if pick == "All campaigns" else reqs[reqs.campaign_name == pick]

# ---- funnel strip ---------------------------------------------------------
if pick == "All campaigns":
    f = funnel.groupby(["stage", "stage_index"], as_index=False)["count"].sum()
else:
    cid = camps[camps.name == pick].campaign_id.iloc[0]
    f = funnel[funnel.campaign_id == cid]
f = f.sort_values("stage_index")

with st.container(border=True):
    cols = st.columns(len(f))
    for col, r in zip(cols, f.itertuples()):
        with col:
            st.markdown(
                f"<div style='text-align:center'>"
                f"<div class='n-num' style='font-size:21px;color:{INK}'>{r.count}</div>"
                f"<div style='font-size:11.5px;color:{INK_3};margin-top:2px'>"
                f"{ui.esc(r.stage)}</div></div>",
                unsafe_allow_html=True)
    top = int(f["count"].max()) or 1
    view_rate = int(f[f.stage == "Viewed"]["count"].iloc[0]) / top if len(f) else 0
    resp = int(f[f.stage == "Countered"]["count"].iloc[0]) / top if len(f) else 0
    acc = int(f[f.stage == "Accepted"]["count"].iloc[0]) / top if len(f) else 0
    st.markdown(
        f"<div style='display:flex;gap:34px;margin-top:12px;font-size:12.5px;color:{INK_2}'>"
        f"<span><b class='n-num'>{view_rate:.0%}</b> view rate</span>"
        f"<span><b class='n-num'>{resp:.0%}</b> response rate</span>"
        f"<span><b class='n-num'>{acc:.0%}</b> acceptance rate</span>"
        f"<span><b class='n-num'>{ui.inr(d.fee_inr.sum())}</b> requested value</span>"
        "</div>", unsafe_allow_html=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='n-h2'>All requests <span style='color:{INK_3};font-weight:500'>"
    f"— {len(d)} results</span></div><div style='height:10px'></div>",
    unsafe_allow_html=True)

rows = []
for r in d.sort_values("stage_index", ascending=False).head(60).itertuples():
    action = ""
    if r.status == "Countered":
        action = (f"<span class='n-chip' style='color:#fff;background:"
                  f"linear-gradient(180deg,#FF6A2C,#FF3E93)'>Accept "
                  f"{ui.inr(r.counter_fee_inr)}</span>")
    elif r.status in ("Sent", "Viewed"):
        action = f"<span style='color:{INK_3};font-style:italic;font-size:12.5px'>Awaiting response</span>"
    rows.append([
        ui.creator_cell(r.creator_name, r.creator_handle, r.initials, r.avatar_color),
        f"<span style='font-size:13px'>{ui.esc(r.campaign_name)}</span>",
        f"<span class='n-num'>{ui.inr(r.fee_inr)}</span>",
        f"<span style='font-size:12.5px;color:{INK_2}'>{ui.esc(r.deliverables)}</span>",
        f"<span class='n-num' style='font-size:12px;color:{INK_3}'>{ui.esc(r.request_id)}</span>",
        ui.chip(r.status),
        action,
    ])
st.markdown(
    ui.table(["Creator", "Campaign", "Fee", "Deliverables", "Ref", "Status", "Action"], rows,
             aligns=["left", "left", "right", "left", "left", "left", "left"]),
    unsafe_allow_html=True)
