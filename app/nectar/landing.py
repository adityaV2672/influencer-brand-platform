"""
The landing page: Nectar's front door.

Replicated from the Figma Make design. Four sections, in order:

    hero            ink panel on paper, oversized headline, impact strip
    both sides      editorial two-column, 01 / BRANDS and 02 / CREATORS
    bridge          one full-bleed sentence
    enter           the role choice, which is where the product starts

The role cards are the only interactive part, and they are real Streamlit
buttons rather than styled anchors - a link would lose the session state that
carries the chosen role into the app.
"""
from __future__ import annotations

import streamlit as st

from .theme import (ACCENT_A, ACCENT_B, GRADIENT, INK, INK_2, INK_3, LINE,
                    LINE_2, MONO, PAPER, SANS)

IMPACTS = [
    ("FASTER DISCOVERY", "Brief → ranked shortlist"),
    ("BETTER FIT", "Campaign + brand compatibility"),
    ("LESS GUESSWORK", "See why every creator matches"),
    ("FASTER DEALS", "Briefs, offers and counters in one place"),
]

POSITIONING = "SCORING · CAMPAIGN FIT · BRAND FIT · VERIFIED CREATOR DATA"

SIDES = [
    {"index": "01", "label": "BRANDS", "accent": ACCENT_A,
     "head": "Stop paying an agency to guess.",
     "body": "Describe the campaign in your own words. Every creator in the "
             "market is scored against it, gated on the things that actually "
             "stop a deal, and ranked with the reasoning attached.",
     "points": ["Campaign Fit, Organisation Fit and Creator Quality kept apart",
                "Competitor conflicts block a creator, they do not just lower them",
                "Audience authenticity checked before you pay for reach",
                "Every score opens into the components behind it"],
     "impact": ("2,000", "creators scored against every brief")},
    {"index": "02", "label": "CREATORS", "accent": ACCENT_B,
     "head": "Get found for what you actually do.",
     "body": "Connect your account and the numbers only you can see - saves, "
             "shares, watch time, who your audience really is - become the "
             "reason a brand picks you instead of someone with more followers.",
     "points": ["Briefs matched to your rates, formats and calendar",
                "Verified metrics rank above inferred ones",
                "See why a brief fits before you spend time on it",
                "Your data stays yours; you choose what to connect"],
     "impact": ("61%", "of creators have connected their insights")},
]


def _rings() -> str:
    """Eleven concentric drop contours, orange fading to pink.

    Generated rather than hand-written: at eleven rings the coordinates are
    not something a person should be maintaining, and vector-effect keeps the
    strokes hairline at every scale instead of thickening with the transform.
    """
    base = ("M 260 130 C 300 178 330 216 330 258 C 330 300 298 330 260 330 "
            "C 222 330 190 300 190 258 C 190 216 220 178 260 130 Z")
    out = []
    for i in range(11):
        s = 0.42 + i * 0.36
        t = i / 10
        colour = ACCENT_A if t < 0.5 else ACCENT_B
        op = 0.42 - 0.028 * i
        group = "a" if i % 2 == 0 else "b"
        out.append(
            f"<g class='ring ring-{group}' transform='translate(260 258) "
            f"scale({s:.3f}) rotate({i * 4}) translate(-260 -258)'>"
            f"<path d='{base}' fill='none' stroke='{colour}' "
            f"stroke-width='1' vector-effect='non-scaling-stroke' "
            f"opacity='{max(op, 0.06):.3f}'/></g>")
    scan = "".join(
        f"<line x1='0' y1='{y}' x2='520' y2='{y}' stroke='{ACCENT_A}' "
        f"stroke-width='0.5' opacity='0.05'/>" for y in range(60, 500, 44))
    return (f"<svg class='hero-art' viewBox='0 0 520 500' aria-hidden='true'>"
            f"{scan}{''.join(out)}"
            f"<circle cx='260' cy='258' r='5' fill='{ACCENT_B}'/></svg>")


def _logo(size: int = 60) -> str:
    inner = int(size * 0.30)
    return (f"<span class='mark' style='width:{size}px;height:{size}px'>"
            f"<span class='dot' style='width:{inner}px;height:{inner}px'></span>"
            f"</span>")


CSS = f"""
<style>
[data-testid="stSidebar"] {{ display:none !important; }}
[data-testid="stMain"] .block-container {{ max-width:1180px; padding-top:1.6rem; padding-bottom:3rem; }}
.hero {{ position:relative; background:{INK}; border-radius:28px; padding:52px 56px 44px; overflow:hidden; animation:heroIn .7s cubic-bezier(.16,.84,.44,1) both; }}
.hero-brand {{ display:flex; align-items:center; gap:16px; margin-bottom:44px; position:relative; z-index:2; }}
.mark {{ display:inline-flex; align-items:center; justify-content:center; border-radius:50% 50% 50% 4px; transform:rotate(45deg); background:{GRADIENT}; animation:logoSettle .8s cubic-bezier(.16,.84,.44,1) both; }}
.mark .dot {{ border-radius:50%; background:{INK}; transform:rotate(-45deg); }}
.wordmark {{ font-size:46px; font-weight:800; letter-spacing:-.04em; color:{PAPER}; line-height:1; }}
.hero-h {{ font-size:clamp(40px,5.4vw,76px); font-weight:800; letter-spacing:-.035em; line-height:1.02; color:{PAPER}; max-width:15ch; position:relative; z-index:2; }}
.hero-h .dot-accent {{ display:inline-block; width:13px; height:13px; border-radius:50%; background:{GRADIENT}; margin-left:10px; vertical-align:middle; }}
.hero-p {{ font-size:17px; line-height:1.55; color:#B4ADB1; max-width:52ch; margin-top:26px; position:relative; z-index:2; }}
.hero-art {{ position:absolute; right:-40px; top:20px; width:520px; height:500px; opacity:.95; pointer-events:none; z-index:1; }}
.ring {{ transform-origin:260px 258px; }}
.ring-a {{ animation:breathe 9s ease-in-out infinite; }}
.ring-b {{ animation:breathe 9s ease-in-out infinite; animation-delay:-4.5s; }}
.impact {{ display:grid; grid-template-columns:repeat(4,1fr); gap:0; margin-top:46px; border-top:1px solid rgba(255,255,255,.10); position:relative; z-index:2; }}
.impact div {{ padding:20px 22px 4px 0; border-right:1px solid rgba(255,255,255,.08); }}
.impact div:last-child {{ border-right:none; }}
.impact .k {{ font-family:{MONO}; font-size:11px; letter-spacing:.08em; color:{ACCENT_A}; font-weight:600; }}
.impact .v {{ font-size:13.5px; color:#B4ADB1; margin-top:7px; line-height:1.45; }}
.hero-mono {{ font-family:{MONO}; font-size:10.5px; letter-spacing:.14em; color:#6E666B; margin-top:34px; position:relative; z-index:2; }}
.eyebrow {{ font-family:{MONO}; font-size:11px; letter-spacing:.14em; color:{ACCENT_A}; font-weight:600; }}
.sec {{ margin-top:82px; }}
.sec-h {{ font-size:38px; font-weight:800; letter-spacing:-.03em; margin-top:12px; color:{INK}; }}
.side {{ background:#fff; border:1px solid {LINE}; border-radius:18px; padding:32px 30px; height:100%; position:relative; overflow:hidden; transition:transform .18s ease, box-shadow .18s ease; }}
.side::before {{ content:""; position:absolute; left:0; right:0; top:0; height:3px; background:var(--accent); opacity:0; transition:opacity .18s ease; }}
.side:hover {{ transform:translateY(-3px); box-shadow:0 10px 30px rgba(24,19,22,.07); }}
.side:hover::before {{ opacity:1; }}
.side .ix {{ font-family:{MONO}; font-size:11px; letter-spacing:.12em; color:{INK_3}; }}
.side .ix b {{ color:var(--accent); font-weight:600; }}
.side h3 {{ font-size:27px; font-weight:800; letter-spacing:-.03em; margin:14px 0 12px; line-height:1.15; }}
.side p {{ font-size:14px; line-height:1.62; color:{INK_2}; }}
.side ul {{ list-style:none; padding:0; margin:22px 0 0; }}
.side li {{ display:flex; gap:10px; align-items:flex-start; font-size:13.5px; line-height:1.55; color:{INK_2}; margin-bottom:11px; }}
.side li span {{ color:var(--accent); font-weight:700; }}
.side .box {{ margin-top:24px; border-top:1px solid {LINE_2}; padding-top:18px; display:flex; align-items:baseline; gap:12px; }}
.side .box b {{ font-family:{MONO}; font-size:30px; font-weight:700; letter-spacing:-.02em; color:{INK}; }}
.side .box span {{ font-size:12.5px; color:{INK_3}; line-height:1.4; }}
.bridge {{ margin:96px 0 0; text-align:center; font-size:clamp(30px,4vw,54px); font-weight:800; letter-spacing:-.035em; line-height:1.1; color:{INK}; }}
.bridge em {{ font-style:normal; color:{INK_3}; }}
.enter {{ margin-top:92px; text-align:center; }}
.enter h2 {{ font-size:38px; font-weight:800; letter-spacing:-.03em; margin-top:10px; }}
.rolecard {{ text-align:left; }}
.rolecard .rl {{ font-family:{MONO}; font-size:10.5px; letter-spacing:.12em; color:{INK_3}; }}
.rolecard h4 {{ font-size:21px; font-weight:700; letter-spacing:-.02em; margin:8px 0 6px; }}
.rolecard p {{ font-size:13.5px; color:{INK_2}; line-height:1.55; margin-bottom:6px; }}
.foot {{ margin-top:74px; padding-top:22px; border-top:1px solid {LINE}; display:flex; justify-content:space-between; font-family:{MONO}; font-size:10.5px; letter-spacing:.1em; color:{INK_3}; }}
@keyframes heroIn {{ from {{ opacity:0; transform:translateY(18px); }} to {{ opacity:1; transform:none; }} }}
@keyframes logoSettle {{ from {{ opacity:0; transform:rotate(45deg) scale(.7); }} to {{ opacity:1; transform:rotate(45deg) scale(1); }} }}
@keyframes breathe {{ 0%,100% {{ transform:scale(1); opacity:1; }} 50% {{ transform:scale(1.035); opacity:.72; }} }}
@media (prefers-reduced-motion: reduce) {{ .hero,.mark,.ring {{ animation:none !important; }} }}
@media (max-width:980px) {{ .hero {{ padding:36px 26px; }} .hero-art {{ display:none; }} .impact {{ grid-template-columns:repeat(2,1fr); }} }}
</style>
"""


def render() -> str | None:
    """Draw the landing page. Returns 'brand' or 'creator' once chosen."""
    st.markdown("\n".join(l for l in CSS.splitlines() if l.strip()),
                unsafe_allow_html=True)

    impacts = "".join(
        f"<div><div class='k'>{k}</div><div class='v'>{v}</div></div>"
        for k, v in IMPACTS)
    st.markdown(
        f"<div class='hero'>{_rings()}"
        f"<div class='hero-brand'>{_logo(60)}<span class='wordmark'>nectar</span></div>"
        f"<div class='hero-h'>Find the right creators.<br>Know exactly why."
        f"<span class='dot-accent'></span></div>"
        f"<div class='hero-p'>Nectar turns influencer marketing into a "
        f"transparent, self-serve marketplace — from campaign brief to ranked "
        f"creator shortlist to deal.</div>"
        f"<div class='impact'>{impacts}</div>"
        f"<div class='hero-mono'>{POSITIONING}</div></div>",
        unsafe_allow_html=True)

    st.markdown("<div class='sec'><div class='eyebrow'>BUILT FOR BOTH SIDES</div>"
                "<div class='sec-h'>One engine, pointed in two directions.</div></div>",
                unsafe_allow_html=True)
    cols = st.columns(2, gap="large")
    for col, s in zip(cols, SIDES):
        with col:
            points = "".join(f"<li><span>✓</span>{p}</li>" for p in s["points"])
            st.markdown(
                f"<div class='side' style='--accent:{s['accent']}'>"
                f"<div class='ix'><b>{s['index']}</b> / {s['label']}</div>"
                f"<h3>{s['head']}</h3><p>{s['body']}</p><ul>{points}</ul>"
                f"<div class='box'><b>{s['impact'][0]}</b>"
                f"<span>{s['impact'][1]}</span></div></div>",
                unsafe_allow_html=True)

    st.markdown("<div class='bridge'>The middleman disappears.<br>"
                "<em>The reasoning doesn't.</em></div>", unsafe_allow_html=True)

    st.markdown("<div class='enter'><div class='eyebrow'>ENTER NECTAR</div>"
                "<h2>Choose your side.</h2></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

    chosen = None
    left, mid, right = st.columns([1, 1, 1], gap="large")
    for col, (role, label, title, body, cta) in zip(
            (left, mid),
            [("brand", "FOR BRANDS", "I'm running a campaign",
              "Brief the campaign, see every creator scored against it, and "
              "read the reasoning before you commit budget.", "Enter Brand OS"),
             ("creator", "FOR CREATORS", "I make the content",
              "Connect your account, get matched to briefs that fit your rates, "
              "formats and calendar, and see why each one fits.",
              "Enter Creator OS")]):
        with col, st.container(border=True):
            st.markdown(
                f"<div class='rolecard'><div class='rl'>{label}</div>"
                f"<h4>{title}</h4><p>{body}</p></div>", unsafe_allow_html=True)
            if st.button(cta, key=f"enter_{role}", type="primary",
                         use_container_width=True):
                chosen = role
    with right:
        st.markdown(
            f"<div style='padding:6px 4px;font-size:12.5px;color:{INK_3};"
            f"line-height:1.65'><b style='color:{INK}'>Not sure?</b><br>"
            f"Brands search and shortlist. Creators receive briefs and reply. "
            f"You can switch sides at any time from the sidebar — the same "
            f"scores are shown to both, which is the point.</div>",
            unsafe_allow_html=True)

    st.markdown(
        f"<div class='foot'><span>NECTAR · MARKETPLACE PROTOTYPE</span>"
        f"<span>SYNTHETIC DATA · NO REAL CREATOR IS REPRESENTED</span></div>",
        unsafe_allow_html=True)
    return chosen
