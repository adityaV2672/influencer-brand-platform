"""
The landing page: Nectar's front door.

One screen, landscape, no scroll. The page is a fixed-height flex column
rather than a document:

    head        logo, wordmark, positioning line
    cols        the argument (left) and the two sides (right) - this row
                takes whatever height is left over and centres its content
    foot        the bridge line and the two role buttons

Because the middle row absorbs the slack and the footer is a rigid flex
item, the role buttons cannot be pushed below the fold no matter how tall
the argument gets - the middle clips before the footer moves.

The role cards are the only interactive part, and they are real Streamlit
buttons rather than styled anchors - a link would lose the session state that
carries the chosen role into the app.
"""
from __future__ import annotations

import streamlit as st

from .theme import (ACCENT_A, ACCENT_B, GRADIENT, INK, INK_2, INK_3, LINE,
                    LINE_2, MONO, PAPER, SANS)
from .ui import logo_svg

IMPACTS = [
    ("FASTER DISCOVERY", "Brief → ranked shortlist"),
    ("BETTER FIT", "Campaign + brand compatibility"),
    ("LESS GUESSWORK", "See why every creator matches"),
    ("FASTER DEALS", "Briefs, offers and counters in one place"),
]

POSITIONING = "SCORING · CAMPAIGN FIT · BRAND FIT · VERIFIED CREATOR DATA"

# The card ground is a flat colour rather than a translucent wash: the drop
# contours behind the page used to show through the cards and cross the text.
CARD = "#1F1A1E"

SIDES = [
    {"index": "01", "label": "BRANDS", "accent": ACCENT_A,
     "head": "Stop paying an agency to guess.",
     "body": "Describe the campaign in your own words. Every creator is scored "
             "against it and ranked with the reasoning attached.",
     "points": ["Campaign, organisation and quality scores kept apart",
                "Conflicts block a creator rather than quietly lowering them",
                "Audience authenticity checked before you pay for reach"],
     "impact": ("2,000", "creators scored against every brief")},
    {"index": "02", "label": "CREATORS", "accent": ACCENT_B,
     "head": "Get found for what you actually do.",
     "body": "Connect your account and the numbers only you can see become the "
             "reason a brand picks you over someone with more followers.",
     "points": ["Briefs matched to your rates, formats and calendar",
                "Connected accounts rank above unverified ones",
                "See why a brief fits before you spend time on it"],
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


def _logo(size: int = 46) -> str:
    """One mark, defined once, in ui.logo_svg."""
    return f"<span class='mark'>{logo_svg(size)}</span>"


CSS = f"""
<style>
/* ---- the frame -------------------------------------------------------
   height:100vh with padding only adds up to one screen if the padding is
   counted inside the height. Without border-box the container measured
   100vh + 20px, which is precisely how far the role buttons used to hang
   below the fold. */
[data-testid="stSidebar"], [data-testid="stHeader"], [data-testid="stToolbar"] {{ display:none !important; }}
[data-testid="stAppViewContainer"], [data-testid="stMain"], .stApp {{ background:{INK} !important; }}
[data-testid="stMain"] .block-container {{
    box-sizing:border-box; max-width:1560px; padding:2.2vh 2.8vw 2.0vh;
    height:100vh; overflow:hidden; display:flex; flex-direction:column; position:relative; }}
[data-testid="stMain"] .block-container *, [data-testid="stMain"] .block-container *::before,
[data-testid="stMain"] .block-container *::after {{ box-sizing:border-box; }}

/* Streamlit wraps the page in two anonymous divs before the vertical block.
   They are made flex columns so the height of the frame reaches the rows. */
[data-testid="stMain"] .block-container > div,
[data-testid="stMain"] .block-container > div > div {{
    flex:1 1 auto; min-height:0; width:100%; display:flex; flex-direction:column; }}
[data-testid="stMain"] .block-container > div > div > [data-testid="stVerticalBlock"] {{
    flex:1 1 auto; min-height:0; display:flex; flex-direction:column; gap:0; }}
[data-testid="stMain"] .block-container > div > div > [data-testid="stVerticalBlock"]
    > [data-testid="stElementContainer"] {{ flex:0 0 auto; }}

/* The hero row absorbs the slack; the role row is rigid. Marker classes are
   used rather than child position so that adding an element to the page
   cannot silently retarget these rules. */
[data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.nl-hero) {{
    flex:1 1 auto; min-height:0; align-items:stretch; }}
[data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.nl-role) {{
    flex:0 0 auto; align-items:end; padding-top:2.2vh; }}
[data-testid="stMain"] [data-testid="stHorizontalBlock"] {{ align-items:stretch; }}
[data-testid="stMain"] [data-testid="stColumn"] > div,
[data-testid="stMain"] [data-testid="stColumn"] > div > [data-testid="stVerticalBlock"] {{
    height:100%; display:flex; flex-direction:column; justify-content:center; gap:0; }}

.nl-wrap {{ position:relative; z-index:2; }}
.nl-head {{ display:flex; align-items:center; justify-content:space-between; }}
.nl-brand {{ display:flex; align-items:center; gap:14px;
    animation:heroIn .6s cubic-bezier(.16,.84,.44,1) both; }}
.mark {{ display:inline-flex; align-items:center; justify-content:center;
    animation:logoSettle .8s cubic-bezier(.16,.84,.44,1) both; }}
.wordmark {{ font-size:clamp(24px,min(2.4vw,4.4vh),46px); font-weight:800;
    letter-spacing:-.04em; color:{PAPER}; line-height:1; }}
.nl-mono {{ font-family:{MONO}; font-size:clamp(8.5px,min(.62vw,1.2vh),10.5px);
    letter-spacing:.14em; color:#6E666B; text-align:right; }}

/* ---- left: the argument --------------------------------------------- */
.hero-h {{ font-size:clamp(30px,min(3.4vw,6.4vh),68px); font-weight:800;
    letter-spacing:-.035em; line-height:1.04; color:{PAPER};
    animation:heroIn .7s cubic-bezier(.16,.84,.44,1) both; }}
.hero-h .tail {{ white-space:nowrap; }}
.hero-h .dot-accent {{ display:inline-block; width:11px; height:11px; border-radius:50%;
    background:{GRADIENT}; margin-left:9px; vertical-align:middle; }}
.hero-p {{ font-size:clamp(12px,min(1.05vw,2.0vh),18.5px); line-height:1.55;
    color:#B4ADB1; max-width:44ch; margin-top:2.2vh; }}
.impact {{ display:grid; grid-template-columns:repeat(4,1fr); gap:0; margin-top:3.4vh;
    border-top:1px solid rgba(255,255,255,.10); }}
.impact div {{ padding:1.6vh 14px 0 0; border-right:1px solid rgba(255,255,255,.08); }}
.impact div:last-child {{ border-right:none; padding-right:0; }}
.impact .k {{ font-family:{MONO}; font-size:clamp(8px,min(.6vw,1.15vh),10.5px);
    letter-spacing:.07em; color:{ACCENT_A}; font-weight:600; }}
.impact .v {{ font-size:clamp(10.5px,min(.8vw,1.5vh),14.5px); color:#B4ADB1;
    margin-top:.7vh; line-height:1.4; }}

/* ---- right: the two sides ------------------------------------------- */
.eyebrow {{ font-family:{MONO}; font-size:clamp(8.5px,min(.62vw,1.2vh),10.5px);
    letter-spacing:.14em; color:{ACCENT_A}; font-weight:600; margin-bottom:1.4vh; }}
.side {{ background:{CARD}; border:1px solid rgba(255,255,255,.09); border-radius:14px;
    padding:2.0vh 17px; height:100%; position:relative; overflow:hidden;
    display:flex; flex-direction:column;
    transition:transform .18s ease, border-color .18s ease; }}
.side::before {{ content:""; position:absolute; left:0; right:0; top:0; height:2px;
    background:var(--accent); opacity:0; transition:opacity .18s ease; }}
.side:hover {{ transform:translateY(-2px); border-color:rgba(255,255,255,.20); }}
.side:hover::before {{ opacity:1; }}
.side .ix {{ font-family:{MONO}; font-size:clamp(8px,min(.58vw,1.1vh),10px);
    letter-spacing:.12em; color:#8A8289; }}
.side .ix b {{ color:var(--accent); font-weight:600; }}
.side h3 {{ font-size:clamp(14px,min(1.18vw,2.25vh),22px); font-weight:800;
    letter-spacing:-.025em; margin:1.0vh 0 .8vh; line-height:1.14; color:{PAPER}; }}
.side p {{ font-size:clamp(10.5px,min(.78vw,1.5vh),14.5px); line-height:1.55;
    color:#B4ADB1; margin:0; }}
.side ul {{ list-style:none; padding:0; margin:1.6vh 0 0; }}
.side li {{ display:flex; gap:8px; align-items:flex-start;
    font-size:clamp(10px,min(.74vw,1.42vh),14px); line-height:1.45; color:#B4ADB1;
    margin-bottom:.9vh; }}
.side li span {{ color:var(--accent); font-weight:700; }}
.side .box {{ margin-top:auto; padding-top:1.6vh;
    border-top:1px solid rgba(255,255,255,.09);
    display:flex; align-items:baseline; gap:9px; }}
.side .box b {{ font-family:{MONO}; font-size:clamp(15px,min(1.35vw,2.6vh),27px);
    font-weight:700; letter-spacing:-.02em; color:{PAPER}; }}
.side .box span {{ font-size:clamp(9.5px,min(.7vw,1.34vh),11.5px); color:#8A8289;
    line-height:1.4; }}

/* ---- foot: the bridge line and the role choice ---------------------- */
.nl-bridge {{ font-size:clamp(11px,min(.86vw,1.65vh),14px); color:#8A8289;
    letter-spacing:-.01em; padding-bottom:.6vh; }}
.nl-bridge b {{ color:{PAPER}; font-weight:700; }}
.rolecard .rl {{ font-family:{MONO}; font-size:clamp(8px,min(.58vw,1.1vh),10px);
    letter-spacing:.12em; color:#8A8289; margin-bottom:.7vh; }}
/* Buttons are real Streamlit widgets, restyled for the dark ground rather
   than redrawn in HTML - a link would lose the session state that carries the
   chosen role into the app. */
[data-testid="stMain"] .stButton > button {{ border-radius:10px; font-weight:600;
    font-size:clamp(11.5px,min(.84vw,1.6vh),14px); padding:1.15vh 12px;
    border:1px solid rgba(255,255,255,.16); background:rgba(255,255,255,.06);
    color:{PAPER}; }}
[data-testid="stMain"] .stButton > button:hover {{ border-color:{ACCENT_A}; color:{PAPER}; }}
[data-testid="stMain"] .stButton > button[kind="primary"] {{ background:{GRADIENT};
    border:none; color:#fff; }}

/* ---- the drop contours behind the page ------------------------------ */
.hero-art {{ position:absolute; right:-7vw; bottom:-10vh; top:auto;
    width:min(40vw,540px); height:min(70vh,540px); opacity:.55;
    pointer-events:none; z-index:0; }}
.ring {{ transform-origin:260px 258px; }}
.ring-a {{ animation:breathe 9s ease-in-out infinite; }}
.ring-b {{ animation:breathe 9s ease-in-out infinite; animation-delay:-4.5s; }}
@keyframes heroIn {{ from {{ opacity:0; transform:translateY(14px); }} to {{ opacity:1; transform:none; }} }}
@keyframes logoSettle {{ from {{ opacity:0; transform:scale(.7); }} to {{ opacity:1; transform:scale(1); }} }}
@keyframes breathe {{ 0%,100% {{ transform:scale(1); opacity:1; }} 50% {{ transform:scale(1.035); opacity:.72; }} }}
@media (prefers-reduced-motion: reduce) {{ .nl-brand,.mark,.ring,.hero-h {{ animation:none !important; }} }}

/* Below a landscape width there is no landscape to fit: the page goes back
   to being a document rather than clipping itself. */
@media (max-width:900px) {{
    [data-testid="stMain"] .block-container {{ height:auto; overflow:auto; }}
    [data-testid="stMain"] [data-testid="stColumn"] > div,
    [data-testid="stMain"] [data-testid="stColumn"] > div > [data-testid="stVerticalBlock"] {{
        justify-content:flex-start; }}
    .hero-art {{ display:none; }} .impact {{ grid-template-columns:repeat(2,1fr); }} }}
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
        f"<div class='nl-wrap'>{_rings()}"
        f"<div class='nl-head'>"
        f"<div class='nl-brand'>{_logo(46)}<span class='wordmark'>nectar</span></div>"
        f"<div class='nl-mono'>{POSITIONING}</div></div></div>",
        unsafe_allow_html=True)

    left, right = st.columns([1.04, 1], gap="large")

    with left:
        st.markdown(
            f"<div class='nl-wrap nl-hero'>"
            f"<div class='hero-h'>Cross-pollinating brands, creators and "
            f"better <span class='tail'>outcomes."
            f"<span class='dot-accent'></span></span></div>"
            f"<div class='hero-p'>Nectar matches brands and creators where "
            f"audience, content, campaign and commercial fit intersect - "
            f"creating stronger partnerships and better value for both "
            f"sides.</div>"
            f"<div class='impact'>{impacts}</div></div>",
            unsafe_allow_html=True)

    with right:
        st.markdown("<div class='nl-wrap'><div class='eyebrow'>BUILT FOR BOTH "
                    "SIDES</div></div>", unsafe_allow_html=True)
        a, b = st.columns(2, gap="small")
        for col, s_ in zip((a, b), SIDES):
            with col:
                points = "".join(f"<li><span>✓</span>{p}</li>" for p in s_["points"])
                st.markdown(
                    f"<div class='nl-wrap'><div class='side' style='--accent:{s_['accent']}'>"
                    f"<div class='ix'><b>{s_['index']}</b> / {s_['label']}</div>"
                    f"<h3>{s_['head']}</h3><p>{s_['body']}</p><ul>{points}</ul>"
                    f"<div class='box'><b>{s_['impact'][0]}</b>"
                    f"<span>{s_['impact'][1]}</span></div></div></div>",
                    unsafe_allow_html=True)

    chosen = None
    foot_l, foot_r = st.columns([1.04, 1], gap="large")
    with foot_l:
        st.markdown(
            "<div class='nl-wrap nl-role nl-bridge'>The middleman disappears. "
            "<b>The reasoning doesn't.</b></div>", unsafe_allow_html=True)
    with foot_r:
        c1, c2 = st.columns(2, gap="small")
        for col, (role, label, cta) in zip(
                (c1, c2),
                [("brand", "FOR BRANDS", "Enter Brand OS"),
                 ("creator", "FOR CREATORS", "Enter Creator OS")]):
            with col:
                st.markdown(f"<div class='nl-wrap rolecard'><div class='rl'>{label}"
                            f"</div></div>", unsafe_allow_html=True)
                if st.button(cta, key=f"enter_{role}",
                             type="primary" if role == "brand" else "secondary",
                             use_container_width=True):
                    chosen = role
    return chosen
