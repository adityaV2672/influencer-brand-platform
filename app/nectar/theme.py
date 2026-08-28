"""
The Nectar design system, expressed as Streamlit-compatible CSS.

Every token here was read off the reference build rather than guessed:
ink #181316, paper #FFFDFB, a vertical #FF6A2C -> #FF3E93 accent gradient,
Inter for prose at 800/-0.03em for display sizes, JetBrains Mono for every
number and every column header, and a 12px corner radius.

Two structural decisions:

  * Streamlit's own chrome (header, toolbar, footer, the default sidebar
    nav list) is removed, not restyled. Half-hiding it produces a page that
    looks like a Streamlit app wearing a costume.
  * Cards are drawn as HTML and interactive controls are Streamlit widgets
    pulled into them with negative margins. Streamlit cannot put a real
    button inside an HTML string, so the seam is hidden rather than denied.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------
INK = "#181316"
INK_2 = "#4A4247"
INK_3 = "#8A8289"
PAPER = "#FFFDFB"
CARD = "#FFFFFF"
LINE = "#EAE6E3"
LINE_2 = "#F2EEEB"

ACCENT_A = "#FF6A2C"
ACCENT_B = "#FF3E93"
GRADIENT = f"linear-gradient(180deg, {ACCENT_A} 0%, {ACCENT_B} 100%)"

GREEN = "#1F8A6E"
GREEN_BG = "#E8F4F0"
AMBER = "#B8860B"
AMBER_BG = "#FBF3E0"
RED = "#C2413F"
RED_BG = "#FBECEC"
BLUE = "#2E6FB7"
BLUE_BG = "#EAF1F9"

SANS = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
MONO = "'JetBrains Mono', 'SFMono-Regular', Menlo, monospace"

SIDEBAR_W = 224

# Status -> (text colour, background). Reserved: never reused for series colour.
STATUS_STYLE = {
    "Live": (GREEN, GREEN_BG), "Active": (GREEN, GREEN_BG),
    "Accepted": (GREEN, GREEN_BG), "Approved": (GREEN, GREEN_BG),
    "Paid": (GREEN, GREEN_BG), "Available": (GREEN, GREEN_BG),
    "High fit": (GREEN, GREEN_BG),
    "Draft": (AMBER, AMBER_BG), "Countered": (AMBER, AMBER_BG),
    "New": (AMBER, AMBER_BG), "Busy": (AMBER, AMBER_BG),
    "Medium fit": (AMBER, AMBER_BG),
    "Declined": (RED, RED_BG), "Blocked": (RED, RED_BG),
    "Unavailable": (RED, RED_BG), "Low fit": (INK_3, LINE_2),
    "Viewed": (BLUE, BLUE_BG), "Sent": (BLUE, BLUE_BG),
    "In Production": (BLUE, BLUE_BG), "In production": (BLUE, BLUE_BG),
    "Delivered": (BLUE, BLUE_BG), "Completed": (INK_3, LINE_2),
    "Drafted": (INK_3, LINE_2),
}


def status_style(label: str) -> tuple[str, str]:
    return STATUS_STYLE.get(str(label), (INK_2, LINE_2))


# --------------------------------------------------------------------------
# CSS
# --------------------------------------------------------------------------

# Streamlit renders this through its markdown pipeline. Two constraints follow
# and both were learned the hard way, by watching the entire stylesheet render
# as body text: the string must OPEN with <style> (a leading <link> makes
# markdown treat the block as a paragraph), and it must contain no blank lines
# (a blank line ends the HTML block and everything after it is escaped). The
# blank-line rule is enforced at injection time in shell.inject_css, so this
# source stays readable.
CSS = f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');
/* ---------- strip Streamlit chrome ---------- */
#MainMenu, footer, header[data-testid="stHeader"] {{ display: none !important; }}
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] {{ display: none !important; }}
[data-testid="stSidebarNav"] {{ display: none !important; }}
[data-testid="stSidebarHeader"] {{ height: 0; padding: 0; }}
[data-testid="stSidebarCollapseButton"] {{ display: none !important; }}
.stAppDeployButton {{ display: none !important; }}

/* ---------- page ---------- */
html, body, [data-testid="stAppViewContainer"] {{
    background: {PAPER};
    font-family: {SANS};
    color: {INK};
}}
[data-testid="stMain"] .block-container {{
    padding: 2.1rem 2.6rem 4rem 2.6rem;
    max-width: 1320px;
}}
* {{ -webkit-font-smoothing: antialiased; }}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {{
    background: {INK};
    width: {SIDEBAR_W}px !important;
    min-width: {SIDEBAR_W}px !important;
    border-right: none;
}}
[data-testid="stSidebar"] > div:first-child {{ padding: 0; }}
[data-testid="stSidebarUserContent"] {{ padding: 0 0 1rem 0; }}
[data-testid="stSidebar"] * {{ color: rgba(255,255,255,0.55); }}

.nectar-brand {{
    display: flex; align-items: center; gap: 9px;
    padding: 20px 18px 18px 18px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 10px;
}}
.nectar-brand .wordmark {{
    font-size: 19px; font-weight: 700; color: #fff !important;
    letter-spacing: -0.03em;
}}


/* sidebar nav built from st.page_link */
[data-testid="stSidebar"] [data-testid="stPageLink"] a,
[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] {{
    padding: 8px 14px; margin: 1px 12px; border-radius: 9px;
    font-size: 14px; font-weight: 500; gap: 11px;
    color: rgba(255,255,255,0.55) !important;
    transition: background 120ms ease, color 120ms ease;
}}
[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:hover {{
    background: rgba(255,255,255,0.06);
    color: rgba(255,255,255,0.85) !important;
}}
[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] span {{
    color: inherit !important; font-weight: 500;
}}
[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][aria-current],
[data-testid="stSidebar"] li:has(a[aria-current]) a {{
    background: rgba(255,255,255,0.10);
    color: #fff !important;
}}
.nectar-navgroup {{
    font-family: {MONO}; font-size: 10px; letter-spacing: 0.10em;
    text-transform: uppercase; color: rgba(255,255,255,0.32) !important;
    padding: 16px 26px 6px 26px;
}}
.nectar-sidebar-rule {{
    height: 1px; background: rgba(255,255,255,0.08); margin: 14px 18px;
}}
.nectar-account {{
    display: flex; align-items: center; gap: 10px; padding: 14px 18px 8px 18px;
    border-top: 1px solid rgba(255,255,255,0.08); margin-top: 8px;
}}
.nectar-account .av {{
    width: 30px; height: 30px; border-radius: 50%; flex: 0 0 30px;
    background: {GRADIENT}; color: #fff !important;
    font-size: 11px; font-weight: 700; font-family: {MONO};
    display: flex; align-items: center; justify-content: center;
}}
.nectar-account .nm {{ font-size: 13px; font-weight: 600; color: #fff !important; line-height: 1.25; }}
.nectar-account .rl {{ font-size: 11.5px; color: rgba(255,255,255,0.45) !important; }}

/* Creator picker. Streamlit 1.6x renders a selectbox as a react-aria ComboBox
   whose value lives in an <input>, not as a baseweb select - styling the
   baseweb selector left a white box with invisible text on the dark rail. */
[data-testid="stSidebar"] [data-testid="stSelectbox"] {{ padding: 0 18px 4px 18px; }}
[data-testid="stSidebar"] [data-testid="stSelectbox"] label {{ display: none; }}
[data-testid="stSidebar"] [data-testid="stSelectbox"] [role="group"] {{
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 9px !important; min-height: 34px;
}}
[data-testid="stSidebar"] [data-testid="stSelectbox"] input {{
    color: rgba(255,255,255,0.85) !important;
    background: transparent !important;
    font-size: 12.5px !important; font-weight: 500;
}}
[data-testid="stSidebar"] [data-testid="stSelectbox"] svg {{
    fill: rgba(255,255,255,0.5); color: rgba(255,255,255,0.5);
}}
/* the role switch is the only sidebar button */
[data-testid="stSidebar"] .stButton > button {{
    width: calc(100% - 36px); margin: 8px 18px 4px 18px;
    background: transparent; border: 1px solid rgba(255,255,255,0.16);
    color: rgba(255,255,255,0.72) !important;
    border-radius: 9px; font-size: 12.5px; font-weight: 500; padding: 7px 10px;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    border-color: rgba(255,255,255,0.34); color: #fff !important;
    background: rgba(255,255,255,0.05);
}}

/* ---------- type ---------- */
.n-h1 {{ font-size: 30px; font-weight: 800; letter-spacing: -0.03em;
        color: {INK}; line-height: 1.15; margin: 0; }}
.n-sub {{ font-size: 14.5px; color: {INK_2}; margin: 6px 0 0 0; }}
.n-eyebrow {{ font-family: {MONO}; font-size: 10.5px; letter-spacing: 0.11em;
             text-transform: uppercase; color: {INK_3}; margin-bottom: 4px; }}
.n-h2 {{ font-size: 17px; font-weight: 700; letter-spacing: -0.02em;
        color: {INK}; margin: 0; }}
.n-h3 {{ font-size: 14px; font-weight: 650; color: {INK}; margin: 0; }}
.n-muted {{ font-size: 13px; color: {INK_3}; }}
.n-num {{ font-family: {MONO}; font-weight: 700; letter-spacing: -0.02em; }}

/* ---------- cards ---------- */
.n-card {{
    background: {CARD}; border: 1px solid {LINE}; border-radius: 14px;
    padding: 18px 20px;
}}
.n-kpi {{
    background: {CARD}; border: 1px solid {LINE}; border-radius: 14px;
    padding: 16px 18px 14px 18px; height: 100%;
}}
.n-kpi .lbl {{ font-family: {MONO}; font-size: 10.5px; letter-spacing: 0.10em;
              text-transform: uppercase; color: {INK_3}; }}
.n-kpi .val {{ font-family: {MONO}; font-size: 29px; font-weight: 700;
              letter-spacing: -0.03em; color: {INK}; line-height: 1.3; }}
.n-kpi .dlt {{ font-size: 12px; font-weight: 500; }}

/* A st.container(border=True) is the only way to put a real Streamlit widget
   (a chart, a button, an input) inside a card. Styling the border wrapper
   turns that container into the Nectar card, so pages can mix HTML and
   widgets inside one visual object instead of an HTML card that stops
   wherever the markdown call ended. */
[data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {{
    background: {CARD}; border: 1px solid {LINE}; border-radius: 14px;
}}
[data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] {{
    padding: 18px 20px; gap: 0.55rem;
}}
/* ---------- chips ---------- */
.n-chip {{
    display: inline-block; padding: 3px 9px; border-radius: 999px;
    font-size: 11.5px; font-weight: 600; line-height: 1.5;
}}
.n-tag {{
    display: inline-block; padding: 3px 9px; border-radius: 7px;
    font-size: 11.5px; font-weight: 500; color: {INK_2};
    background: {LINE_2}; border: 1px solid {LINE}; margin-right: 5px;
}}

/* ---------- avatars ---------- */
.n-av {{
    border-radius: 50%; color: #fff; font-family: {MONO}; font-weight: 700;
    display: inline-flex; align-items: center; justify-content: center;
    flex: 0 0 auto;
}}

/* ---------- tables ---------- */
.n-table {{ width: 100%; border-collapse: collapse; }}
.n-table th {{
    font-family: {MONO}; font-size: 10.5px; letter-spacing: 0.09em;
    text-transform: uppercase; color: {INK_3}; font-weight: 500;
    text-align: left; padding: 12px 14px; border-bottom: 1px solid {LINE};
}}
.n-table td {{
    padding: 14px; border-bottom: 1px solid {LINE_2}; font-size: 13.5px;
    color: {INK}; vertical-align: middle;
}}
.n-table tr:last-child td {{ border-bottom: none; }}
.n-table td.num {{ font-family: {MONO}; font-weight: 500; }}

/* ---------- progress ---------- */
.n-bar {{ height: 5px; border-radius: 3px; background: {LINE_2};
         overflow: hidden; min-width: 60px; }}
.n-bar > i {{ display: block; height: 100%; border-radius: 3px; }}

/* ---------- buttons ---------- */
[data-testid="stMain"] .stButton > button {{
    border-radius: 10px; font-size: 13px; font-weight: 600;
    padding: 8px 16px; border: 1px solid {LINE};
    background: {CARD}; color: {INK};
    transition: border-color 120ms ease, background 120ms ease;
}}
[data-testid="stMain"] .stButton > button:hover {{
    border-color: {INK_3}; color: {INK}; background: {CARD};
}}
[data-testid="stMain"] .stButton > button[kind="primary"],
[data-testid="stMain"] .stButton > button[data-testid="stBaseButton-primary"] {{
    background: {GRADIENT}; color: #fff; border: none; font-weight: 600;
}}
[data-testid="stMain"] .stButton > button[kind="primary"]:hover {{
    filter: brightness(1.06); color: #fff;
}}
[data-testid="stMain"] .stDownloadButton > button {{
    border-radius: 10px; font-size: 13px; font-weight: 600;
    border: 1px solid {LINE}; background: {CARD}; color: {INK};
}}

/* ---------- inputs ---------- */
[data-testid="stMain"] input, [data-testid="stMain"] textarea,
[data-testid="stMain"] [data-testid="stSelectbox"] [role="group"],
[data-testid="stMain"] [data-testid="stMultiSelect"] [role="group"] {{
    border-radius: 10px !important; font-size: 13.5px !important;
    border-color: {LINE} !important; background: {CARD} !important;
}}
[data-testid="stMain"] [data-testid="stWidgetLabel"] p {{
    font-size: 12.5px; font-weight: 600; color: {INK_2};
}}

/* Segmented control = the Best match / Engagement / Reach toggle, and the
   All / Live / Draft / Completed pills on Campaigns.
   Streamlit renders it as a button group, NOT as a "stSegmentedControl"
   testid; each option carries data-variant="segmented_control" and the active
   one carries data-selected="true". Targeting the testid that does not exist
   left the active pill with Streamlit's default tinted background instead of
   the black pill the design uses. */
[data-testid="stMain"] [data-testid="stButtonGroup"] {{ flex-wrap: nowrap; gap: 4px; }}
[data-testid="stMain"] button[data-variant="segmented_control"] {{
    border-radius: 999px !important; font-size: 12.5px; font-weight: 600;
    border: none !important; background: transparent !important;
    color: {INK_2} !important; white-space: nowrap; padding: 6px 15px !important;
}}
[data-testid="stMain"] button[data-variant="segmented_control"]:hover {{
    background: {LINE_2} !important;
}}
[data-testid="stMain"] button[data-variant="segmented_control"][data-selected="true"] {{
    background: {INK} !important;
}}
[data-testid="stMain"] button[data-variant="segmented_control"][data-selected="true"],
[data-testid="stMain"] button[data-variant="segmented_control"][data-selected="true"] p {{
    color: #fff !important;
}}
/* pill filter tabs */
[data-testid="stMain"] [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: none; }}
[data-testid="stMain"] [data-baseweb="tab"] {{
    border-radius: 999px; padding: 6px 15px; font-size: 13px; font-weight: 600;
    color: {INK_2}; background: transparent; border: none;
}}
[data-testid="stMain"] [data-baseweb="tab"][aria-selected="true"] {{
    background: {INK}; color: #fff;
}}
[data-testid="stMain"] [data-baseweb="tab-highlight"],
[data-testid="stMain"] [data-baseweb="tab-border"] {{ display: none; }}

/* checkbox / radio labels in the filter rail */
[data-testid="stMain"] [data-testid="stCheckbox"] label p {{
    font-size: 13px; color: {INK_2}; font-weight: 500;
}}

/* ---------- misc ---------- */
hr, [data-testid="stDivider"] hr {{ border-color: {LINE}; margin: 1.4rem 0; }}
[data-testid="stExpander"] {{ border: 1px solid {LINE}; border-radius: 12px; }}
[data-testid="stExpander"] summary {{ font-size: 13.5px; font-weight: 600; }}
.n-empty {{ text-align: center; padding: 90px 20px; }}
.n-empty .ic {{
    width: 52px; height: 52px; border-radius: 14px; border: 1px solid {LINE};
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 21px; color: {INK_3}; margin-bottom: 18px;
}}
</style>
"""
