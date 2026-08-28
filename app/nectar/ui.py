"""
Nectar components. Each function returns an HTML string; nothing here calls
Streamlit, so the markup can be unit-tested and reused between pages.

Formatting note: this product quotes money in Indian units. 4.8 lakh is
written as 4.8L and 1.2 crore as 1.2Cr, because that is how the audience for
this product reads rupees - not as 480,000.
"""
from __future__ import annotations

import html
from typing import Iterable, Sequence

from .theme import (
    ACCENT_A, ACCENT_B, AMBER, CARD, GRADIENT, GREEN, INK, INK_2, INK_3,
    LINE, LINE_2, MONO, RED, status_style,
)

def logo_svg(size: int = 22, dot: str = INK) -> str:
    """The nectar mark: a droplet with a punched-out centre.

    Built as SVG rather than a CSS border-radius trick - the trick renders as
    an ellipse at small sizes because the corner radii get clamped.
    """
    gid = f"ng{size}"
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none"
     style="transform:rotate(-18deg);flex:0 0 auto">
  <defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="{ACCENT_A}"/><stop offset="100%" stop-color="{ACCENT_B}"/>
  </linearGradient></defs>
  <path d="M12 1.6c4.7 4.6 8.2 7.7 8.2 12.1a8.2 8.2 0 1 1-16.4 0C3.8 9.3 7.3 6.2 12 1.6z"
        fill="url(#{gid})"/>
  <circle cx="12" cy="14.4" r="3.5" fill="{dot}"/>
</svg>"""



# --------------------------------------------------------------------------
# Formatters
# --------------------------------------------------------------------------

def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def platforms_of(row) -> dict:
    """The creator's platforms, with the empty ones removed.

    Parquet stores the platform split as a struct, so every creator carries
    every key in the union and the ones they are not on come back as None.
    Reading the raw dict put a "YouTube — followers" tile on the profile of a
    creator who has no YouTube account.
    """
    raw = dict(row.platforms or {})
    return {k: int(v) for k, v in raw.items() if v is not None and float(v) > 0}


def count(n) -> str:
    """Audience counts in K/M, not lakh/crore.

    Money and people are counted in different units in this product, on
    purpose: Indian creators quote reach as "284K" and fees as "20K", but a
    brand's budget is "4.8L". Formatting followers as 2.8L reads as a typo.
    """
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "—"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.0f}K"
    return f"{n:,.0f}"


def inr(n, dp: int | None = None) -> str:
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "—"
    if n >= 1e7:
        return f"₹{n / 1e7:.2f}Cr"
    if n >= 1e5:
        return f"₹{n / 1e5:.2f}L"
    if n >= 1e3:
        return f"₹{n / 1e3:.0f}K"
    if dp is not None:
        return f"₹{n:.{dp}f}"
    return f"₹{n:,.0f}"


def pct(x, dp: int = 1) -> str:
    try:
        return f"{float(x) * 100:.{dp}f}%"
    except (TypeError, ValueError):
        return "—"


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------

def page_header(title: str, subtitle: str = "", eyebrow: str = "") -> str:
    out = ["<div style='margin-bottom:20px'>"]
    if eyebrow:
        out.append(f"<div class='n-eyebrow'>{esc(eyebrow)}</div>")
    out.append(f"<div class='n-h1'>{esc(title)}</div>")
    if subtitle:
        out.append(f"<div class='n-sub'>{esc(subtitle)}</div>")
    out.append("</div>")
    return "".join(out)


def chip(label: str) -> str:
    fg, bg = status_style(label)
    return f"<span class='n-chip' style='color:{fg};background:{bg}'>{esc(label)}</span>"


def tag(label: str) -> str:
    return f"<span class='n-tag'>{esc(label)}</span>"


def avatar(initials: str, color: str, size: int = 40) -> str:
    fs = max(10, int(size * 0.34))
    return (f"<span class='n-av' style='width:{size}px;height:{size}px;"
            f"background:{esc(color)};font-size:{fs}px'>{esc(initials)}</span>")


def bar(fraction: float, color: str = GREEN, width: str = "70px") -> str:
    f = max(0.0, min(1.0, float(fraction or 0)))
    return (f"<span class='n-bar' style='display:inline-block;width:{width}'>"
            f"<i style='width:{f * 100:.0f}%;background:{color}'></i></span>")


def kpi(label: str, value: str, delta: str = "", delta_tone: str = "good") -> str:
    tone = {"good": GREEN, "warn": AMBER, "bad": RED, "flat": INK_3}.get(delta_tone, INK_3)
    d = f"<div class='dlt' style='color:{tone}'>{esc(delta)}</div>" if delta else ""
    return (f"<div class='n-kpi'><div class='lbl'>{esc(label)}</div>"
            f"<div class='val'>{esc(value)}</div>{d}</div>")


def card(inner: str, pad: str = "18px 20px") -> str:
    return f"<div class='n-card' style='padding:{pad}'>{inner}</div>"


def section(title: str, subtitle: str = "") -> str:
    s = f"<div class='n-muted' style='margin-top:2px'>{esc(subtitle)}</div>" if subtitle else ""
    return f"<div style='margin-bottom:12px'><div class='n-h2'>{esc(title)}</div>{s}</div>"


def empty_state(icon: str, title: str, body: str) -> str:
    return (f"<div class='n-empty'><div class='ic'>{esc(icon)}</div>"
            f"<div class='n-h1' style='font-size:22px'>{esc(title)}</div>"
            f"<div class='n-sub' style='max-width:360px;margin:8px auto 0'>{esc(body)}</div></div>")


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def table(headers: Sequence[str], rows: Iterable[Sequence[str]],
          aligns: Sequence[str] | None = None) -> str:
    aligns = aligns or ["left"] * len(headers)
    head = "".join(f"<th style='text-align:{a}'>{esc(h)}</th>"
                   for h, a in zip(headers, aligns))
    body = []
    for r in rows:
        tds = "".join(f"<td style='text-align:{a}'>{c}</td>"
                      for c, a in zip(r, aligns))
        body.append(f"<tr>{tds}</tr>")
    return (f"<div class='n-card' style='padding:2px 6px'>"
            f"<table class='n-table'><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>")


def creator_cell(name: str, handle: str, initials: str, color: str,
                 verified: bool = False, sub: str = "") -> str:
    tick = (f"<span style='color:{GREEN};font-size:12px;margin-left:4px'>✓</span>"
            if verified else "")
    second = esc(sub) if sub else esc(handle)
    return (f"<div style='display:flex;align-items:center;gap:10px'>"
            f"{avatar(initials, color, 32)}"
            f"<div><div style='font-weight:600;font-size:13.5px;line-height:1.3'>"
            f"{esc(name)}{tick}</div>"
            f"<div style='font-size:12px;color:{INK_3};line-height:1.3'>{second}</div>"
            f"</div></div>")


# --------------------------------------------------------------------------
# Fit tiles - the two big percentages on a creator card
# --------------------------------------------------------------------------

def fit_tile(label: str, value: float, band: str, sub: str = "") -> str:
    """The two big percentages on a creator card.

    `value` is a PERCENTILE within the campaign's candidate pool, not the raw
    composite. The composite's spread is only a few points wide, so showing it
    raw made an 86 look meaningfully better than an 84 when it was not.

    Shown to one decimal above the 99th percentile: with a pool of two thousand
    the whole top twenty rounds to 100, and three cards in a row reading "100th"
    reads as a broken number rather than a close race.
    """
    fg, bg = status_style(band)
    caption = esc(sub) if sub else f"● {esc(band)}"
    shown = f"{value:.0f}" if value < 99 else f"{value:.1f}"
    return (f"<div style='flex:1;background:{bg};border-radius:11px;padding:11px 13px'>"
            f"<div style='font-size:11.5px;color:{INK_2};font-weight:500'>{esc(label)}</div>"
            f"<div class='n-num' style='font-size:26px;color:{fg};line-height:1.25'>"
            f"{shown}<span style='font-size:15px'>th</span></div>"
            f"<div style='font-size:11.5px;color:{fg};font-weight:600'>{caption}</div>"
            f"</div>")


def metric_strip(items: Sequence[tuple[str, str]]) -> str:
    cells = "".join(
        f"<div style='flex:1;min-width:0'>"
        f"<div style='font-size:11.5px;color:{INK_3}'>{esc(l)}</div>"
        f"<div class='n-num' style='font-size:13.5px;color:{INK}'>{esc(v)}</div></div>"
        for l, v in items)
    return f"<div style='display:flex;gap:12px;margin:14px 0 12px 0'>{cells}</div>"


def reason_list(reasons: Sequence[str], tone: str = "good") -> str:
    mark, colour = ("✓", GREEN) if tone == "good" else ("!", AMBER)
    lis = "".join(
        f"<div style='display:flex;gap:8px;margin-bottom:5px;align-items:flex-start'>"
        f"<span style='color:{colour};font-size:12px;line-height:1.5'>{mark}</span>"
        f"<span style='font-size:12.5px;color:{INK_2};line-height:1.5'>{esc(r)}</span></div>"
        for r in reasons)
    return lis


def creator_card_html(r, campaign_label: str = "") -> str:
    """The Discover card's body.

    Draws NO border of its own. The card frame is a st.container(border=True)
    opened by the caller, so the real Streamlit action buttons sit inside the
    same box as this markup. The earlier version drew its own three-sided
    border and relied on a CSS :has() rule to weld the button row onto the
    bottom edge; that matched inconsistently and left some cards with their
    buttons floating outside the frame.
    """
    cats = "".join(tag(c) for c in list(r.categories)[:2])
    plats = "".join(tag(p) for p in list(r.platform_names)[:2])
    tick = (f"<span style='color:{GREEN};font-size:13px;margin-left:5px'>✓</span>"
            if bool(r.verified) else "")
    avail_colour = GREEN if r.availability == "Available" else AMBER

    flags = []
    if bool(r.verified):
        flags.append("Verified socials")
    if not bool(r.blocked):
        flags.append("No safety flags")
    flags.append(str(r.availability))
    flagline = " ".join(
        f"<span style='color:{GREEN};font-size:11.5px;margin-right:9px;white-space:nowrap'>"
        f"✓ {esc(f)}</span>" for f in flags)

    return f"""<div>
  <div style='display:flex;align-items:flex-start;gap:11px'>
    {avatar(r.initials, r.avatar_color, 40)}
    <div style='min-width:0;flex:1'>
      <div style='font-weight:700;font-size:15px;letter-spacing:-0.01em'>{esc(r.name)}{tick}</div>
      <div style='font-size:12.5px;color:{INK_3}'>{esc(r.nectar_handle)} · {esc(r.city)}</div>
    </div>
  </div>
  <div style='margin:10px 0 0 0'>{cats}{plats}</div>
  {metric_strip([
      ("Followers", count(r.followers)),
      ("Engagement", f"{r.engagement_rate * 100:.1f}%"),
      ("Rate from", inr(r.rate_from)),
  ])}
  <div style='font-size:11.5px;color:{INK_3}'>Available</div>
  <div style='font-size:12.5px;color:{avail_colour};font-weight:600;margin-bottom:12px'>
    {esc(r.available_window)}</div>
  <div style='display:flex;gap:10px;margin-bottom:14px'>
    {fit_tile("Campaign Fit", r.campaign_fit, r.fit_band,
              f"{r.fit_band} · #{int(r.rank_best):,} of {int(r.eligible_pool_size):,}")}
    {fit_tile("Organisation Fit", r.org_fit, r.org_band, f"{r.org_band} · percentile")}
  </div>
  <div class='n-h3' style='margin-bottom:7px'>Why this creator?</div>
  {reason_list(list(r.match_reasons))}
  <div style='margin:11px 0 4px 0;line-height:2'>{flagline}</div>
</div>"""
