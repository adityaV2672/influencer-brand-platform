"""
Plotly charts in the Nectar house style.

Three rules, applied everywhere:
  * one hue per measure, assigned by slot and never cycled, so hiding a series
    never repaints the others;
  * no second y-axis - two measures on different scales get two charts;
  * status colours (green/amber/red) are reserved for status and are never
    reused as series colours.
"""
from __future__ import annotations

import plotly.graph_objects as go

from .theme import AMBER, GREEN, INK, INK_2, INK_3, LINE, LINE_2, MONO, SANS

SERIES = {
    "reach": "#1F8A6E",
    "engagement": "#D4A017",
    "spend": "#9B9298",
    "primary": "#2E6FB7",
    "muted": "#D9D3D7",
}


def _base(fig: go.Figure, height: int = 300, legend: bool = True,
          ytitle: str = "", xtitle: str = "") -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=6, r=6, t=26, b=6),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=SANS, size=12, color=INK_2),
        hoverlabel=dict(bgcolor="#fff", font_size=12, bordercolor=LINE,
                        font_family=SANS),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11.5)),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=LINE,
                     tickfont=dict(size=11, color=INK_3, family=MONO),
                     title_text=xtitle)
    fig.update_yaxes(showgrid=True, gridcolor=LINE_2, zeroline=False,
                     linecolor="rgba(0,0,0,0)",
                     tickfont=dict(size=11, color=INK_3, family=MONO),
                     title_text=ytitle)
    return fig


def multiline(df, x: str, series: dict[str, str], height: int = 300) -> go.Figure:
    """series = {column: legend label}. Each measure is normalised to its own
    maximum so three quantities on wildly different scales share one axis
    honestly - the axis is therefore unlabelled and the hover carries the
    real number."""
    fig = go.Figure()
    for col, label in series.items():
        vals = df[col].astype(float)
        peak = vals.max() or 1
        fig.add_trace(go.Scatter(
            x=df[x], y=vals / peak, name=label, mode="lines+markers",
            line=dict(color=SERIES.get(col, SERIES["primary"]), width=2.2,
                      shape="spline", smoothing=0.6),
            marker=dict(size=5),
            customdata=vals,
            hovertemplate=f"<b>{label}</b><br>%{{x}}: %{{customdata:,.0f}}<extra></extra>",
        ))
    _base(fig, height)
    fig.update_yaxes(showticklabels=False, range=[0, 1.12])
    return fig


def funnel_bars(labels, values, height: int = 300) -> go.Figure:
    peak = max(values) if len(values) else 1
    fig = go.Figure(go.Bar(
        x=list(labels), y=list(values),
        marker=dict(color=[f"rgba(31,138,110,{0.35 + 0.55 * (v / peak):.2f})" for v in values],
                    line=dict(width=0)),
        text=[f"{v:,}" for v in values], textposition="outside",
        textfont=dict(family=MONO, size=11, color=INK_2),
        hovertemplate="%{x}: %{y:,} requests<extra></extra>",
    ))
    _base(fig, height, legend=False)
    fig.update_yaxes(showticklabels=False, showgrid=False,
                     range=[0, peak * 1.22])
    fig.update_xaxes(tickfont=dict(size=10.5, color=INK_3, family=SANS))
    return fig


def compare_pair(predicted: float, actual: float, colour: str,
                 height: int = 190) -> go.Figure:
    """Two bars: what the model said, and what happened."""
    fig = go.Figure(go.Bar(
        x=["Predicted", "Actual"], y=[predicted, actual],
        marker=dict(color=[SERIES["muted"], colour], line=dict(width=0)),
        width=[0.52, 0.52],
        text=[f"{predicted:,.0f}", f"{actual:,.0f}"], textposition="outside",
        textfont=dict(family=MONO, size=12, color=INK),
        hovertemplate="%{x}: %{y:,.0f}<extra></extra>",
    ))
    _base(fig, height, legend=False)
    fig.update_yaxes(showticklabels=False, showgrid=False,
                     range=[0, max(predicted, actual) * 1.28])
    return fig


def hbars(labels, values, colour: str = GREEN, height: int = 260,
          suffix: str = "%") -> go.Figure:
    fig = go.Figure(go.Bar(
        x=list(values)[::-1], y=list(labels)[::-1], orientation="h",
        marker=dict(color=colour, line=dict(width=0)),
        text=[f"{v:.0f}{suffix}" for v in list(values)[::-1]],
        textposition="outside",
        textfont=dict(family=MONO, size=11, color=INK_2),
        hovertemplate="%{y}: %{x:.1f}" + suffix + "<extra></extra>",
    ))
    _base(fig, height, legend=False)
    fig.update_xaxes(showticklabels=False, showgrid=False,
                     range=[0, max(values) * 1.25 if len(values) else 1])
    fig.update_yaxes(tickfont=dict(size=12, color=INK_2, family=SANS))
    return fig


def column(labels, values, colour: str = GREEN, height: int = 240,
           fmt: str = ",.0f") -> go.Figure:
    fig = go.Figure(go.Bar(
        x=list(labels), y=list(values),
        marker=dict(color=colour, line=dict(width=0)), width=0.55,
        hovertemplate="%{x}: %{y:" + fmt + "}<extra></extra>",
    ))
    _base(fig, height, legend=False)
    return fig


def scatter(df, x: str, y: str, text: str, size: str | None = None,
            colour: str = SERIES["primary"], height: int = 300,
            xtitle: str = "", ytitle: str = "") -> go.Figure:
    import numpy as np
    marker = dict(color=colour, opacity=0.72, line=dict(width=1, color="#fff"))
    if size:
        marker["size"] = np.clip(np.log10(df[size].clip(lower=1)) * 3.2, 6, 20)
    else:
        marker["size"] = 8
    fig = go.Figure(go.Scatter(
        x=df[x], y=df[y], mode="markers", marker=marker, text=df[text],
        hovertemplate="<b>%{text}</b><br>%{x}<br>%{y}<extra></extra>",
    ))
    _base(fig, height, legend=False, xtitle=xtitle, ytitle=ytitle)
    return fig


CONFIG = {"displayModeBar": False, "staticPlot": False}
