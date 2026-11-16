"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import ksw_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.22), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _vline(fig, x, text, color, dash="dash", position="top"):
    fig.add_vline(x=x, line=dict(color=color, width=1.5, dash=dash), annotation_text=text, annotation_position=position, annotation_font_color=color)


def build_hand_curve(curve, r):
    """Gesamtkosten der Gruppenregel „Maßnahme bei p̂ > t“ über der Schwelle t (Stufenfunktion: sie ändert sich nur, wenn t eine Vorhersage überschreitet)."""
    fig = go.Figure(go.Scatter(x=[t for t, _ in curve], y=[c for _, c in curve], mode="lines", line=dict(color=C.COLOR_RAW, width=3, shape="hv"),
                               name="Gesamtkosten", hovertemplate="Schwelle %{x:.2f}: Kosten %{y:.0f}<extra></extra>"))
    _vline(fig, 1.0 / (1 + r), f"1/(1+r) = {1 / (1 + r):.3f}".replace(".", ","), C.COLOR_ORACLE)
    _vline(fig, 0.5, "0,5", C.COLOR_FIXED, position="bottom")
    fig.update_xaxes(title_text="Schwelle t (Maßnahme bei p̂ > t)", range=[0, 1])
    fig.update_yaxes(title_text="Gesamtkosten")
    return _base(fig, 320)


def build_cost_curve(grid, curves, r, r_assumed=None):
    """Erwartete Kosten je 1000 Entscheidungen über der Schwelle t; `curves`: Liste (Name, Werte, Farbe, Strichart)."""
    fig = go.Figure()
    for name, values, color, dash in curves:
        fig.add_trace(go.Scatter(x=list(grid), y=list(values), mode="lines", line=dict(color=color, width=3, dash=dash), name=name,
                                 hovertemplate=f"{name}, Schwelle %{{x:.2f}}: %{{y:.0f}}<extra></extra>"))
    _vline(fig, 1.0 / (1 + r), f"1/(1+r) = {1 / (1 + r):.3f}".replace(".", ","), C.COLOR_ORACLE)
    if r_assumed is not None and r_assumed != r:
        _vline(fig, 1.0 / (1 + r_assumed), f"1/(1+r′) = {1 / (1 + r_assumed):.3f}".replace(".", ","), C.COLOR_TUNED, dash="dot", position="top left")
    _vline(fig, 0.5, "0,5", C.COLOR_FIXED, position="bottom")
    fig.update_xaxes(title_text="Schwelle t (Maßnahme bei p̂ > t)", range=[0, 1])
    fig.update_yaxes(title_text="Erwartete Kosten je 1000 Entscheidungen", rangemode="tozero")
    return _base(fig)


def build_reliability(series, title_x="Vorhersage p̂ (Mittel im Fach)"):
    """Zuverlässigkeitsdiagramm: `series` = Liste (Name, Zeilen (mittleres p̂, Häufigkeit, Zahl), Farbe). Die Diagonale ist perfekte Kalibrierung."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#808895", dash="dot", width=1.5), name="perfekt kalibriert", hoverinfo="skip"))
    for name, rows, color in series:
        if not rows:
            continue
        fig.add_trace(go.Scatter(x=[a for a, _, _ in rows], y=[b for _, b, _ in rows], mode="lines+markers", line=dict(color=color, width=2),
                                 marker=dict(color=color, size=[max(6, min(18, 5 + (c ** 0.5) / 6)) for _, _, c in rows]), name=name,
                                 customdata=[c for _, _, c in rows], hovertemplate=f"{name}: p̂ %{{x:.3f}}, beobachtet %{{y:.3f}}, %{{customdata}} Container<extra></extra>"))
    fig.update_xaxes(title_text=title_x, range=[0, 1])
    fig.update_yaxes(title_text="Beobachtete Häufigkeit Standgeld", range=[0, 1])
    return _base(fig, 380)


def build_methods_bar(labels, series):
    """Regret je 1000 Entscheidungen: `series` = Liste (Regel, Mittelwerte, Streuungen, Farbe) über den Beschriftungen (Modelle oder Größen)."""
    fig = go.Figure()
    for name, means, sds, color in series:
        fig.add_trace(go.Bar(x=list(labels), y=list(means), error_y=dict(type="data", array=list(sds), visible=True), marker_color=color, name=name,
                             hovertemplate=f"{name}: %{{y:.1f}}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Regret je 1000 Entscheidungen (Mittel ± Streuung)", rangemode="tozero")
    return _base(fig, 400)


def build_size_curves(sizes, series, events):
    """Regret über der Größe der Rekalibriermenge m (Mittel über die Trainingsmengen mit Band ± Streuung); `series` = Liste (Name, Mittel, Streuung, Farbe)."""
    fig = go.Figure()
    xs = [f"{m} ({e:.0f})" for m, e in zip(sizes, events)]
    for name, means, sds, color in series:
        means, sds = np.asarray(means), np.asarray(sds)
        fig.add_trace(go.Scatter(x=xs, y=list(means), mode="lines+markers", line=dict(color=color, width=3), name=name,
                                 error_y=dict(type="data", array=list(sds), visible=True, thickness=1, width=3), hovertemplate=f"{name}: %{{y:.1f}}<extra></extra>"))
    fig.update_xaxes(title_text="Rekalibriermenge m Container (im Mittel Container mit Standgeld in Klammern)", type="category")
    fig.update_yaxes(title_text="Regret je 1000 Entscheidungen", rangemode="tozero")
    return _base(fig, 380)
