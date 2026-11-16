"""Kalibrierung und Schwelle – aus einer Wahrscheinlichkeit wird eine Entscheidung - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Vehikel: Container-Verweilzeit, Ergebnis „Standgeld fällig“ (Verweilzeit über 7 Tage). Die Live-Rechnung passt eine Trainingsmenge an; die Messungen über 40 Trainingsmengen
sind vorgerechnet (`ksw_frozen.json`, erzeugt von `tools/ksw_make_frozen.py`). Aufbau und Regeln: DEMO-PLAYBOOK §2.

Lauffähig mit: streamlit run app.py
"""

from fractions import Fraction

import streamlit as st

import ksw_constants as C
import ksw_core as K
import ksw_evaluation as E
import ksw_frozen as FZ
import ksw_hand as H
import ksw_models as M
from ksw_core import BINNINGS
from ksw_layout import dunkles_design, kopf_css
from ksw_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from ksw_visualization import (
    build_cost_curve,
    build_hand_curve,
    build_methods_bar,
    build_reliability,
    build_size_curves,
)

st.set_page_config(page_title="Kalibrierung und Schwelle – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _live(settings):
    return E.analyse_instance(settings)


@st.cache_data(show_spinner=False)
def _population():
    return E.population()


@st.cache_data(show_spinner=False)
def _matrix():
    return E.regret_matrix(_population())


@st.cache_data(show_spinner=False)
def _population_curve(r):
    return E.cost_curve(_population(), _population(), r)


FZ_DATA = FZ.load()
REPS = E.reps(FZ_DATA)
LABEL = M.MODEL_LABELS
HAND_DEFAULT = ((20, 10, 0), (30, 10, 2), (40, 10, 7))     # Vorhersage in Prozent, Container, davon mit Standgeld


def _reset_hand():
    for i, (p, n, e) in enumerate(HAND_DEFAULT):
        st.session_state[f"hand_p{i}"], st.session_state[f"hand_n{i}"], st.session_state[f"hand_e{i}"] = p, n, e


def _show(verdict):
    kind, text = verdict
    getattr(st, kind)(text)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in ksw_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🎚️ Kalibrierung und Schwelle – aus einer Wahrscheinlichkeit wird eine Entscheidung")
st.markdown(
    """
Ein Modell sagt für jeden Container eine **Wahrscheinlichkeit** voraus, dass Standgeld fällig wird (Verweilzeit über 7 Tage). Eine **Entscheidung** (den Kunden vorwarnen oder nicht) ergibt sich daraus erst
über die **Kosten**: kostet ein verpasstes Standgeld das *r*-Fache einer unnötigen Maßnahme, lohnt sich die Maßnahme bei *p* > 1/(1+*r*), vorausgesetzt die Vorhersage stimmt als Häufigkeit. Die Demo zeigt,
was die feste Schwelle 0,5 dagegen kostet, woran man erkennt, ob die Vorhersage stimmt (die Kalibrierung im Mittel und ein kleiner ECE zeigen es nicht), und was **Rekalibrierung** behebt und was nicht.
Alle Kosten sind angenommen (kein Geldbetrag), die Messungen gelten für diesen Containerstrom, 500 Trainingscontainer und die genannten Modelle.

Weiter unten: die fünf Experimente, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert Kalibrierung und Schwelle", expanded=True):
    st.markdown(
        """
1. **Entscheiden heißt vergleichen:** Eine Maßnahme bei einem Container ohne Standgeld kostet 1, keine Maßnahme bei einem Container mit Standgeld kostet *r* (richtig entschieden kostet 0). Bei wahrer Wahrscheinlichkeit
   *p* kostet die Maßnahme im Mittel 1 − *p*, die Unterlassung *r*·*p*: die Maßnahme ist besser genau bei **p > 1/(1+r)**. Die Schwelle gehört zum Kostenverhältnis, nicht zur Zahl 0,5.
2. **Das gilt nur für die wahre Wahrscheinlichkeit.** Ein Modell liefert *p̂*. Stimmt *p̂* nicht als Häufigkeit, ist die Kostenschwelle auf *p̂* nicht mehr die beste Regel. Ob *p̂* stimmt, zeigt das
   Zuverlässigkeitsdiagramm (Vorhersage gegen beobachtete Häufigkeit) und die Steigung einer Anpassung y ~ a + b·logit *p̂*; ein Mittel von *p̂* gleich dem Anteil Standgeld zeigt es nicht.
3. **Rekalibrieren** heißt, auf einer eigenen **Rekalibriermenge** (Container mit bekanntem Ergebnis, nicht die Trainingscontainer) eine Abbildung von *p̂* auf die beobachtete Häufigkeit zu lernen: **Platt** (zwei
   Parameter, ein Logit auf logit *p̂*) oder **isotonisch** (nichtfallende Anpassung mit Pool Adjacent Violators, zwischen den Stützstellen linear interpoliert). Das behebt eine verzerrte Skala, nicht eine fehlende Rangfolge, und braucht Daten.

**Konventionen dieser Demo.** r = Kosten verpasst : Kosten unnötig (andere Quellen schreiben das Verhältnis umgekehrt; dann lautet die Schwelle ρ/(1+ρ) mit ρ = 1/r). Maßnahme bei *p̂* > Schwelle.
**Regret** = erwartete Kosten der Regel bei wahrer Wahrscheinlichkeit minus die der Orakelregel (wahres *p*, Schwelle 1/(1+r)), **je 1000 Entscheidungen**. **ECE** = Σ (Anteil im Fach) · |mittleres *p̂* − beobachtete
Häufigkeit| mit der gewählten Fächerzahl. **Rekalibriermenge** heißt hier nie „Kalibriermenge“ (dieses Wort meint in anderen Demos die Menge für Überdeckung bei Intervallen).
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Entscheidung**")
    r = st.select_slider("Kostenverhältnis r (verpasst : unnötig)", options=C.RATIOS, key="r_select",
                         help="Ein verpasstes Standgeld kostet das r-Fache einer unnötigen Maßnahme; die Kostenschwelle ist 1/(1+r). Angenommen, nicht gemessen; die vorgerechneten Messungen gelten für 1, 3, 9 und 19.")
    st.markdown("**Modell und Trainingsmenge**")
    model = st.selectbox("Modell", M.MODEL_KEYS, format_func=LABEL.get, key="model_select",
                         help="Fünf angepasste Modelle und drei kontrolliert verzerrte Fassungen der Haupteffekte (nur Skala verzerrt, Rangfolge gleich).")
    seed = st.number_input("Zufalls-Seed der Trainingsmenge", *bounds("seed_input"), key="seed_input", step=1,
                           help="Bestimmt die 500 Trainingscontainer, die Rekalibriermenge und damit die Live-Rechnung. Die vorgerechneten Messungen nutzen die Zufalls-Seeds 500 bis 539.")
    st.button("🎲 Neue Trainingsmenge ziehen", width="stretch", on_click=randomize_seed)
    st.markdown("**Rekalibrierung und Diagramm**")
    m = st.select_slider("Rekalibriermenge m (Container)", options=C.CAL_SIZES, key="m_select",
                         help="Container mit bekanntem Ergebnis, auf denen die Rekalibrierung gelernt wird. Je nach Standgeld-Anteil (rund 23 %) sind das etwa m/4 Container mit Standgeld.")
    bins = st.selectbox("Fächer des Zuverlässigkeitsdiagramms und des ECE", BINNINGS, key="bins_select",
                        help="Die Fächerzahl ändert die Werte des ECE, in den gemessenen Modellen aber nicht die Reihenfolge der acht Modelle.")

sync_query_params({"r_select": int(r), "model_select": model, "m_select": int(m), "seed_input": int(seed), "bins_select": bins})

settings = E.Settings(int(r), model, int(m), int(seed), bins)
with st.spinner("Trainingsmenge anpassen und Rekalibrierung lernen (Live-Rechnung, rund 20 000 Testcontainer) …"):
    live = _live(settings)
reg = E.live_regrets(live, settings.r)
good = E.live_goodness(live, bins)
label = LABEL[model]

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
st.caption(f"Live: {label}, eine Trainingsmenge (Zufalls-Seed {settings.seed}, {C.N_TRAIN} Container), {E.tsd(C.N_LIVE_TEST)} Testcontainer"
           f"; Regret je 1000 Entscheidungen bei Verhältnis r = {settings.r} (Orakel = 0). Eine Trainingsmenge ist ein Einzelfall: die Mittel über {REPS} Trainingsmengen stehen in den Kernfragen.")
m1, m2, m3, m4 = st.columns(4)
# Delta immer als „mein Wert minus Referenz“; die Farbe regelt delta_color="inverse" (Vorzeichen-Falle, DEMO-PLAYBOOK §2).
m1.metric("Regret roh (Schwelle 1/(1+r))", E.de(reg["roh"]), delta="Referenz", delta_color="off")
m2.metric("Regret feste Schwelle 0,5", E.de(reg["t05"]), delta=f"{reg['t05'] - reg['roh']:+.1f}".replace(".", ",") + " gegen roh", delta_color="inverse")
m3.metric("Regret nach Platt", E.de(reg["platt"]), delta=f"{reg['platt'] - reg['roh']:+.1f}".replace(".", ",") + " gegen roh", delta_color="inverse")
m4.metric("Steigung der Kalibrierung", E.de(good["slope"], 2) if good["slope"] == good["slope"] else "nicht bestimmt", delta="1 = kalibriert", delta_color="off")
grid = E.COST_GRID
st.plotly_chart(
    build_cost_curve(grid, [("wahres p (Orakel)", _population_curve(settings.r), C.COLOR_ORACLE, "solid"), (f"{label}", E.cost_curve(live.pf, live.p, settings.r), C.COLOR_RAW, "solid")], settings.r),
    width="stretch", key="main_chart")
st.caption("Erwartete Kosten je 1000 Entscheidungen, wenn bei p̂ > t eine Maßnahme ergriffen wird. Grün: die Orakelregel (wahres p, 100 000 Container) hat ihr Minimum bei 1/(1+r); die Regel des Modells (orange) nutzt p̂. "
           "Die rote Senkrechte ist die feste Schwelle 0,5.")

# --- Kernfragen -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen (Mittel über die vorgerechneten Trainingsmengen)")
cst = E.c_stats(FZ_DATA, model, settings.r)
pop = _matrix()
st.markdown("**1. Welche Schwelle gehört zu welchem Kostenverhältnis, und was kostet die 0,5?**")
_show(E.verdict_threshold(settings.r, label, cst, pop[(settings.r, "fest")], REPS))
st.markdown("**2. Stimmt p̂ als Häufigkeit, und woran sieht man es?**")
rk = E.measure_ranks(FZ_DATA, settings.r)
_show(E.verdict_calibration(label, E.good_stats(FZ_DATA, model), E.max_mean_gap(FZ_DATA, model), rk["ece"][model], rk["brier"][model], rk["regret"][model], settings.r, REPS))
st.markdown("**3. Was repariert Rekalibrierung, was nicht, und wie viele Ereignisse braucht sie?**")
_show(E.verdict_recalibration(settings.r, label, settings.m, E.e_stats(FZ_DATA, model, settings.m, settings.r), REPS))

st.markdown("---")

# --- Experiment 1: von Hand ------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Das Beispiel von Hand – 30 Container in drei Gruppen")
st.caption("Jede Gruppe hat eine Vorhersage p̂; beobachtet ist, wie viele ihrer Container tatsächlich Standgeld hatten. Kosten: unnötige Maßnahme 1, verpasstes Standgeld r (Regler links). "
           "Alle Zahlen sind exakte Brüche. Die Voreinstellung drückt die Vorhersage zur Mitte (Vorhersage 20, 30, 40 %, beobachtet 0, 20, 70 %).")
for i, (p0, n0, e0) in enumerate(HAND_DEFAULT):
    for suffix, v in (("p", p0), ("n", n0), ("e", e0)):
        st.session_state.setdefault(f"hand_{suffix}{i}", v)
cols = st.columns(3)
groups = []
for i, col in enumerate(cols):
    with col:
        st.markdown(f"**Gruppe {i + 1}**")
        p_pct = col.slider("Vorhersage p̂ (Prozent)", 1, 99, key=f"hand_p{i}")
        n_g = col.number_input("Container", 1, 200, key=f"hand_n{i}", step=1)
        if st.session_state[f"hand_e{i}"] > n_g:
            st.session_state[f"hand_e{i}"] = int(n_g)
        e_g = col.number_input("davon mit Standgeld", 0, int(n_g), key=f"hand_e{i}", step=1)
        groups.append(H.Group(Fraction(int(p_pct), 100), int(n_g), int(e_g)))
st.button("↩️ Beispiel von Hand wiederherstellen", on_click=_reset_hand)
groups = tuple(groups)
hs = H.summary(groups, settings.r)
n_all = H.total_containers(groups)
st.dataframe([
    {"Gruppe": i + 1, "Vorhersage p̂": E.frac(g.p_hat), "Container": g.n, "mit Standgeld": g.events, "beobachtet": E.frac(g.observed),
     "Maßnahme nach p̂ (Schwelle 1/(1+r))": "ja" if g.p_hat > hs["t"] else "nein", "Maßnahme nach beobachtet": "ja" if g.observed > hs["t"] else "nein"}
    for i, g in enumerate(groups)], hide_index=True, width="stretch")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Gesamtkosten roh", E.frac(hs["roh"]), delta=f"Schwelle {E.frac(hs['t'])} auf p̂", delta_color="off")
c2.metric("Feste Schwelle 1/2", E.frac(hs["fest"]), delta=E.frac(hs["fest"] - hs["roh"]) + " gegen roh", delta_color="inverse")
c3.metric("Nach beobachteter Häufigkeit", E.frac(hs["beobachtet"]), delta=E.frac(hs["beobachtet"] - hs["roh"]) + " gegen roh", delta_color="inverse")
c4.metric(f"Bestes aller {len(hs['alle'])} Gruppenregeln", E.frac(hs["bestes"]), delta="Aufzählung", delta_color="off")
st.plotly_chart(build_hand_curve(H.cost_curve(groups, settings.r), settings.r), width="stretch", key="hand_chart")
brier, rel, res, unc = H.brier_decomposition(groups)
mean_p, share = H.mean_prediction(groups), H.event_share(groups)
st.markdown(
    f"**Kalibrierung im Mittel:** mittleres p̂ = {E.frac(mean_p)}, Anteil Standgeld = {E.frac(share)} → {'im Mittel kalibriert' if mean_p == share else 'im Mittel nicht kalibriert'}. "
    f"**Brier** = {E.frac(brier)} = REL {E.frac(rel)} − RES {E.frac(res)} + UNC {E.frac(unc)} (Murphy-Zerlegung, exakt). "
    f"**ECE hängt am Fach:** 2 / 4 / 5 / 10 gleich breite Fächer: {' / '.join(E.frac(H.ece(groups, b)) for b in (2, 4, 5, 10))}; "
    f"bei 5 Fächern mit rechts geschlossenen Fächern {E.frac(H.ece(groups, 5, left_closed=False))} (liegt p̂ genau auf einer Fachgrenze, entscheidet die Randregel)."
)
_show(E.verdict_hand(hs, n_all))
with st.expander("Schwelle aus einer allgemeinen Kostenmatrix (a, b, c₀₀, c₁₁)"):
    st.markdown("Maßnahme bei Container ohne Standgeld kostet *a*, keine Maßnahme bei Standgeld *b*, keine Maßnahme ohne Standgeld *c₀₀*, Maßnahme bei Standgeld *c₁₁*. Die Maßnahme ist besser bei *p* > (a − c₀₀) / ((a − c₀₀) + (b − c₁₁)) (bei Gleichheit sind beide Entscheidungen gleich teuer): "
                "nur die Unterschiede je Entscheidung zählen, eine Konstante je wahrer Klasse ändert die Schwelle nicht, eine Konstante je Entscheidung verschiebt sie.")
    mc = st.columns(4)
    ma, mb, mc0, mc1 = (mc[0].number_input("a (Maßnahme ohne Standgeld)", 1, 100, 1, key="mat_a"), mc[1].number_input("b (keine Maßnahme bei Standgeld)", 1, 100, 3, key="mat_b"),
                        mc[2].number_input("c₀₀ (keine Maßnahme ohne Standgeld)", -50, 50, 0, key="mat_c00"), mc[3].number_input("c₁₁ (Maßnahme bei Standgeld)", -50, 50, 0, key="mat_c11"))
    try:
        st.info(f"Schwelle = **{E.frac(H.threshold_from_matrix(ma, mb, mc0, mc1))}**.")
    except ValueError:
        st.warning("Die Matrix braucht a − c₀₀ > 0 und b − c₁₁ > 0, sonst gibt es keine Schwelle (eine der beiden Entscheidungen wäre immer besser).")
    st.caption("Mit a = 1, b = r, c₀₀ = c₁₁ = 0 ist das die Schwelle 1/(1+r) der Demo. Elkan 2001 (Quellenbericht, Stufe W) sagt dasselbe: für die Entscheidung hat jede 2×2-Matrix einen Freiheitsgrad; die Umformung ist hier selbst nachgerechnet.")

st.markdown("---")

# --- Experiment 2: Kosten der Schwelle -------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Was kostet die feste 0,5, und was kostet ein falsch angenommenes r?")
st.caption("Bei wahrem p (100 000 Container, Zufalls-Seed 7) ohne Rauschen der Ergebnisse. Darunter dasselbe für das gewählte Modell über die vorgerechneten Trainingsmengen.")
r_assumed = st.select_slider("Angenommenes Verhältnis r′ (statt des wahren r)", options=C.RATIOS, value=3, key="rprime_select",
                             help="Die Regel „Maßnahme bei p > 1/(1+r′)“ wird mit dem wahren r bewertet. Experimentregler, ändert nur dieses Experiment.")
curves = [("wahres p (Orakel)", _population_curve(settings.r), C.COLOR_ORACLE, "solid")]
st.plotly_chart(build_cost_curve(grid, curves, settings.r, r_assumed), width="stretch", key="cost_chart")
st.caption("Erwartete Kosten je 1000 Entscheidungen der Regel „Maßnahme bei p > t“ bei wahrem p. Das Minimum liegt bei 1/(1+r) (grün); die rote Senkrechte ist 0,5, die gepunktete Senkrechte die angenommene Schwelle 1/(1+r′).")
st.markdown("**Regret je 1000 Entscheidungen bei wahrem p** (Zeile: wahres r; Spalte: angenommenes r′, Schwelle 1/(1+r′); Diagonale 0):")
st.dataframe([{"wahres r": rr, **{f"r′ = {r2}": E.de(pop[(rr, r2)]) for r2 in C.RATIOS}, "feste Schwelle 0,5": E.de(pop[(rr, "fest")])} for rr in C.RATIOS], hide_index=True, width="stretch")
st.markdown(f"**{label}, über {REPS} Trainingsmengen** (Regret je 1000, Mittel ± Streuung):")
st.dataframe([{"Regel": E.METHOD_LABELS[k], "Regret": f"{E.de(cst[k][0])} ± {E.de(cst[k][1])}"} for k in ("roh", "t05")] +
             [{"Regel": "0,5 teurer als roh in", "Regret": f"{cst['n_t05_gt_roh']} von {cst['reps']} Trainingsmengen"}], hide_index=True, width="stretch")
_show(E.verdict_wrong_ratio(settings.r, pop))
st.caption("Die Schwelle ist das τ-Quantil eines 0/1-Ergebnisses (τ = r/(1+r)): dieselbe Regel wie beim Newsvendor, aber für ein Ja/Nein statt für eine Menge ([newsvendor-direkt-demo](https://github.com/sebastian-hanisch/newsvendor-direkt-demo)).")

st.markdown("---")

# --- Experiment 3: Stimmt p̂? ------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Stimmt p̂ als Häufigkeit?")
st.caption(f"Live: {label}, Zufalls-Seed {settings.seed}, {E.tsd(C.N_LIVE_TEST)} Testcontainer; Fächer: {bins}. Die Punktgröße zeigt die Zahl der Container im Fach.")
st.plotly_chart(build_reliability([("rohe Vorhersage p̂", K.reliability_bins(live.pf, live.y, bins), C.COLOR_RAW)]), width="stretch", key="rel_chart")
g1, g2, g3, g4, g5 = st.columns(5)
g1.metric("AUC", E.de(good["auc"], 3))
g2.metric(f"ECE ({bins})", E.de(good["ece"], 4))
g3.metric("Steigung b", E.de(good["slope"], 2) if good["slope"] == good["slope"] else "nicht bestimmt")
g4.metric("Achsenabschnitt a", E.de(good["icpt"], 2) if good["icpt"] == good["icpt"] else "nicht bestimmt")
g5.metric("Brier", E.de(good["brier"], 4))
st.caption(f"Brier (auf zwei Stellen gerundete Vorhersagen) {E.de(good['brier_round'], 4)} = REL {E.de(good['rel'], 4)} − RES {E.de(good['res'], 4)} + UNC {E.de(good['unc'], 4)}. "
           f"Mittel der Vorhersage im Training minus Anteil Standgeld im Training: {E.sci(live.train_mean_gap)}.")
floor_key = {"5 gleich breit": "ece5", "10 gleich breit": "ece10", "20 gleich breit": "ece20"}.get(bins)
if floor_key:
    fl = E.ece_floor_row(FZ_DATA, 20_000, floor_key)
    st.info(f"**ECE-Boden:** bei {E.tsd(C.N_LIVE_TEST)} Containern (Fächer: {bins}) hat selbst die wahre Wahrscheinlichkeit (perfekt kalibriert, eigene Ergebnisse gezogen) einen ECE von {E.de(fl[0], 4)} ± {E.de(fl[1], 4)} "
            f"(100 Wiederholungen, vorgerechnet). Ein ECE in dieser Größe ist von „perfekt“ nicht zu trennen.")
else:
    st.info("Für Fächer gleicher Masse ist der ECE-Boden nicht gemessen.")
rows = E.rank_rows(FZ_DATA, settings.r, bins)
st.markdown(f"**Alle acht Modelle** (vorgerechnet, Mittel über {REPS} Trainingsmengen, 100 000 Testcontainer; Rang 1 = bester Wert):")
st.dataframe([{"Modell": LABEL[x["key"]], "AUC": E.de(x["auc"], 3), f"ECE ({bins})": E.de(x["ece"], 4), "Rang ECE": x["ece_rank"], "Brier": E.de(x["brier"], 4), "Rang Brier": x["brier_rank"],
               "Steigung": E.de(x["slope"][0], 2) + " ± " + E.de(x["slope"][1], 2), f"Regret roh (r = {settings.r})": E.de(x["regret"]), "Rang Regret": x["regret_rank"]} for x in rows],
              hide_index=True, width="stretch")
_show(E.verdict_ece_vs_regret(rows, settings.r, E.ece_floor_row(FZ_DATA, 2000), E.n_models_changing_rank(FZ_DATA), bins))

st.markdown("---")

# --- Experiment 4: Rekalibrieren -------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Rekalibrieren – und wie viele Ereignisse braucht es?")
n_ev, n_non = int(live.yc.sum()), int(len(live.yc) - live.yc.sum())
st.caption(f"Live: Rekalibriermenge m = {settings.m} Container mit {n_ev} Ereignissen (Standgeld) und {n_non} Nichtereignissen; Platt: a = "
           + ("nicht bestimmt" if live.platt[1] != live.platt[1] else f"{E.de(live.platt[0], 2)}, b = {E.de(live.platt[1], 2)}") + ".")
if n_ev < 200 or n_non < 200:
    st.warning(f"Weniger als 200 Ereignisse oder 200 Nichtereignisse in der Rekalibriermenge ({n_ev} / {n_non}): nach Van Calster u. a. 2019 (Orientierung, keine Schranke) ist die Kalibrierkurve dann unsicher.")
est = E.e_stats(FZ_DATA, model, settings.m, settings.r)
st.dataframe([
    {"Regel": E.METHOD_LABELS[k], "Regret live (diese Trainingsmenge)": E.de(reg[k]), f"Regret vorgerechnet (Mittel ± Streuung, {REPS} Trainingsmengen)": f"{E.de(est[k][0])} ± {E.de(est[k][1])}"}
    for k in ("roh", "platt", "iso", "tuned")], hide_index=True, width="stretch")
st.caption(f"Die abgestimmte Schwelle (live {E.de(reg['tuned_t'], 2)}) wird direkt auf den Kosten der Rekalibriermenge gewählt (Raster 0,02 bis 0,98 in Schritten von 0,02). Live und vorgerechnet können abweichen: eine Trainingsmenge ist ein Einzelfall.")
st.plotly_chart(build_reliability([("roh", K.reliability_bins(live.pf, live.y, bins), C.COLOR_RAW), ("nach Platt", K.reliability_bins(live.pp, live.y, bins), C.COLOR_PLATT),
                                    ("isotonisch", K.reliability_bins(live.pi, live.y, bins), C.COLOR_ISO)]), width="stretch", key="recal_chart")
size_rows = []
for mm in C.CAL_SIZES:
    s_ = E.e_stats(FZ_DATA, model, mm, settings.r)
    size_rows.append({"m": mm, "events": s_["events"], "roh": s_["roh"][0], "platt": s_["platt"][0], "iso": s_["iso"][0], "tuned": s_["tuned"][0], "n_platt_lt_roh": s_["n_platt_lt_roh"],
                      "sd": {k: s_[k][1] for k in ("roh", "platt", "iso", "tuned")}})
st.plotly_chart(build_size_curves(
    [x["m"] for x in size_rows], [("roh", [x["roh"] for x in size_rows], [x["sd"]["roh"] for x in size_rows], C.COLOR_RAW), ("Platt", [x["platt"] for x in size_rows], [x["sd"]["platt"] for x in size_rows], C.COLOR_PLATT),
                                  ("isotonisch", [x["iso"] for x in size_rows], [x["sd"]["iso"] for x in size_rows], C.COLOR_ISO),
                                  ("abgestimmte Schwelle", [x["tuned"] for x in size_rows], [x["sd"]["tuned"] for x in size_rows], C.COLOR_TUNED)], [x["events"] for x in size_rows]),
    width="stretch", key="size_chart")
st.caption(f"Vorgerechnet: {label}, r = {settings.r}, Mittel ± Streuung über {REPS} Trainingsmengen; je Trainingsmenge eine neue Rekalibriermenge der Größe m.")
_show(E.verdict_sample_size(settings.r, label, size_rows, REPS))

st.markdown("---")

# --- Experiment 5: Wo Rekalibrierung nichts rettet ---------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Wo Rekalibrierung nichts mehr rettet")
mt = E.model_table(FZ_DATA, settings.r, bins)
st.dataframe([{"Modell": LABEL[x["key"]], "AUC": E.de(x["auc"], 3), **{E.METHOD_LABELS[k]: f"{E.de(x[k][0])} ± {E.de(x[k][1])}" for k in ("roh", "t05", "platt", "iso", "tuned")},
               "Platt billiger als roh": f"{x['n_platt_lt_roh']} von {x['reps']}", "isotonisch billiger als roh": f"{x['n_iso_lt_roh']} von {x['reps']}"} for x in mt], hide_index=True, width="stretch")
st.caption(f"Regret je 1000 Entscheidungen bei r = {settings.r}, Mittel ± Streuung über {REPS} Trainingsmengen (vorgerechnet; Rekalibriermenge {C.N_CAL} Container, 100 000 Testcontainer).")
st.plotly_chart(build_methods_bar([LABEL[x["key"]] for x in mt], [("roh", [x["roh"][0] for x in mt], [x["roh"][1] for x in mt], C.COLOR_RAW), ("Platt", [x["platt"][0] for x in mt], [x["platt"][1] for x in mt], C.COLOR_PLATT),
                                                                  ("isotonisch", [x["iso"][0] for x in mt], [x["iso"][1] for x in mt], C.COLOR_ISO)]), width="stretch", key="models_chart")
_show(E.verdict_no_rescue(settings.r, mt, REPS))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
hs_e = E.e_stats(FZ_DATA, "haupt", 100, 9)
st.markdown(
    f"""
| Annahme | Was passiert, wenn sie verletzt ist | Gemessen |
|---|---|---|
| **Das Kostenverhältnis r ist bekannt und gilt für jeden Container gleich** | Ein falsch angenommenes r kostet Regret; die feste 0,5 ist eine Annahme r = 1. | Bei wahrem r = 9 kostet r′ = 3 {E.de(pop[(9, 3)])} je 1000, r′ = 19 {E.de(pop[(9, 19)])}, die feste 0,5 {E.de(pop[(9, 'fest')])} (bei wahrem p, 100 000 Container). |
| **p̂ stimmt als Häufigkeit** | Die Kostenschwelle auf p̂ ist nicht mehr die beste Regel; das Mittel im Training sagt nichts über die Steigung. | Haupteffekte: Steigung {E.de(E.good_stats(FZ_DATA, 'haupt')['slope'][0], 2)} bei einem größten Abstand {E.sci(E.max_mean_gap(FZ_DATA, 'haupt'))} (Mittel von p̂ gegen Anteil Standgeld im Training), {REPS} Trainingsmengen. |
| **Die Rekalibriermenge ist groß genug** | Mit wenig Ereignissen verschlechtert Rekalibrieren die Entscheidung. | Haupteffekte, r = 9, m = 100 (im Mittel {E.de(hs_e['events'], 0)} Ereignisse): roh {E.de(hs_e['roh'][0])} → Platt {E.de(hs_e['platt'][0])}, isotonisch {E.de(hs_e['iso'][0])}. |
| **Die Rangfolge der Container stimmt** | Rekalibrieren kann eine fehlende Rangfolge nicht herstellen. | Siehe Experiment 5 („ohne Containerart“). |
| **Ein Containerstrom, eine Frist (7 Tage), n = 500** | Nicht gemessen sind andere Fristen, andere Trainingsgrößen, Kosten je Container, Teilgruppen und mehr als zwei Ausgänge. | Keine Messung. |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Gegeben:** Container mit Merkmalen $x$, Ergebnis $y \in \{0, 1\}$ (1 = Standgeld fällig), eine Vorhersage $\hat p(x) \in (0, 1)$ und ein Kostenverhältnis $r > 0$. **Gesucht:** eine Regel „Maßnahme bei $\hat p > t$“ mit kleinsten erwarteten Kosten.

| Symbol | Bedeutung |
|---|---|
| $p$ | wahre Wahrscheinlichkeit $P(y = 1 \mid x)$ |
| $\hat p$ | Vorhersage des Modells |
| $r$ | Kosten eines verpassten Standgelds geteilt durch Kosten einer unnötigen Maßnahme |
| $t$ | Schwelle: Maßnahme bei $\hat p > t$ |
| $a, b, c_{00}, c_{11}$ | Kostenmatrix: Maßnahme ohne Ereignis, keine Maßnahme bei Ereignis, keine Maßnahme ohne Ereignis, Maßnahme bei Ereignis |
| $\sigma$ | logistische Funktion $\sigma(z) = 1/(1 + e^{-z})$ |
| $n_k, \bar f_k, \bar y_k$ | Zahl der Container, mittlere Vorhersage und beobachtete Häufigkeit im Fach $k$ |

**Erwartete Kosten je Container** bei wahrem $p$:

$$\mathbb{E}[\text{Kosten}] = \begin{cases} 1 - p & \text{Maßnahme} \\ r\,p & \text{keine Maßnahme} \end{cases}$$

Die Maßnahme ist besser genau bei

$$p > \frac{1}{1 + r}.$$

Für eine allgemeine 2×2-Kostenmatrix (Elkan 2001) gilt

$$t^{*} = \frac{a - c_{00}}{(a - c_{00}) + (b - c_{11})}.$$

**Regret** der Regel „Maßnahme bei $\hat p > t$“ gegen die Orakelregel, je 1000 Entscheidungen, über $n$ Testcontainer:

$$\text{Regret}(t) = \frac{1000}{n} \sum_{i=1}^{n} \bigl|(1 + r)\,p_i - 1\bigr| \cdot \mathbf{1}\bigl[(\hat p_i > t) \neq (p_i > \tfrac{1}{1+r})\bigr].$$

**Kalibrierung.** Die Zwei-Parameter-Anpassung (Platt-Skalierung ohne geglättete Zielwerte) schätzt $a, b$ aus

$$P(y = 1 \mid \hat p) = \sigma\bigl(a + b\,\operatorname{logit}\hat p\bigr),\qquad \operatorname{logit} u = \ln\frac{u}{1 - u},$$

durch Newton-Verfahren mit Schritthalbierung (derselbe Code wie beim Logit der Modelle). Kalibriert heißt $a = 0$ und $b = 1$; $b < 1$ bedeutet zu extreme Vorhersagen (Van Calster u. a. 2019). Die **isotonische**
Rekalibrierung löst $\min \sum_i (y_i - g(\hat p_i))^2$ über nichtfallende $g$ mit Pool Adjacent Violators; zwischen den Stützstellen wird linear interpoliert, außerhalb gekappt.

**ECE** mit $B$ gleich breiten Fächern, Fach $k = \lfloor B\,\hat p \rfloor$ (links geschlossen):

$$\text{ECE} = \sum_{k} \frac{n_k}{n}\,\bigl|\bar f_k - \bar y_k\bigr|.$$

**Brier-Zerlegung** (Murphy) auf zwei Stellen gerundeten Vorhersagen:

$$\text{Brier} = \frac{1}{n}\sum_i (\hat p_i - y_i)^2 = \text{REL} - \text{RES} + \text{UNC},$$

mit $\text{REL} = \tfrac1n \sum_k n_k(\bar f_k - \bar y_k)^2$, $\text{RES} = \tfrac1n \sum_k n_k(\bar y_k - \bar y)^2$ und $\text{UNC} = \bar y(1 - \bar y)$ (hier sind die Klassen die verschiedenen gerundeten Vorhersagen).

**Logit der Modelle** (Newton, Achsenabschnitt unbestraft, $\lambda = 1/C$):

$$\min_{\beta}\ \sum_i \bigl[\ln(1 + e^{x_i^\top \beta}) - y_i\,x_i^\top \beta\bigr] + \frac{\lambda}{2}\,\lVert \beta_{1:} \rVert^2.$$

Die Modelle „verdichtet“, „überkonfident“ und „verschoben“ ersetzen $\hat p$ durch $\sigma(a + b\,\operatorname{logit}\hat p)$ mit $(a; b) = (0; 0{,}5)$, $(0; 2)$ und $(1; 1)$: die Rangfolge der Container bleibt, die Skala ist verzerrt.

**Literatur** (nach dem Quellenbericht, Stufen: W im Wortlaut gelesen, S nur Suchtreffer): Elkan, The Foundations of Cost-Sensitive Learning, IJCAI 2001 (W; Gleichung (2) nicht neu geöffnet, Umformung selbst nachgerechnet).
Van Calster u. a., Calibration: the Achilles heel of predictive analytics, BMC Medicine 17:230 (2019) (W). Dimitriadis, Gneiting, Jordan, Evaluating probabilistic classifiers: Reliability diagrams and score decompositions
revisited, arXiv 2008.03033 (W; das dort vorgeschlagene Diagramm über PAV statt Fächern ist hier nicht umgesetzt). Niculescu-Mizil und Caruana, Predicting Good Probabilities with Supervised Learning, ICML 2005 (nur S: isotonische
Rekalibrierung sei bei knappen Daten überanpassungsanfälliger als Platt; die Richtung passt zu Experiment 4, nachgelesen ist es nicht). Guo u. a. 2017 betrifft tiefe Netze und gilt nach dem Quellenbericht nicht für ein Logit mit Achsenabschnitt.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
