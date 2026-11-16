"""Auswertung: Live-Rechnung für eine Trainingsmenge, Kosten bei wahrem p, Kennzahlen aus den vorgerechneten Reihen und alle Verdikte als reine Funktionen.

Jedes Verdikt ist `verdict_*(…) -> (Art, Text)`; die App ruft `getattr(st, Art)(Text)`. Die Zahlen im Text kommen aus den Argumenten, nie aus dem Gedächtnis
(`tests/test_verdicts.py` prüft Schwellen und jede Zahl an handgesetzten Werten)."""

from dataclasses import dataclass

import numpy as np

import ksw_constants as C
import ksw_core as K
import ksw_frozen as FZ
import ksw_generator as G
import ksw_hand as H
import ksw_models as M

METHOD_LABELS = {"roh": "roh (Schwelle 1/(1+r))", "t05": "feste Schwelle 0,5", "platt": "nach Platt", "iso": "isotonisch", "tuned": "abgestimmte Schwelle"}
ECE_KEYS = {"5 gleich breit": "ece5", "10 gleich breit": "ece10", "20 gleich breit": "ece20", "10 gleiche Masse": "ece10m"}


def de(x: float, nd: int = 1) -> str:
    """Zahl mit deutschem Dezimalkomma."""
    return f"{x:.{nd}f}".replace(".", ",")


def sci(x: float) -> str:
    """Wissenschaftliche Schreibweise mit Dezimalkomma (1,2e-13)."""
    return f"{x:.1e}".replace(".", ",")


def tsd(n: int) -> str:
    """Ganze Zahl mit Punkt als Tausendertrenner."""
    return f"{int(n):,}".replace(",", ".")


def frac(f) -> str:
    """Bruch als Text (ganze Zahlen ohne Nenner)."""
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


# --- Live-Rechnung: eine Trainingsmenge ------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    r: int = C.DEFAULT_RATIO
    model: str = C.DEFAULT_MODEL
    m: int = C.DEFAULT_CAL
    seed: int = C.DEFAULT_SEED
    bins: str = C.DEFAULT_BINNING


@dataclass(frozen=True)
class Live:
    pf: np.ndarray       # rohe Vorhersage auf den Testcontainern
    pp: np.ndarray       # nach Platt
    pi: np.ndarray       # isotonisch
    pc: np.ndarray       # Vorhersage der Rekalibriermenge
    yc: np.ndarray       # Ergebnisse der Rekalibriermenge
    p: np.ndarray        # wahre Wahrscheinlichkeit der Testcontainer
    y: np.ndarray        # Standgeld der Testcontainer
    platt: tuple         # (a, b) der Rekalibrierung
    train_mean_gap: float  # mittleres p̂ minus Anteil Standgeld im Training


def analyse_instance(s: Settings) -> Live:
    """Eine Trainingsmenge (500 Container, Zufalls-Seed s.seed), eine Rekalibriermenge (s.m Container) und 20 000 Testcontainer."""
    train = G.make_dataset(C.N_TRAIN, s.seed)
    cal = G.make_dataset(s.m, C.LIVE_CAL_SEED + s.seed)
    test = G.make_dataset(C.N_LIVE_TEST, C.LIVE_TEST_SEED)
    pred = M.fit_model(s.model, train.X, train.d, train.y)
    pf, pc = pred(test.X), pred(cal.X)
    a, b = K.fit_platt(pc, cal.y)
    pp = pf if np.isnan(b) else K.platt_apply(pf, a, b)
    pi = K.isotonic_apply(K.isotonic_fit(pc, cal.y), pf)
    return Live(pf, pp, pi, pc, cal.y, test.p, test.y, (a, b), float(np.mean(pred(train.X)) - train.y.mean()))


def live_regrets(live: Live, r: int) -> dict:
    """Regret je 1000 Entscheidungen der fünf Regeln auf den Testcontainern der Live-Rechnung (bei wahrem p)."""
    tb = K.tuned_threshold(live.pc, live.yc, r, np.linspace(0.02, 0.98, 49))
    return {"roh": K.regret_per_1000(live.pf, live.p, r), "t05": K.regret_per_1000(live.pf, live.p, r, t=0.5),
            "platt": K.regret_per_1000(live.pp, live.p, r), "iso": K.regret_per_1000(live.pi, live.p, r), "tuned": K.regret_per_1000(live.pf, live.p, r, t=tb),
            "tuned_t": tb}


def live_goodness(live: Live, bins: str) -> dict:
    a, b = K.fit_platt(live.pf, live.y)
    bz = K.brier_decomposition(live.pf, live.y)
    return {"auc": K.auc(live.pf, live.y), "ece": K.ece(live.pf, live.y, bins), "icpt": a, "slope": b, "brier": K.brier(live.pf, live.y),
            "brier_round": bz[0], "rel": bz[1], "res": bz[2], "unc": bz[3], "n": len(live.y)}


COST_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


def cost_curve(score, p, r: float, grid=COST_GRID):
    """Erwartete Kosten je 1000 Entscheidungen der Regel „Maßnahme bei score > t“ über der Schwelle t."""
    return [1000.0 * K.expected_cost(score > t, p, r) for t in grid]


# --- Kosten bei wahrem p (Experiment 2) -------------------------------------------------------------------------------------------------------

def population():
    """Wahre Wahrscheinlichkeiten von 100 000 Containern (Zufalls-Seed 7)."""
    return G.population(C.POPULATION_N, C.POPULATION_SEED)[0]


def regret_matrix(p) -> dict:
    """Regret je 1000 bei wahrem p: {(r, r′): Regret der Schwelle 1/(1+r′) bei wahrem Verhältnis r} und {(r, 'fest'): Regret der festen 0,5}."""
    out = {}
    for r in C.RATIOS:
        for r2 in C.RATIOS:
            out[(r, r2)] = K.regret_per_1000(p, p, r, t=K.threshold(r2))
        out[(r, "fest")] = K.regret_per_1000(p, p, r, t=0.5)
    return out


# --- Kennzahlen aus den vorgerechneten Reihen ------------------------------------------------------------------------------------------------

def mean_sd(a) -> tuple:
    a = np.asarray(a, float)
    return float(np.mean(a)), float(np.std(a))


def reps(fz) -> int:
    return int(fz["meta"]["reps"])


def c_stats(fz, model: str, r: int) -> dict:
    """Mittel und Streuung des Regrets je Regel über die Trainingsmengen (Messung C) und Zählungen („in k von 40 Trainingsmengen“)."""
    R = {meth: FZ.series_c(fz, model, meth, r) for meth in ("roh", "t05", "platt", "iso", "tuned")}
    out = {meth: mean_sd(v) for meth, v in R.items()}
    out["n_t05_gt_roh"] = int(np.sum(R["t05"] > R["roh"]))
    out["n_platt_lt_roh"] = int(np.sum(R["platt"] < R["roh"]))
    out["n_iso_lt_roh"] = int(np.sum(R["iso"] < R["roh"]))
    out["n_platt_lt_iso"] = int(np.sum(R["platt"] < R["iso"]))
    out["n_tuned_lt_platt"] = int(np.sum(R["tuned"] < R["platt"]))
    out["reps"] = len(R["roh"])
    return out


def e_stats(fz, model: str, m: int, r: int) -> dict:
    """Wie c_stats für die Messung E (Rekalibriermenge der Größe m, andere Zufalls-Seeds als in C)."""
    R = {meth: FZ.series_e(fz, model, m, meth, r) for meth in ("roh", "t05", "platt", "iso", "tuned")}
    out = {meth: mean_sd(v) for meth, v in R.items()}
    out["n_platt_lt_roh"] = int(np.sum(R["platt"] < R["roh"]))
    out["n_iso_lt_roh"] = int(np.sum(R["iso"] < R["roh"]))
    out["events"] = float(np.mean(fz["E"][model][str(m)]["ereignisse"]))
    out["reps"] = len(R["roh"])
    return out


def good_stats(fz, model: str) -> dict:
    return {g: mean_sd(v) for g, v in fz["C"][model]["gute"].items()}


def max_mean_gap(fz, model: str) -> float:
    """Größter Betrag von „mittleres p̂ minus Anteil Standgeld“ im Training über die Trainingsmengen."""
    return float(np.max(np.abs(fz["C"][model]["gute"]["mitt"])))


def ranks(values: dict) -> dict:
    """Rang 1 = kleinster Wert; Gleichstände bekommen den kleineren Rang."""
    order = sorted(values.values())
    return {k: order.index(v) + 1 for k, v in values.items()}


def measure_ranks(fz, r: int, bins: str = C.DEFAULT_BINNING) -> dict:
    """Rang der acht Modelle nach ECE (gewähltes Fächerschema), Brier, RMSE und Regret roh bei Verhältnis r (1 = bester)."""
    mean = lambda k, g: float(np.mean(fz["C"][k]["gute"][g]))
    keys = M.MODEL_KEYS
    return {"ece": ranks({k: mean(k, ECE_KEYS[bins]) for k in keys}), "brier": ranks({k: mean(k, "brier") for k in keys}),
            "rmse": ranks({k: mean(k, "rmse") for k in keys}),
            "regret": ranks({k: float(np.mean(FZ.series_c(fz, k, "roh", r))) for k in keys})}


def n_models_changing_rank(fz) -> int:
    """Zahl der Modelle, deren Platz sich zwischen den sechs Maßen (ECE mit 5, 10, 20 gleich breiten Fächern, 10 gleiche Masse, Brier, RMSE) ändert."""
    keys = M.MODEL_KEYS
    all_ranks = [ranks({k: float(np.mean(fz["C"][k]["gute"][g])) for k in keys}) for g in ("ece5", "ece10", "ece20", "ece10m", "brier", "rmse")]
    return sum(len({rk[k] for rk in all_ranks}) > 1 for k in keys)


def ece_floor_row(fz, n: int, bins_key: str = "ece10") -> tuple:
    for row in fz["D"]:
        if row["n"] == n:
            return mean_sd(row[bins_key])
    raise KeyError(n)


def model_table(fz, r: int, bins: str = C.DEFAULT_BINNING) -> list[dict]:
    """Eine Zeile je Modell für die Tabelle der Experimente 3 und 5 (Zahlen als Rohwerte)."""
    rows = []
    for k in M.MODEL_KEYS:
        g = good_stats(fz, k)
        c = c_stats(fz, k, r)
        rows.append({"key": k, "auc": g["auc"][0], "ece": g[ECE_KEYS[bins]][0], "brier": g["brier"][0], "slope": g["slope"], **{m: c[m] for m in ("roh", "t05", "platt", "iso", "tuned")},
                     "n_platt_lt_roh": c["n_platt_lt_roh"], "n_iso_lt_roh": c["n_iso_lt_roh"], "n_tuned_lt_platt": c["n_tuned_lt_platt"], "reps": c["reps"]})
    return rows


# --- Verdikte -------------------------------------------------------------------------------------------------------------------------------

def verdict_hand(s: dict, n_containers: int) -> tuple[str, str]:
    """Experiment 1 (Beispiel von Hand): `s` ist `ksw_hand.summary(groups, r)`."""
    r, t = s["r"], s["t"]
    text = (f"r = {r} (Schwelle {frac(t)}), {n_containers} Container: roh (nach der Vorhersage) {frac(s['roh'])}, feste Schwelle 1/2: {frac(s['fest'])}, "
            f"nach den beobachteten Häufigkeiten {frac(s['beobachtet'])}; das beste aller {len(s['alle'])} Gruppenregeln kostet {frac(s['bestes'])}.")
    if s["roh"] == s["beobachtet"] == s["fest"]:
        return "info", text + " Bei diesen Gruppen entscheiden alle drei Regeln gleich: kein Unterschied."
    if s["beobachtet"] == s["bestes"] and s["roh"] > s["beobachtet"]:
        return "success", text + f" Die rohe Vorhersage kostet {frac(s['roh'] - s['beobachtet'])} mehr als die Entscheidung nach beobachteter Häufigkeit."
    if s["beobachtet"] == s["bestes"]:
        return "info", text + " Die Regel nach beobachteter Häufigkeit ist die beste."
    return "warning", text + " Die Regel nach beobachteter Häufigkeit ist hier nicht die beste aller Gruppenregeln (Gruppen mit wenigen Containern)."


def verdict_threshold(r: int, label: str, st: dict, pop_regret_05: float, reps_: int) -> tuple[str, str]:
    """Kernfrage 1: Was kostet die feste Schwelle 0,5 gegenüber 1/(1+r)? `st` ist `c_stats`; `pop_regret_05` der Regret von 0,5 bei wahrem p (Population)."""
    if r == 1:
        return "info", (f"Bei r = 1 ist die Kostenschwelle 1/(1+r) gleich 0,5: feste Schwelle und Kostenschwelle sind dieselbe Regel, es gibt keinen Unterschied "
                        f"({label}: Regret {de(st['roh'][0])} ± {de(st['roh'][1])} je 1000 Entscheidungen, {reps_} Trainingsmengen).")
    text = (f"{label}, r = {r}, {reps_} Trainingsmengen zu je {C.N_TRAIN} Containern: die feste Schwelle 0,5 kostet im Mittel {de(st['t05'][0])} ± {de(st['t05'][1])} Einheiten je 1000 "
            f"Entscheidungen mehr als das Orakel, die Schwelle 1/(1+r) = {de(1 / (1 + r), 3)} nur {de(st['roh'][0])} ± {de(st['roh'][1])}; 0,5 ist teurer in {st['n_t05_gt_roh']} von {reps_}. "
            f"Bei bekanntem wahrem p kostet 0,5 {de(pop_regret_05)} (Orakel 0).")
    if st["n_t05_gt_roh"] < reps_:
        text += f" Ausnahme: in {reps_ - st['n_t05_gt_roh']} von {reps_} Trainingsmengen war 0,5 nicht teurer."
    if st["t05"][0] < st["roh"][0]:
        return "info", text + " Bei diesem verzerrten Modell ist 0,5 im Mittel die billigere Regel: das liegt an der Verzerrung, nicht an der Schwelle."
    return "warning", text


def verdict_calibration(label: str, g: dict, gap: float, rank_ece: int, rank_brier: int, rank_regret: int, r: int, n_reps: int) -> tuple[str, str]:
    """Kernfrage 2: Stimmt p̂ als Häufigkeit? `g` ist `good_stats` des Modells (Mittel, Streuung)."""
    slope, sd = g["slope"]
    text = (f"{label}: Steigung der Kalibrierung auf den Testdaten {de(slope, 2)} ± {de(sd, 2)} (1 hieße kalibriert, {n_reps} Trainingsmengen); größter Betrag von mittlerem p̂ minus "
            f"Anteil Standgeld im Training {sci(gap)}. ECE (10 gleich breite Fächer) {de(g['ece10'][0], 3)}, Rang {rank_ece} von {len(M.MODEL_KEYS)} (1 = kleinster); "
            f"Rang nach Brier {rank_brier}; Rang nach Regret bei r = {r} {rank_regret}.")
    if abs(slope - 1.0) <= 0.1 + 1e-9:
        return "success", text + " Die Steigung liegt nahe bei 1; der Achsenabschnitt geht in dieses Urteil nicht ein, p̂ stimmt als Häufigkeit nur, wenn er auch nahe 0 liegt."
    direction = "zu extreme" if slope < 1 else "zu flache, zur Mitte gedrückte"
    return "warning", text + f" Die Steigung weicht von 1 ab ({direction} Vorhersagen), obwohl das Mittel im Training stimmen kann."


def verdict_recalibration(r: int, label: str, m: int, st: dict, reps_: int) -> tuple[str, str]:
    """Kernfrage 3: Was bringt Rekalibrierung? `st` ist `e_stats` (Messung E) für Modell und Größe m."""
    base = st["roh"][0]
    text = (f"{label}, r = {r}, Rekalibriermenge m = {m} (im Mittel {de(st['events'], 0)} Container mit Standgeld), {reps_} Trainingsmengen: Regret je 1000 roh {de(base)} → "
            f"Platt {de(st['platt'][0])} (billiger als roh in {st['n_platt_lt_roh']} von {reps_}), isotonisch {de(st['iso'][0])} (billiger in {st['n_iso_lt_roh']} von {reps_}), "
            f"abgestimmte Schwelle {de(st['tuned'][0])}.")
    if st["platt"][0] >= base and st["iso"][0] >= base:
        return "warning", text + " Hier schadet die Rekalibrierung im Mittel: zu wenig Daten oder keine Verzerrung, die sich beheben ließe."
    if st["platt"][0] >= base:
        return "info", text + " Platt ist hier im Mittel nicht billiger als roh."
    if st["iso"][0] >= base:
        return "info", text + " Die isotonische Rekalibrierung ist hier im Mittel nicht billiger als roh."
    return "success", text


def verdict_wrong_ratio(r: int, regrets: dict) -> tuple[str, str]:
    """Experiment 2: Kosten bei wahrem p. `regrets` ist `regret_matrix`; nennt die Schwelle für jedes angenommene r′ (r′ = 1 ist die feste Schwelle 0,5) und die feste 0,5."""
    fest = regrets[(r, "fest")]
    if r == 1:
        return "info", (f"Wahres r = 1 (bei wahrem p, 100 000 Container): die Kostenschwelle ist 0,5, die feste Schwelle kostet {de(fest)} mehr (Regret 0). "
                        f"Ein zu hoch angenommenes r′ kostet je 1000 Entscheidungen: " + ", ".join(f"r′ = {r2}: {de(regrets[(r, r2)])}" for r2 in C.RATIOS if r2 != 1) + ".")
    others = [r2 for r2 in C.RATIOS if r2 not in (r, 1)]
    cells = ", ".join(f"r′ = {r2}: {de(regrets[(r, r2)])}" for r2 in others)
    text = (f"Wahres r = {r} (bei wahrem p, 100 000 Container): die Schwelle für ein falsch angenommenes Verhältnis kostet je 1000 Entscheidungen mehr als die richtige: {cells}; "
            f"die feste Schwelle 0,5 (das ist r′ = 1): {de(fest)}.")
    if all(regrets[(r, r2)] < fest for r2 in others):
        return "info", text + " Jedes andere falsch angenommene r′ kostet hier weniger als die feste 0,5."
    return "info", text + " Ein falsch angenommenes r′ kann hier mehr kosten als die feste 0,5."


def verdict_ece_vs_regret(rows: list[dict], r: int, floor2000: tuple, n_changed: int, bins: str = "10 gleich breit") -> tuple[str, str]:
    """Experiment 3: ECE-Rang und Regret-Rang der acht Modelle. `rows`: Wörterbücher mit key, ece_rank, regret, regret_rank, ece."""
    worst = max(rows, key=lambda x: x["regret"])
    ece_part = ", ".join(f"{M.MODEL_LABELS[x['key']]} {de(x['ece'], 4)} (Rang {x['ece_rank']})" for x in sorted(rows, key=lambda x: x["ece_rank"])[:3])
    text = (f"ECE ({bins}), die drei kleinsten von {len(rows)} Modellen: {ece_part}. Der höchste Regret bei r = {r} (roh) gehört zu „{M.MODEL_LABELS[worst['key']]}“: "
            f"{de(worst['regret'])} je 1000, ECE-Rang {worst['ece_rank']}. Der ECE perfekt kalibrierter wahrer Wahrscheinlichkeiten ist bei 2000 Containern (10 gleich breite Fächer) {de(floor2000[0], 4)} ± "
            f"{de(floor2000[1], 4)} (Boden des Maßes). Der Platz eines Modells ändert sich zwischen den sechs Maßen bei {n_changed} von {len(rows)} Modellen.")
    if worst["ece_rank"] <= len(rows) // 2:
        return "warning", text + " Ein kleiner ECE sagt hier nicht, welches Modell besser entscheidet."
    return "info", text + " Hier hat das Modell mit dem höchsten Regret auch einen großen ECE."


def verdict_sample_size(r: int, label: str, rows: list[dict], reps_: int) -> tuple[str, str]:
    """Experiment 4: Rekalibriermenge. `rows`: je Größe m ein Wörterbuch (m, events, roh, platt, iso, n_platt_lt_roh); Platt gegen roh nach Zählung."""
    first = next((x["m"] for x in rows if x["n_platt_lt_roh"] > reps_ / 2), None)
    cells = "; ".join(f"m = {x['m']}: Platt {de(x['platt'])}, isotonisch {de(x['iso'])} (Platt billiger als roh in {x['n_platt_lt_roh']} von {reps_})" for x in rows)
    head = f"{label}, r = {r}, Regret je 1000 roh {de(rows[0]['roh'])}. {cells}."
    if first is None:
        return "warning", head + f" Bei keiner gemessenen Größe ist Platt in mehr als der Hälfte der {reps_} Trainingsmengen billiger als roh."
    row = next(x for x in rows if x["m"] == first)
    return "info", head + f" Platt ist ab m = {first} (im Mittel {de(row['events'], 0)} Container mit Standgeld) in mehr als der Hälfte der Trainingsmengen billiger als roh."


def verdict_no_rescue(r: int, rows: list[dict], reps_: int) -> tuple[str, str]:
    """Experiment 5: wo Rekalibrierung nichts mehr rettet. `rows` wie `model_table` (Mittel, Streuung als Paare)."""
    by = {x["key"]: x for x in rows}
    no_help = [M.MODEL_LABELS[k] for k, x in by.items() if x["platt"][0] >= x["roh"][0]]
    oa = by["ohne_art"]
    cut = 100.0 * (oa["roh"][0] - oa["platt"][0]) / oa["roh"][0] if oa["roh"][0] > 0 else 0.0
    ja = by["je_art"]
    hp = by["haupt"]
    text = (f"r = {r}: „ohne Containerart“ (AUC {de(oa['auc'], 3)}): roh {de(oa['roh'][0])} → Platt {de(oa['platt'][0])} ({de(cut, 0)} % weniger), isotonisch {de(oa['iso'][0])}; "
            f"„je Art“: Platt billiger als roh in {ja['n_platt_lt_roh']} von {reps_} Trainingsmengen ({de(ja['platt'][0])} gegen {de(ja['roh'][0])}), isotonisch in {ja['n_iso_lt_roh']} von {reps_}. "
            f"Modelle, bei denen Platt im Mittel nicht billiger ist als roh: {', '.join(no_help) if no_help else 'keines'}. "
            f"Die abgestimmte Schwelle ist bei den Haupteffekten billiger als Platt in {hp['n_tuned_lt_platt']} von {reps_} Trainingsmengen.")
    return "info", text


def rank_rows(fz, r: int, bins: str = C.DEFAULT_BINNING) -> list[dict]:
    """Zeilen für `verdict_ece_vs_regret` und die Tabelle des Experiments 3: ECE und Regret roh je Modell mit Rängen (1 = bester) nach ECE, Brier und Regret."""
    rk = measure_ranks(fz, r, bins)
    rows = []
    for x in model_table(fz, r, bins):
        k = x["key"]
        rows.append({"key": k, "ece": x["ece"], "regret": x["roh"][0], "auc": x["auc"], "brier": x["brier"], "slope": x["slope"],
                     "ece_rank": rk["ece"][k], "brier_rank": rk["brier"][k], "regret_rank": rk["regret"][k]})
    return rows
