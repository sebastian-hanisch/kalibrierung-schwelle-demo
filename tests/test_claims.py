"""Claims: die Zahlen der README stehen dort als Text und stimmen mit der vorgerechneten Datei und der Neuberechnung überein (DEMO-PLAYBOOK §4, §11a).

Drei Arten von Prüfung (die Tabelle „Zahl → Test“ steht in den Hinweisen der README):
* **Aus der Datei, exakt:** Tabellen der Abschnitte 2 bis 7 werden aus `ksw_frozen.json` neu formatiert; die Zeile muss in der README stehen. Die Datei selbst gleicht `test_frozen.py` mit dem Code ab.
* **Neu gerechnet, als Band:** Abschnitt 1 (Kosten bei wahrem p) und die Orakel-Kosten werden hier neu gerechnet und mit der README verglichen (Band 0,05 je 1000: Rundung auf eine Stelle).
* **Handwerte im Test:** Kopfzahlen als Zahl im Test (ändert sich die Datei, schlägt es hier an)."""

import re
from pathlib import Path

import numpy as np
import pytest

import ksw_constants as C
import ksw_core as K
import ksw_evaluation as E
import ksw_frozen as FZ
import ksw_models as M

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
DATA = FZ.load()
d = E.de
ms = lambda t, nd=1: f"{d(t[0], nd)} ± {d(t[1], nd)}"
L = M.MODEL_LABELS
cs = lambda k, r: E.c_stats(DATA, k, r)
es = lambda k, m, r: E.e_stats(DATA, k, m, r)
gs = lambda k: E.good_stats(DATA, k)
REPS = 40


def _numbers(text):
    return [float(x.replace(",", ".")) for x in re.findall(r"-?\d+,\d+|-?\d+", text)]


def test_population_table_equals_a_recomputation_as_a_band():
    mat = E.regret_matrix(E.population())
    for r in C.RATIOS:
        row = re.search(rf"^\| {r} \| ([0-9, |]+)\|$", README, re.M)
        assert row, r
        got = _numbers(row.group(1))
        want = [mat[(r, r2)] for r2 in C.RATIOS]
        assert got == pytest.approx(want, abs=0.051), r


def test_headline_cost_of_the_fixed_threshold_at_true_probability():
    mat = E.regret_matrix(E.population())
    assert mat[(9, "fest")] == pytest.approx(561.5, abs=0.05) and mat[(9, 3)] == pytest.approx(111.8, abs=0.05) and mat[(9, 19)] == pytest.approx(20.2, abs=0.05)
    assert f"kostet die feste Schwelle 0,5 bei r = 9 und bekanntem wahrem *p* {d(mat[(9, 'fest')])} Einheiten" in README


def test_population_share_and_oracle_costs_in_the_readme():
    p, art = E.G.population(C.POPULATION_N, C.POPULATION_SEED)
    assert f"Anteil Standgeld {d(100 * float(p.mean()), 1)} % nach der wahren Wahrscheinlichkeit" in README
    assert "23,5 %" in README and abs(float(p.mean()) - 0.2347) < 0.0005                          # Messwert der Quelle (400 000 Container): 0,2347
    orc = DATA["oracle_cost_per_1000"]
    assert f"{d(orc['1'])} / {d(orc['3'])} / {d(orc['9'])} / {d(orc['19'])} bei r = 1 / 3 / 9 / 19" in README
    assert [round(orc[str(r)], 1) for r in C.RATIOS] == pytest.approx([160.4, 277.7, 393.9, 462.5], abs=0.05)


def test_headline_numbers_of_the_main_effects_model():
    h3, h9, h19 = cs("haupt", 3), cs("haupt", 9), cs("haupt", 19)
    assert (ms(h9["roh"]), ms(h9["t05"])) == ("41,7 ± 18,8", "625,3 ± 101,0") and h9["n_t05_gt_roh"] == 40
    assert ms(gs("haupt")["slope"], 2) == "0,81 ± 0,10"
    assert f"{ms(h9['roh'])} mit der Schwelle 1/10 und bei {ms(h9['t05'])} mit 0,5" in README
    assert f"({ms(gs('haupt')['slope'], 2)} auf den Testdaten" in README
    for r, s in ((3, h3), (9, h9), (19, h19)):
        assert f"| {r} | {ms(s['roh'])} | {ms(s['t05'])} | {s['n_t05_gt_roh']} von 40 |" in README
    assert f"| 1 | {ms(cs('haupt', 1)['roh'])} | {ms(cs('haupt', 1)['t05'])} | 0 von 40 |" in README


def test_ratio_of_means_is_stated_with_the_spread_of_the_denominator():
    h = {r: cs("haupt", r) for r in (3, 9, 19)}
    ratio = " / ".join(d(h[r]["t05"][0] / h[r]["roh"][0]) for r in (3, 9, 19))
    spread = " / ".join(d(h[r]["roh"][1]) for r in (3, 9, 19))
    assert f"ist für die Haupteffekte {ratio} bei r = 3 / 9 / 19, aber der Nenner streut mit ± {spread}" in README
    assert ratio == "4,2 / 15,0 / 31,0"
    assert "112" in README and "Differenz" in README                               # die frühere Zahl wird nur als Korrektur genannt


def test_exception_sentence_names_the_shifted_model_with_its_count():
    s = cs("verschoben", 3)
    assert s["n_t05_gt_roh"] == 3
    assert f"beim Modell „verschoben“ (Achsenabschnitt +1) ist 0,5 bei r = 3 nur in {s['n_t05_gt_roh']} von 40 Trainingsmengen teurer als 1/(1+r) ({ms(s['t05'])} gegen {ms(s['roh'])})" in README


def test_model_table_of_calibration_measures():
    rk = E.measure_ranks(DATA, 9)
    for k in M.MODEL_KEYS:
        g = gs(k)
        row = f"| {L[k]} | {d(g['auc'][0], 3)} | {d(g['ece10'][0], 4)} | {rk['ece'][k]} | {d(g['brier'][0], 4)} | {rk['brier'][k]} | {ms(g['slope'], 2)} | {d(cs(k, 9)['roh'][0])} | {rk['regret'][k]} |"
        assert row in README, k
    assert rk["ece"]["ohne_art"] == 3 and rk["regret"]["ohne_art"] == 8 and rk["brier"]["ohne_art"] == 8


def test_ece_ranking_is_the_same_for_all_four_binnings_and_differs_for_brier():
    base = E.measure_ranks(DATA, 9)["ece"]
    for bins in K.BINNINGS:
        assert E.measure_ranks(DATA, 9, bins)["ece"] == base
    assert base != E.measure_ranks(DATA, 9)["brier"]
    assert E.n_models_changing_rank(DATA) == 8 and "für 8 von 8 Modellen ändert sich der Platz zwischen den sechs Maßen" in README
    g = gs("geglaettet")
    assert " / ".join(d(g[x][0], 4) for x in ("ece5", "ece10", "ece20", "ece10m")) in README


def test_mean_calibration_gaps_of_the_logit_models_in_the_readme():
    for k in ("haupt", "je_art", "geglaettet", "ohne_art"):
        assert E.max_mean_gap(DATA, k) < 1e-7
    assert f"Haupteffekte {E.sci(E.max_mean_gap(DATA, 'haupt'))}, je Art {E.sci(E.max_mean_gap(DATA, 'je_art'))}" in README
    assert f"im Training höchstens {E.sci(max(E.max_mean_gap(DATA, k) for k in ('haupt', 'ohne_art', 'geglaettet', 'je_art')))} Abstand" in README
    slopes = " / ".join(ms(gs(k)["slope"], 2) for k in ("haupt", "je_art", "geglaettet", "ohne_art"))
    assert f"die Steigung auf den Testdaten ist dabei {slopes}" in README
    assert gs("je_art")["slope"][0] < 0.5 and gs("geglaettet")["slope"][0] > 1.2             # Steigung unter 1: zu extreme, über 1: zu flache Vorhersagen


def test_ece_floor_values_in_the_readme():
    floor = {row["n"]: row for row in DATA["D"]}
    f = lambda n, k: ms((float(np.mean(floor[n][k])), float(np.std(floor[n][k]))), 4)
    assert f"{f(200, 'ece10')} / {f(2000, 'ece10')} / {f(20000, 'ece10')} / {f(100000, 'ece10')}" in README
    assert f(2000, "ece10") == "0,0178 ± 0,0048" and f(2000, "ece5") in README and f(2000, "ece20") in README
    assert [row["reps"] for row in DATA["D"]] == [200, 200, 100, 20]


def test_regret_tables_after_recalibration():
    for k in M.MODEL_KEYS:
        s = cs(k, 9)
        assert f"| {L[k]} | {ms(s['roh'])} | {ms(s['t05'])} | {ms(s['platt'])} | {ms(s['iso'])} | {ms(s['tuned'])} |" in README, k
        assert f"| {L[k]} | {s['n_t05_gt_roh']} | {s['n_platt_lt_roh']} | {s['n_iso_lt_roh']} | {s['n_platt_lt_iso']} | {s['n_tuned_lt_platt']} |" in README, k


def test_distorted_models_recalibrate_to_the_same_value_and_the_missing_ranking_does_not():
    plat = {k: cs(k, 9)["platt"][0] for k in ("haupt", "verdichtet", "ueberkonfident", "verschoben")}
    base_auc = np.array(DATA["C"]["haupt"]["gute"]["auc"])
    for k in ("verdichtet", "verschoben"):
        assert np.allclose(DATA["C"][k]["gute"]["auc"], base_auc, atol=1e-9, rtol=0)             # AUC gleich in 40 von 40 Trainingsmengen
    diff = np.abs(np.array(DATA["C"]["ueberkonfident"]["gute"]["auc"]) - base_auc)               # „überkonfident“: Kappung bei 1e-9 erzeugt einzelne Gleichstände
    assert int(np.sum(diff < 1e-9)) == 34 and diff.max() < 2.1e-4
    assert "34 von 40 bei „überkonfident“ mit höchstens 2,0·10⁻⁴" in README and abs(diff.max() - 2.0e-4) < 0.05e-4
    assert max(plat.values()) - min(plat.values()) < 0.1                      # Platt ist logit-affin invariant
    for k in ("verdichtet", "ueberkonfident"):
        assert cs(k, 9)["n_platt_lt_roh"] == 40 and cs(k, 9)["n_iso_lt_roh"] == 40
    oa = cs("ohne_art", 9)
    assert (oa["roh"][0] - oa["platt"][0]) / oa["roh"][0] < 0.1                # weniger als 10 % Verbesserung ohne Rangfolge
    ja = cs("je_art", 9)
    assert ja["platt"][0] > ja["roh"][0] and ja["iso"][0] < ja["roh"][0]       # Negativbefund: Platt teurer als roh bei „je Art“


def test_tuned_threshold_statements():
    t = {r: np.array(DATA["C"]["haupt"]["tuned_t"][str(r)]) for r in C.RATIOS}
    from scipy.special import expit
    assert expit(2 * K.logit(0.05)) < 0.02 and round(float(expit(2 * K.logit(0.05))), 3) == 0.003      # Orakelschwelle 1/20 nach der Verzerrung „überkonfident“ liegt unter dem Raster
    assert "expit(2·logit 0,05) ≈ 0,003" in README
    assert f"r = 9: {d(t[9].mean(), 3)} ± {d(t[9].std(), 3)} gegen 0,100; r = 19: {d(t[19].mean(), 3)} ± {d(t[19].std(), 3)} gegen 0,050" in README
    counts = [cs("haupt", r)["n_tuned_lt_platt"] for r in C.RATIOS]
    assert f"billiger als Platt in {counts[0]} / {counts[1]} / {counts[2]} / {counts[3]} von 40" in README and max(counts) < 21


def test_calibration_set_size_table():
    for m in C.CAL_SIZES:
        s = es("haupt", m, 9)
        assert f"| {m} | {d(s['events'], 0)} | {ms(s['roh'])} | {ms(s['platt'])} ({s['n_platt_lt_roh']}) | {ms(s['iso'])} ({s['n_iso_lt_roh']}) | {ms(s['tuned'])} |" in README, m
    few, many = es("haupt", 100, 9), es("haupt", 2000, 9)
    assert few["platt"][0] > few["roh"][0] and few["iso"][0] > few["platt"][0]          # m = 100: Rekalibrieren schadet, isotonisch mehr als Platt
    first = next(m for m in C.CAL_SIZES if es("haupt", m, 9)["n_platt_lt_roh"] > REPS / 2)
    assert first == 1000 and "Platt ist erst ab m = 1000 in mehr als der Hälfte der Trainingsmengen billiger als roh" in README
    assert es("verdichtet", 100, 9)["platt"][0] < es("verdichtet", 100, 9)["roh"][0] and es("verdichtet", 100, 9)["n_platt_lt_roh"] >= 38
    assert all(es("haupt", m, 9)["iso"][0] < es("haupt", m, 9)["platt"][0] for m in (2000, 3000, 10000)) and 20 < few["events"] < 30
    assert es("haupt", 1000, 9)["iso"][0] > es("haupt", 1000, 9)["platt"][0]


def test_second_seed_set_rows():
    F = DATA["F"]
    for k in ("haupt", "geglaettet", "ohne_art"):
        for r in C.RATIOS:
            R = {m: np.array(F["modelle"][k][str(r)][m]) for m in ("roh", "t05", "platt", "iso")}
            mm = lambda a: ms((float(np.mean(a)), float(np.std(a))))
            assert f"| {L[k]} | {r} | {mm(R['roh'])} | {mm(R['t05'])} | {mm(R['platt'])} | {mm(R['iso'])} | {int(np.sum(R['t05'] > R['roh']))} von 40 |" in README
    first, second = cs("haupt", 9)["roh"], (np.mean(F["modelle"]["haupt"]["9"]["roh"]), np.std(F["modelle"]["haupt"]["9"]["roh"]))
    assert abs(first[0] - second[0]) < second[1] and abs(first[0] - second[0]) < first[1]     # innerhalb einer Streuung


def test_ridge_rows_and_the_flat_prediction_case():
    for row in DATA["ridge"]:
        c = str(row["C"]).replace(".", ",")
        assert f"| {c} | {d(row['auc'], 3)} | {d(row['slope'], 2)} | {d(row['ece'], 3)} | {d(row['akt'], 3)} | {d(row['roh'])} | {d(row['platt'])} |" in README
    aucs = [row["auc"] for row in DATA["ridge"]]
    slopes = [row["slope"] for row in DATA["ridge"]]
    assert aucs == sorted(aucs) and slopes == sorted(slopes, reverse=True)           # schwächeres Ridge: AUC steigt, Steigung fällt gegen 1
    assert DATA["ridge"][0]["akt"] == 1.0 and DATA["ridge"][0]["roh_sd"] < 1e-9      # C = 0,003: immer Maßnahme


def test_hand_example_numbers_stand_in_the_readme():
    for text in ("Brier 17/100 = REL 7/150 − RES 13/150 + UNC 21/100", "r = 3: 11 / 27 / 9; r = 1: 9 / 9 / 5; r = 9: 21 / 81 / 11", "0 / 2/15 / 1/5 / 1/5", "(1, 3, 0, 0) und (11, 7, 10, 4) geben 1/4", "7/17"):
        assert text in README


def test_corrections_section_names_the_deviations():
    s = README[README.index("## Befunde und Korrekturen"):README.index("## Quellen")]
    assert "112" in s and "teuerste Regel" in s and "205,9 auf 29,3" in s and "C = 0,003" in s and "60,5" in s


def test_no_planning_words_in_the_readme():
    for word in ("Stück", "Vormessung", "Plan Z", "geplant", "trägt", "tragen", "läuft lokal", "Saatwert"):
        assert word not in README, word


def test_warum_dieses_problem_sentence_has_a_main_verb_and_the_rate_is_scoped():
    s = README[README.index("## Warum dieses Problem"):README.index("## Modell")]
    assert "Bei einem Logit auf allen Merkmalen, angepasst auf 500 Containern, liegt der Regret im Mittel" in s
    assert "Ein Logit auf allen Merkmalen, angepasst" not in README
    assert "gilt für den Generator dieser Demo" in README                                      # Standgeld-Rate (23,5 %) ohne Zahlenvergleich mit anderen Demos
    assert "meiner Erinnerung" not in README                                                   # kein Ich-Bezug der Quellenstufe G


def test_sources_state_their_level_and_name_the_unread_ones():
    s = README[README.index("## Quellen"):README.index("## Ehrliche Grenzen")]
    for text in ("Elkan", "Van Calster", "Dimitriadis", "Niculescu-Mizil", "Stufe W", "Stufe S", "Stufe G", "nicht nachgelesen", "Guo u. a."):
        assert text in s, text


def test_readme_lists_every_module_file():
    files = {p.name for p in Path(__file__).resolve().parent.parent.glob("ksw_*.py")} | {"app.py"}
    for name in files:
        assert f"`{name}`" in README or name.replace(".py", "") in README, name
