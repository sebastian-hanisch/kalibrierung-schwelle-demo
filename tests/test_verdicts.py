"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zeilen, nicht Zufalls-Seeds: der Test prüft die Grenzen und jede genannte Zahl direkt."""

from fractions import Fraction as Fr

import ksw_constants as C
import ksw_evaluation as E


def _hand(roh, fest, beob, best, r=3, n_rules=8):
    return {"r": r, "t": Fr(1, 1 + r), "roh": Fr(roh), "fest": Fr(fest), "beobachtet": Fr(beob), "bestes": Fr(best), "alle": {i: 0 for i in range(n_rules)}}


def test_hand_verdict_success_names_all_costs_and_the_difference():
    kind, text = E.verdict_hand(_hand(11, 27, 9, 9), 30)
    assert kind == "success"
    assert "r = 3 (Schwelle 1/4), 30 Container" in text and "roh (nach der Vorhersage) 11" in text and "feste Schwelle 1/2: 27" in text
    assert "nach den beobachteten Häufigkeiten 9;" in text and "beste aller 8 Gruppenregeln kostet 9" in text and "kostet 2 mehr" in text


def test_hand_verdict_all_rules_equal_is_info():
    kind, text = E.verdict_hand(_hand(5, 5, 5, 5), 12)
    assert kind == "info" and "entscheiden alle drei Regeln gleich" in text


def test_hand_verdict_observed_best_but_raw_equal_is_info_without_difference_sentence():
    kind, text = E.verdict_hand(_hand(9, 12, 9, 9), 30)
    assert kind == "info" and "ist die beste" in text and "mehr als" not in text


def test_hand_verdict_observed_not_best_is_a_warning():
    kind, text = E.verdict_hand(_hand(11, 27, 10, 9), 30)
    assert kind == "warning" and "nicht die beste" in text


def _stat(roh=(40.0, 10.0), t05=(600.0, 100.0), n=40, reps=40):
    return {"roh": roh, "t05": t05, "n_t05_gt_roh": n, "reps": reps}


def test_threshold_verdict_for_ratio_nine_names_means_counts_and_population_value():
    kind, text = E.verdict_threshold(9, "Haupteffekte", _stat(), 561.5, 40)
    assert kind == "warning"
    assert "r = 9, 40 Trainingsmengen zu je 500 Containern" in text and "kostet im Mittel 600,0 ± 100,0 Einheiten je 1000" in text
    assert "1/(1+r) = 0,100 nur 40,0 ± 10,0" in text and "teurer in 40 von 40" in text and "kostet 0,5 561,5" in text and "Ausnahme" not in text


def test_threshold_verdict_names_the_exception_in_the_same_text():
    kind, text = E.verdict_threshold(3, "verschoben", _stat(roh=(53.2, 13.7), t05=(28.2, 9.5), n=3), 80.6, 40)
    assert kind == "info" and "teurer in 3 von 40" in text and "Ausnahme: in 37 von 40 Trainingsmengen war 0,5 nicht teurer" in text and "billigere Regel" in text


def test_threshold_verdict_for_ratio_one_reports_no_difference_and_no_counts():
    kind, text = E.verdict_threshold(1, "Haupteffekte", _stat(roh=(11.6, 4.7), t05=(11.6, 4.7), n=0), 0.0, 40)
    assert kind == "info" and "keinen Unterschied" in text and "11,6 ± 4,7" in text and "teurer in" not in text


def _gstat(slope=(0.81, 0.10)):
    return {"slope": slope, "ece10": (0.0301, 0.005)}


def test_calibration_verdict_threshold_is_inclusive_at_one_tenth():
    assert E.verdict_calibration("M", _gstat((1.1, 0.1)), 1e-9, 2, 1, 2, 9, 40)[0] == "success"
    assert E.verdict_calibration("M", _gstat((0.9, 0.1)), 1e-9, 2, 1, 2, 9, 40)[0] == "success"
    assert "Achsenabschnitt" in E.verdict_calibration("M", _gstat((1.0, 0.1)), 1e-9, 2, 1, 2, 9, 40)[1]     # das Urteil nennt, dass der Achsenabschnitt nicht eingeht
    assert E.verdict_calibration("M", _gstat((1.101, 0.1)), 1e-9, 2, 1, 2, 9, 40)[0] == "warning"
    assert E.verdict_calibration("M", _gstat((0.899, 0.1)), 1e-9, 2, 1, 2, 9, 40)[0] == "warning"


def test_calibration_verdict_names_slope_gap_ranks_and_direction():
    kind, text = E.verdict_calibration("Haupteffekte", _gstat(), 1.2e-13, 2, 1, 3, 9, 40)
    assert kind == "warning"
    assert "Steigung der Kalibrierung auf den Testdaten 0,81 ± 0,10" in text and "40 Trainingsmengen" in text and "im Training 1,2e-13" in text
    assert "ECE (10 gleich breite Fächer) 0,030, Rang 2 von 8" in text and "Rang nach Brier 1" in text and "Rang nach Regret bei r = 9 3" in text and "zu extreme" in text
    assert "zu flache" in E.verdict_calibration("M", _gstat((1.6, 0.2)), 0.0, 5, 3, 4, 9, 40)[1]


def _est(roh=40.0, platt=35.0, iso=36.0, tuned=38.0, n_p=30, n_i=28, events=471.0):
    return {"roh": (roh, 18.0), "platt": (platt, 17.0), "iso": (iso, 16.0), "tuned": (tuned, 15.0), "n_platt_lt_roh": n_p, "n_iso_lt_roh": n_i, "events": events, "reps": 40}


def test_recalibration_verdict_success_names_every_number():
    kind, text = E.verdict_recalibration(9, "Haupteffekte", 2000, _est(), 40)
    assert kind == "success"
    assert "r = 9, Rekalibriermenge m = 2000 (im Mittel 471 Container mit Standgeld), 40 Trainingsmengen" in text
    assert "roh 40,0 → Platt 35,0 (billiger als roh in 30 von 40), isotonisch 36,0 (billiger in 28 von 40), abgestimmte Schwelle 38,0" in text


def test_recalibration_verdict_branches():
    assert E.verdict_recalibration(9, "M", 100, _est(platt=50.0, iso=60.0), 40)[0] == "warning"      # beide teurer als roh: Rekalibrieren schadet
    assert E.verdict_recalibration(9, "M", 100, _est(platt=50.0, iso=30.0), 40)[0] == "info"         # nur Platt teurer
    assert E.verdict_recalibration(9, "M", 100, _est(platt=30.0, iso=50.0), 40)[0] == "info"         # nur isotonisch teurer
    assert E.verdict_recalibration(9, "M", 100, _est(platt=40.0, iso=30.0), 40)[0] == "info"         # Gleichstand mit roh zählt nicht als billiger


def _matrix_by_hand():
    m = {}
    for r in C.RATIOS:
        for r2 in C.RATIOS:
            m[(r, r2)] = 0.0 if r == r2 else 100.0 + 10 * r + r2
        m[(r, "fest")] = 500.0 + r
    return m


def test_wrong_ratio_verdict_names_every_other_assumption_and_the_fixed_threshold():
    kind, text = E.verdict_wrong_ratio(9, _matrix_by_hand())
    assert kind == "info"
    assert "Wahres r = 9" in text and "r′ = 3: 193,0, r′ = 19: 209,0" in text and "die feste Schwelle 0,5 (das ist r′ = 1): 509,0" in text
    assert text.endswith("Jedes andere falsch angenommene r′ kostet hier weniger als die feste 0,5.")


def test_wrong_ratio_verdict_when_a_wrong_assumption_costs_more_than_the_fixed_threshold():
    m = _matrix_by_hand()
    m[(3, "fest")] = 50.0
    assert E.verdict_wrong_ratio(3, m)[1].endswith("Ein falsch angenommenes r′ kann hier mehr kosten als die feste 0,5.")


def test_wrong_ratio_verdict_for_ratio_one():
    kind, text = E.verdict_wrong_ratio(1, _matrix_by_hand())
    assert kind == "info" and "die Kostenschwelle ist 0,5" in text and "r′ = 3: 113,0" in text and "r′ = 19: 129,0" in text and "r′ = 1:" not in text


def _rows(ece_rank_of_worst):
    base = [("haupt", 0.02, 40.0, 2), ("lognormal", 0.01, 35.0, 1), ("ohne_art", 0.04, 400.0, ece_rank_of_worst), ("je_art", 0.07, 160.0, 6)]
    return [{"key": k, "ece": e, "regret": r, "ece_rank": rk} for k, e, r, rk in base]


def test_ece_verdict_warns_when_the_worst_model_has_a_good_ece_rank():
    kind, text = E.verdict_ece_vs_regret(_rows(2), 9, (0.0178, 0.0048), 3, "10 gleich breit")
    assert kind == "warning"
    assert "ECE (10 gleich breit), die drei kleinsten von 4 Modellen" in text and "(Rang 1)" in text
    assert "höchste Regret bei r = 9 (roh) gehört zu „ohne Containerart (Rangfolge fehlt)“: 400,0 je 1000, ECE-Rang 2" in text
    assert "0,0178 ± 0,0048 (Boden des Maßes)" in text and "bei 3 von 4 Modellen" in text


def test_ece_verdict_is_info_when_the_worst_model_also_has_a_bad_ece_rank():
    kind, text = E.verdict_ece_vs_regret(_rows(4), 9, (0.0178, 0.0048), 2, "5 gleich breit")
    assert kind == "info" and "auch einen großen ECE" in text and "ECE (5 gleich breit)" in text


def _size_rows(counts):
    return [{"m": m, "events": m / 4.2, "roh": 41.7, "platt": 40.0, "iso": 45.0, "n_platt_lt_roh": c} for m, c in zip((100, 300, 1000), counts)]


def test_sample_size_verdict_names_the_first_size_with_more_than_half_of_the_sets():
    kind, text = E.verdict_sample_size(9, "Haupteffekte", _size_rows((16, 20, 26)), 40)
    assert kind == "info"
    assert "Regret je 1000 roh 41,7" in text and "m = 100: Platt 40,0, isotonisch 45,0 (Platt billiger als roh in 16 von 40)" in text
    assert text.endswith("Platt ist ab m = 1000 (im Mittel 238 Container mit Standgeld) in mehr als der Hälfte der Trainingsmengen billiger als roh.")


def test_sample_size_verdict_exactly_half_is_not_more_than_half():
    kind, text = E.verdict_sample_size(9, "Haupteffekte", _size_rows((20, 20, 20)), 40)
    assert kind == "warning" and "Bei keiner gemessenen Größe" in text


def _model_row(key, auc, roh, platt, iso, n_p=0, n_i=0, n_t=0):
    return {"key": key, "auc": auc, "roh": (roh, 1.0), "platt": (platt, 1.0), "iso": (iso, 1.0), "n_platt_lt_roh": n_p, "n_iso_lt_roh": n_i, "n_tuned_lt_platt": n_t}


def test_no_rescue_verdict_names_the_models_where_platt_does_not_help():
    rows = [_model_row("haupt", 0.88, 40.0, 38.0, 37.0, n_t=19), _model_row("ohne_art", 0.68, 400.0, 360.0, 350.0), _model_row("je_art", 0.84, 160.0, 180.0, 140.0, n_p=12, n_i=37)]
    kind, text = E.verdict_no_rescue(9, rows, 40)
    assert kind == "info"
    assert "„ohne Containerart“ (AUC 0,680): roh 400,0 → Platt 360,0 (10 % weniger), isotonisch 350,0" in text
    assert "Platt billiger als roh in 12 von 40 Trainingsmengen (180,0 gegen 160,0), isotonisch in 37 von 40" in text
    assert "Platt im Mittel nicht billiger ist als roh: je Art (Wechselwirkung, überangepasst)." in text and "billiger als Platt in 19 von 40" in text


def test_no_rescue_verdict_without_any_failing_model_says_none():
    rows = [_model_row("haupt", 0.88, 40.0, 38.0, 37.0), _model_row("ohne_art", 0.68, 400.0, 360.0, 350.0), _model_row("je_art", 0.84, 160.0, 150.0, 140.0)]
    assert "nicht billiger ist als roh: keines." in E.verdict_no_rescue(9, rows, 40)[1]


def test_formatting_of_numbers_in_the_texts():
    assert E.de(1234.567, 1) == "1234,6" and E.de(0.0178, 4) == "0,0178" and E.tsd(20000) == "20.000"
