"""Kern (`ksw_core.py`) an Mini-Instanzen, die von Hand nachgerechnet sind (DEMO-PLAYBOOK §1b). Die unabhängigen Rechenwege stehen in `test_oracle_kalibrierung_schwelle.py`."""

import numpy as np
import pytest

import ksw_core as K


def test_threshold_formula_and_invalid_ratio():
    assert [K.threshold(r) for r in (1, 3, 9, 19)] == [0.5, 0.25, 0.1, 0.05]
    for r in (0, -1, float("nan")):
        with pytest.raises(ValueError):
            K.threshold(r)


def test_expected_cost_of_four_containers_by_hand():
    """p = 0,1 / 0,2 / 0,5 / 0,9 und r = 3. Maßnahme bei den beiden letzten: Kosten (1 − 0,5) + (1 − 0,9) = 0,6; keine bei den ersten: 3 · 0,1 + 3 · 0,2 = 0,9; zusammen 1,5, je Container 0,375."""
    p = np.array([0.1, 0.2, 0.5, 0.9])
    assert K.expected_cost(p > 0.25, p, 3) == pytest.approx(0.375)
    assert K.expected_cost(np.ones(4, bool), p, 3) == pytest.approx((0.9 + 0.8 + 0.5 + 0.1) / 4)       # immer Maßnahme: Σ (1 − p)
    assert K.expected_cost(np.zeros(4, bool), p, 3) == pytest.approx(3 * p.sum() / 4)                   # nie Maßnahme: r Σ p


def test_regret_by_hand_and_by_formula():
    """Orakelregel bei r = 3: Maßnahme bei p > 1/4, also 0,5 und 0,9. Die feste Schwelle 0,5 lässt p = 0,5 aus (p > 0,5 gilt nicht): dieser eine Container kostet 3 · 0,5 − (1 − 0,5) = 1 mehr, je 4 Container 0,25, je 1000: 250."""
    p = np.array([0.1, 0.2, 0.5, 0.9])
    assert K.regret_per_1000(p, p, 3, t=0.5) == pytest.approx(250.0)
    assert K.regret_formula(p, p, 3, 0.5) == pytest.approx(250.0)
    assert K.regret_per_1000(p, p, 3) == 0.0


def test_pav_on_the_hand_sequence():
    """(0, 1, 0, 0, 1, 1): die 1 an Stelle 2 verletzt die Ordnung gegen die folgenden Nullen, der Block (1, 0, 0) hat das Mittel 1/3."""
    np.testing.assert_allclose(K.pav([0, 1, 0, 0, 1, 1]), [0, 1 / 3, 1 / 3, 1 / 3, 1, 1])
    np.testing.assert_allclose(K.pav([1, 0]), [0.5, 0.5])
    np.testing.assert_allclose(K.pav([0, 1]), [0, 1])


def test_pav_with_weights():
    """Gewicht 3 auf dem ersten Wert: Mittel (3 · 1 + 1 · 0) / 4 = 0,75."""
    np.testing.assert_allclose(K.pav([1, 0], [3, 1]), [0.75, 0.75])


def test_isotonic_pools_ties_in_the_prediction_and_interpolates_linearly():
    """Vorhersagen 0,2 (Ergebnisse 0, 1), 0,5 (1), 0,8 (1, 1): Stützstellen 0,2 → 0,5; 0,5 → 1; 0,8 → 1. Zwischen 0,2 und 0,5 wird linear interpoliert, außerhalb gekappt."""
    model = K.isotonic_fit([0.2, 0.2, 0.5, 0.8, 0.8], [0, 1, 1, 1, 1])
    np.testing.assert_allclose(model[0], [0.2, 0.5, 0.8])
    np.testing.assert_allclose(model[1], [0.5, 1.0, 1.0])
    np.testing.assert_allclose(K.isotonic_apply(model, [0.1, 0.2, 0.35, 0.5, 0.9]), [0.5, 0.5, 0.75, 1.0, 1.0])


def test_fit_logit_with_one_binary_column_gives_the_group_log_odds():
    """Ein Achsenabschnitt und eine 0/1-Spalte: der Logit passt die beiden Gruppenhäufigkeiten exakt (Gruppe 0: 2 von 8, Gruppe 1: 6 von 8): β0 = logit(1/4), β0 + β1 = logit(3/4)."""
    x = np.array([0] * 8 + [1] * 8, float)
    y = np.array([1, 1] + [0] * 6 + [1] * 6 + [0] * 2, float)
    beta = K.fit_logit(np.column_stack([np.ones(16), x]), y, penalty=0.0 * np.ones(2))
    assert beta[0] == pytest.approx(np.log(1 / 3), abs=1e-8)
    assert beta[0] + beta[1] == pytest.approx(np.log(3), abs=1e-8)


def test_platt_on_a_perfectly_calibrated_toy_sample():
    """Vorhersage 0,25 mit Häufigkeit 1/4 und 0,75 mit Häufigkeit 3/4 (je 40 Container): die Anpassung auf logit p̂ ist a = 0, b = 1."""
    f = np.array([0.25] * 40 + [0.75] * 40)
    y = np.array([1] * 10 + [0] * 30 + [1] * 30 + [0] * 10, float)
    a, b = K.fit_platt(f, y)
    assert a == pytest.approx(0.0, abs=1e-6) and b == pytest.approx(1.0, abs=1e-6)


def test_ece_by_hand_with_boundary_values():
    """Vorhersage 0,2 (Häufigkeit 0), 0,3 (0,2), 0,4 (0,7), je 10 Container. 5 gleich breite Fächer: 0,2 liegt auf einer Grenze und gehört links geschlossen ins Fach [0,2; 0,4) zu 0,3:
    Fach mit 20 Containern, mittleres p̂ 0,25, Häufigkeit 2/20 = 0,1, Abstand 0,15 mit Anteil 2/3; Fach [0,4; 0,6) mit Abstand 0,3 und Anteil 1/3: ECE 0,1 + 0,1 = 0,2.
    10 und 20 Fächer trennen die drei Gruppen: (0,2 + 0,1 + 0,3) / 3 = 0,2."""
    f = np.repeat([0.2, 0.3, 0.4], 10)
    y = np.array([0] * 10 + [1] * 2 + [0] * 8 + [1] * 7 + [0] * 3, float)
    assert K.ece(f, y, "5 gleich breit") == pytest.approx(0.2)
    assert K.ece(f, y, "10 gleich breit") == pytest.approx(0.2)
    assert K.ece(f, y, "20 gleich breit") == pytest.approx(0.2)
    assert [c for _, _, c in K.reliability_bins(f, y, "5 gleich breit")] == [20, 10]


def test_reliability_bins_counts_and_unknown_binning():
    f = np.array([0.05, 0.15, 0.15, 0.95, 1.0])
    rows = K.reliability_bins(f, np.array([0, 0, 1, 1, 1], float), "10 gleich breit")
    assert [c for _, _, c in rows] == [1, 2, 2]                     # Fach 0, Fach 1, Fach 9 (p̂ = 1 landet im letzten Fach)
    with pytest.raises(ValueError):
        K.reliability_bins(f, np.zeros(5), "7 gleich breit")


def test_auc_by_hand_with_ties():
    """Positive Scores 0,8 und 0,5, negative 0,5 und 0,2: Paare (+,−): 0,8 > 0,5 → 1; 0,8 > 0,2 → 1; 0,5 = 0,5 → 1/2; 0,5 > 0,2 → 1; AUC = 3,5 / 4 = 0,875."""
    assert K.auc(np.array([0.8, 0.5, 0.5, 0.2]), np.array([1, 1, 0, 0], float)) == pytest.approx(0.875)
    assert np.isnan(K.auc(np.array([0.1, 0.2]), np.array([0, 0], float)))


def test_brier_decomposition_on_rounded_predictions():
    f = np.array([0.204, 0.196, 0.5, 0.5])         # runden auf 0,2 / 0,2 / 0,5 / 0,5
    y = np.array([0, 1, 1, 0], float)
    b, rel, res, unc = K.brier_decomposition(f, y)
    assert b == pytest.approx((0.04 + 0.64 + 0.25 + 0.25) / 4)
    assert (rel, res, unc) == (pytest.approx(0.5 * (0.2 - 0.5) ** 2 + 0.5 * 0.0), pytest.approx(0.0), pytest.approx(0.25))
    assert b == pytest.approx(rel - res + unc)


def test_tuned_threshold_by_hand():
    """Rekalibriermenge: p̂ 0,1 (y 0), 0,3 (y 1), 0,6 (y 1), 0,9 (y 0), r = 2. Schwelle 0,2: Maßnahme bei 0,3 / 0,6 / 0,9 → Fehlalarm 1 (0,9) = 1, verpasst 0 → 1; Schwelle 0,05: zusätzlich 0,1 → 2;
    Schwelle 0,95: nie → 2 verpasste · 2 = 4. Beste Kandidatin 0,2."""
    pc = np.array([0.1, 0.3, 0.6, 0.9])
    yc = np.array([0, 1, 1, 0], float)
    assert K.tuned_threshold(pc, yc, 2, [0.05, 0.2, 0.95]) == 0.2


def test_tuned_threshold_first_minimum_wins_on_ties():
    pc = np.array([0.4, 0.6])
    yc = np.array([0, 1], float)
    assert K.tuned_threshold(pc, yc, 1, [0.45, 0.5, 0.55]) == 0.45          # alle drei Kandidaten kosten 0: der erste gilt


def test_logit_clips_the_extremes():
    assert np.isfinite(K.logit(np.array([0.0, 1.0]))).all()
    assert K.logit(0.5) == pytest.approx(0.0)
