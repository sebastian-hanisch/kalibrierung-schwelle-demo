"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): zwei Elemente, konstante Vorhersage, nur eine Klasse, Gleichstand an der Schwelle, Fachgrenzen."""

import numpy as np
import pytest

import ksw_core as K
import ksw_evaluation as E
import ksw_generator as G
import ksw_hand as H
import ksw_models as M


def test_exactly_two_containers_for_pav_isotonic_and_platt():
    """n = 2: der kleinste Fall, in dem es etwas zu vergleichen gibt."""
    np.testing.assert_allclose(K.pav([1, 0]), [0.5, 0.5])
    ux, fit = K.isotonic_fit([0.3, 0.7], [0, 1])
    np.testing.assert_allclose(fit, [0, 1])
    a, b = K.fit_platt([0.3, 0.7], [0, 1])               # getrennt (Separation): kein Fehler, endliche Werte (die Strafe 1e-8 hält sie begrenzt)
    assert np.isfinite(a) and np.isfinite(b) and b > 0


def test_single_container_and_empty_input():
    assert K.pav([1]).tolist() == [1.0]
    assert np.isnan(K.fit_platt([0.4], [1])[1])
    a, b = K.fit_platt([], [])
    assert np.isnan(a) and np.isnan(b)


def test_constant_prediction_leaves_the_slope_undetermined_and_isotonic_has_one_block():
    f = np.full(50, 0.3)
    y = np.array([1] * 15 + [0] * 35, float)
    a, b = K.fit_platt(f, y)
    assert np.isnan(a) and np.isnan(b)
    ux, fit = K.isotonic_fit(f, y)
    assert ux.tolist() == [0.3] and fit.tolist() == [pytest.approx(0.3)]
    assert K.auc(f, y) == pytest.approx(0.5)               # alle Gleichstände: AUC 1/2 nach Definition


def test_recalibration_set_with_a_single_class_does_not_crash():
    """Alle Ergebnisse 1 (oder alle 0): Platt trennt völlig; es gibt endliche Koeffizienten und gültige Wahrscheinlichkeiten."""
    f = np.linspace(0.1, 0.9, 40)
    for y in (np.ones(40), np.zeros(40)):
        a, b = K.fit_platt(f, y)
        assert np.isfinite(a) and np.isfinite(b)
        p = K.platt_apply(f, a, b)
        assert np.all((p >= 0) & (p <= 1))
        ux, fit = K.isotonic_fit(f, y)
        assert np.all(fit == y[0])


def test_prediction_exactly_at_the_threshold_gets_no_measure():
    """Konvention „Maßnahme bei p̂ > Schwelle“: bei p̂ = 1/(1+r) kostet Maßnahme (1 − p) = r p dasselbe wie Unterlassung."""
    r = 3
    t = K.threshold(r)
    assert not (np.array([t]) > t)[0]
    assert (1 - t) == pytest.approx(r * t)
    assert K.regret_per_1000(np.array([t]), np.array([t]), r) == 0.0


def test_ratio_must_be_positive_everywhere():
    for r in (0, -2):
        with pytest.raises(ValueError):
            K.threshold(r)
        with pytest.raises(ValueError):
            K.regret_per_1000(np.array([0.5]), np.array([0.5]), r)
    with pytest.raises(ValueError):
        H.summary(H.DEFAULT_GROUPS, 0)


def test_bin_boundaries_follow_the_left_closed_rule_in_floats_like_in_fractions():
    """Vorhersage genau auf einer Fachgrenze (0,1 · k bei 10 Fächern): links geschlossen, auch wenn 0,3 · 10 in Gleitkomma 3,0000000000000004 oder 2,9999999999999996 ergäbe."""
    for k in range(1, 10):
        f = np.array([k / 10, k / 10])
        rows = K.reliability_bins(f, np.array([0.0, 1.0]), "10 gleich breit")
        assert len(rows) == 1
        # das Fach der Vorhersage k/10 ist das Fach k (nicht k − 1), wie in Brüchen floor(k/10 · 10) = k
        assert K._bin_index(f, 10).tolist() == [k, k]
    assert K._bin_index(np.array([1.0]), 10).tolist() == [9]          # p̂ = 1 gehört ins letzte Fach


def test_only_one_group_is_occupied():
    f = np.full(20, 0.25)
    y = np.array([1] * 5 + [0] * 15, float)
    assert K.ece(f, y, "10 gleich breit") == pytest.approx(0.0)
    assert len(K.reliability_bins(f, y, "10 gleiche Masse")) == 10          # gleiche Masse teilt auch Gleichstände auf


def test_hand_example_with_a_single_group_and_boundary_probabilities():
    one = (H.Group(H.Fr(1, 2), 4, 2),)
    s = H.summary(one, 1)
    assert s["roh"] == s["fest"] == 2 and len(s["alle"]) == 2
    with pytest.raises(ValueError):
        H.Group(H.Fr(1), 4, 2)


def test_training_set_with_fewer_containers_than_columns():
    train = G.make_dataset(12, 1)
    p = M.fit_model("haupt", train.X, train.d, train.y)(G.draw_features(40, 2))
    assert np.all(np.isfinite(p))


def test_regret_of_the_always_act_rule_by_hand():
    """Immer Maßnahme (t = 0): die Orakelregel handelt nur bei p > 1/(1+r); bei den übrigen kostet die Maßnahme (1 − p) statt r p, also 1 − (1 + r) p mehr. Regret = 1000 · Mittel dieser Mehrkosten."""
    p, _ = G.population(2000, 3)
    r = 9
    want = 1000 * np.mean(np.where(p <= 1 / (1 + r), 1 - (1 + r) * p, 0.0))
    assert K.regret_per_1000(np.ones_like(p), p, r, t=0.0) == pytest.approx(want, abs=1e-9)


def test_formatting_helpers():
    assert E.de(3.14159, 2) == "3,14" and E.frac(H.Fr(6, 3)) == "2" and E.frac(H.Fr(2, 15)) == "2/15"
    assert E.tsd(100000) == "100.000" and E.sci(0.000123) == "1,2e-04"
