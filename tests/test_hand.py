"""Das Beispiel von Hand (`ksw_hand.py`): jede Zahl ist von Hand in Brüchen nachgerechnet (Rechnung im Test, nicht aus dem Code abgeschrieben)."""

import itertools
from fractions import Fraction as Fr

import pytest

import ksw_hand as H


def test_mean_prediction_equals_event_share_but_not_the_slope(hand_groups):
    """Mittleres p̂ = (10 · 1/5 + 10 · 3/10 + 10 · 2/5) / 30 = 9/30 = 3/10; Standgeld 0 + 2 + 7 = 9 von 30 = 3/10: im Mittel kalibriert."""
    assert H.mean_prediction(hand_groups) == Fr(2 + 3 + 4, 30) == Fr(3, 10)
    assert H.event_share(hand_groups) == Fr(9, 30) == Fr(3, 10)
    # in der Steigung nicht: die Häufigkeiten 0, 1/5, 7/10 laufen steiler als die Vorhersagen 1/5, 3/10, 2/5
    assert [g.observed for g in hand_groups] == [0, Fr(1, 5), Fr(7, 10)]


def test_brier_decomposition_by_hand(hand_groups):
    """Brier über die Gruppen ausgeschrieben: Gruppe 1 10 · (1/5)² = 2/5; Gruppe 2 2 · (7/10)² + 8 · (3/10)² = 17/10; Gruppe 3 7 · (3/5)² + 3 · (2/5)² = 3; Summe 5,1 durch 30 = 17/100."""
    g1 = 10 * Fr(1, 5) ** 2
    g2 = 2 * (1 - Fr(3, 10)) ** 2 + 8 * Fr(3, 10) ** 2
    g3 = 7 * (1 - Fr(2, 5)) ** 2 + 3 * Fr(2, 5) ** 2
    assert (g1 + g2 + g3) / 30 == Fr(17, 100)
    brier, rel, res, unc = H.brier_decomposition(hand_groups)
    assert (brier, rel, res, unc) == (Fr(17, 100), Fr(7, 150), Fr(13, 150), Fr(21, 100))
    assert brier == rel - res + unc


def test_brier_equals_the_sum_over_single_containers(hand_groups):
    singles = []
    for g in hand_groups:
        singles += [(g.p_hat, 1)] * g.events + [(g.p_hat, 0)] * (g.n - g.events)
    assert sum((f - y) ** 2 for f, y in singles) / len(singles) == Fr(17, 100)


@pytest.mark.parametrize("r, roh, fest, beobachtet, bestes", [(3, 11, 27, 9, 9), (1, 9, 9, 5, 5), (9, 21, 81, 11, 11)])
def test_costs_of_the_rules_by_hand(hand_groups, r, roh, fest, beobachtet, bestes):
    """r = 3, Schwelle 1/4: p̂ > 1/4 trifft Gruppe 2 und 3 (Fehlalarme 8 + 3 = 11, keine verpassten); fest 1/2: nie Maßnahme, verpasst (0 + 2 + 7) · 3 = 27; nach Häufigkeit 0, 1/5, 7/10 > 1/4:
    nur Gruppe 3, Kosten 3 Fehlalarme + 2 verpasste · 3 = 9. Die anderen Zeilen entsprechend (r = 1: 9 verpasste; r = 9: 21 Fehlalarme gegen (2 + 7) · 9 = 81)."""
    s = H.summary(hand_groups, r)
    assert (s["roh"], s["fest"], s["beobachtet"], s["bestes"]) == (roh, fest, beobachtet, bestes)


def test_rule_after_observed_frequency_is_the_best_of_all_eight(hand_groups):
    for r in (1, 3, 9, 19):
        s = H.summary(hand_groups, r)
        assert len(s["alle"]) == 8 and s["beobachtet"] == s["bestes"]


def test_ece_depends_on_the_number_of_bins(hand_groups):
    assert [H.ece(hand_groups, b) for b in (2, 4, 5, 10)] == [0, Fr(2, 15), Fr(1, 5), Fr(1, 5)]
    # p̂ = 1/5 liegt bei 5 Fächern genau auf einer Fachgrenze: rechts geschlossen gehört sie ins Fach darunter
    assert H.ece(hand_groups, 5, left_closed=True) == Fr(1, 5) and H.ece(hand_groups, 5, left_closed=False) == Fr(2, 15)


@pytest.mark.parametrize("matrix, expected", [((1, 3, 0, 0), Fr(1, 4)), ((11, 7, 10, 4), Fr(1, 4)), ((3, 3, 0, 2), Fr(3, 4)), ((1, 3, 0, -2), Fr(1, 6))])
def test_threshold_from_a_cost_matrix(matrix, expected):
    """(11, 7, 10, 4) = (1, 3, 0, 0) plus 10 auf die Spalte „kein Ereignis“ und 4 auf die Spalte „Ereignis“: dieselbe Schwelle; +2 auf die Zeile „Maßnahme“ verschiebt sie auf 3/4."""
    assert H.threshold_from_matrix(*matrix) == expected
    a, b, c00, c11 = matrix
    n = 2520
    first = next(Fr(k, n) for k in range(n + 1) if Fr(k, n) * c11 + (1 - Fr(k, n)) * a <= Fr(k, n) * b + (1 - Fr(k, n)) * c00)   # Aufzählung
    assert first == expected


def test_threshold_of_the_r_form():
    assert all(H.threshold_from_matrix(1, r) == Fr(1, 1 + r) for r in (1, 3, 9, 19))
    with pytest.raises(ValueError):
        H.threshold_from_matrix(0, 3)           # a − c00 = 0: die Maßnahme ist nie schlechter
    with pytest.raises(ValueError):
        H.threshold_from_matrix(3, 1, c11=1)    # b − c11 = 0


def test_threshold_is_the_tau_quantile_of_a_zero_one_outcome():
    """Maßnahme bei p > 1/(1+r) genau dann, wenn das τ-Quantil von Bernoulli(p) gleich 1 ist, τ = r/(1+r): Aufzählung über p = k/40 und τ = j/40, 1599 Paare."""
    count = bad = 0
    for k in range(41):
        for j in range(1, 40):
            p, tau = Fr(k, 40), Fr(j, 40)
            a, b = 1 - tau, tau
            quantile = 0 if (1 - p) >= tau else 1             # kleinstes q mit F(q) >= τ, F(0) = 1 − p
            count += 1
            bad += (p > H.threshold_from_matrix(a, b)) != (quantile == 1)
    assert count == 1599 and bad == 0


def test_hand_example_with_pure_logit_distortion():
    """Wahre Quote 1/16, 1, 16 (p = 1/17, 1/2, 16/17), Vorhersage mit der Wurzel der Quote (1/5, 1/2, 4/5); die Quote zu quadrieren stellt p wieder her."""
    odds = lambda p: p / (1 - p)
    back = lambda o: o / (1 + o)
    truth = [back(Fr(1, 16)), back(Fr(1)), back(Fr(16))]
    pred = [back(Fr(1, 4)), back(Fr(1)), back(Fr(4))]
    assert truth == [Fr(1, 17), Fr(1, 2), Fr(16, 17)] and pred == [Fr(1, 5), Fr(1, 2), Fr(4, 5)]
    assert [back(odds(p) ** 2) for p in pred] == truth
    t = Fr(1, 10)                                            # r = 9
    assert [p > t for p in truth] == [False, True, True] and [p > t for p in pred] == [True, True, True]
    extra = (1 - truth[0]) - 9 * truth[0]                    # unnötige Maßnahme kostet 1 − p, Unterlassung 9 p
    assert extra == Fr(7, 17)
    assert [p > Fr(1, 2) for p in pred] == [p > Fr(1, 2) for p in truth]    # bei r = 1 ändert die Verzerrung keine Entscheidung


def test_group_validation():
    with pytest.raises(ValueError):
        H.Group(Fr(0), 10, 0)
    with pytest.raises(ValueError):
        H.Group(Fr(1, 2), 10, 11)
    with pytest.raises(ValueError):
        H.Group(Fr(1, 2), 0, 0)
    with pytest.raises(ValueError):
        H.summary(H.DEFAULT_GROUPS, 0)


def test_cost_curve_is_a_step_function_with_the_right_ends(hand_groups):
    curve = H.cost_curve(hand_groups, 3)
    assert curve[0] == (0.0, 21.0) and curve[-1][1] == 27.0                  # t = 0: Maßnahme überall (21 Fehlalarme); t = 1: nie (9 verpasste · 3)
    at_quarter = dict(curve)[0.25]
    assert at_quarter == 11.0 == float(H.summary(hand_groups, 3)["roh"])       # bei der Kostenschwelle 1/4 gilt die rohe Regel
    assert min(c for _, c in curve) == 9.0 == float(H.summary(hand_groups, 3)["bestes"])   # eine Schwelle zwischen 0,3 und 0,4 trennt hier die Gruppen richtig


def test_all_rule_costs_by_enumeration(hand_groups):
    allc = H.all_rules_costs(hand_groups, 3)
    expect = {act: sum((g.n - g.events) if a else g.events * 3 for a, g in zip(act, hand_groups)) for act in itertools.product((False, True), repeat=3)}
    assert allc == expect
