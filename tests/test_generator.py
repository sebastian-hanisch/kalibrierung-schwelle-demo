"""Generator des Containerstroms (`ksw_generator.py`): Modellformeln von Hand, Verteilungen gegen gemessene Werte der Quelle als Band, Eigenschaften des Zufallsstroms."""

import math
from statistics import NormalDist

import numpy as np
import pytest

import ksw_generator as G

ND = NormalDist()


def test_customer_effects_equal_the_numpy_stream_of_the_source():
    """Die zwölf Kundeneffekte sind als Zahlen übernommen; die Quelle zieht sie mit numpy.random.default_rng(12345).normal(0, 0,3, 12)."""
    np.testing.assert_allclose(G.UK, np.random.default_rng(12345).normal(0.0, 0.30, 12), atol=1e-15, rtol=0)


def test_true_probability_by_hand_for_a_leer_container():
    """Leer (Art 3), Kunde 0, Wochenende 0: Median 7,0 Tage, mu = ln 7 + UK[0], s = 0,9. p = P(Verweilzeit > 7) = 1 − Φ((ln 7 − mu) / s) = Φ(UK[0] / 0,9)."""
    mu, s = G.log_median_and_s(3, 0, 0, 0, 0.0)
    p = G.true_probability(mu, s)
    assert float(mu) == pytest.approx(math.log(7.0) + G.UK[0]) and float(s) == pytest.approx(0.9)
    assert float(p) == pytest.approx(ND.cdf(G.UK[0] / 0.9), abs=1e-12)


def test_true_probability_by_hand_for_an_import_with_customs_hold_on_a_weekend():
    """Import (Art 0), Kunde 1, Zoll 1, Wochenende 1: Median 3,5 · e^(0,7 + 0,3) = 3,5 e; Kundeneffekt UK[1] · 1,0; s = 0,55."""
    mu, s = G.log_median_and_s(0, 1, 1, 1, 0.0)
    expected_mu = math.log(3.5) + 1.0 + G.UK[1]
    assert float(mu) == pytest.approx(expected_mu) and float(s) == pytest.approx(0.55)
    assert float(G.true_probability(mu, s)) == pytest.approx(1 - ND.cdf((math.log(7.0) - expected_mu) / 0.55), abs=1e-12)


def test_export_median_grows_with_the_lead_time_and_customer_effect_is_a_third():
    mu0, _ = G.log_median_and_s(1, 3, 0, 0, 2.0)
    mu1, _ = G.log_median_and_s(1, 3, 0, 0, 8.0)
    assert float(mu0) == pytest.approx(math.log(0.6 + 0.85 * 2.0) + 0.3 * G.UK[3])
    assert float(mu1) == pytest.approx(math.log(0.6 + 0.85 * 8.0) + 0.3 * G.UK[3])


def test_population_matches_the_measured_values_of_the_source_as_a_band():
    """Messwerte der Quelle (400 000 Container, andere Zufalls-Seeds): Anteil Standgeld 0,2347 (Mittel der wahren p); je Art 0,265 / 0,267 / 0,038 / 0,524; Anteil p > 0,5: 0,197; > 0,25: 0,362; > 0,1: 0,510; > 0,05: 0,601."""
    p, art = G.population(100_000, 7)
    assert p.mean() == pytest.approx(0.2347, abs=0.004)
    for a, want in enumerate((0.265, 0.267, 0.038, 0.524)):
        assert p[art == a].mean() == pytest.approx(want, abs=0.012)
    for t, want in ((0.5, 0.197), (0.25, 0.362), (0.1, 0.510), (0.05, 0.601)):
        assert np.mean(p > t) == pytest.approx(want, abs=0.012)


def test_feature_distributions():
    X = G.draw_features(100_000, 3)
    art = X[:, 0].astype(int)
    for a, share in enumerate(G.ART_P):
        assert np.mean(art == a) == pytest.approx(share, abs=0.01)
    assert set(np.unique(X[:, 1])) == set(range(12)) and np.all(np.abs(np.bincount(X[:, 1].astype(int)) / len(X) - 1 / 12) < 0.01)
    assert np.all(X[art != 0, 2] == 0) and np.mean(X[art == 0, 2]) == pytest.approx(0.25, abs=0.015)          # Zollhalt nur bei Import
    assert np.mean(X[:, 3]) == pytest.approx(2 / 7, abs=0.01)                                              # Wochenende: Tage 5 und 6 von 7
    assert np.all(X[(art == 0) | (art == 3), 4] == 0)
    assert 1.0 <= X[art == 1, 4].min() and X[art == 1, 4].max() <= 10.0 and 0.5 <= X[art == 2, 4].min() and X[art == 2, 4].max() <= 6.0
    assert X[:, 5].mean() == pytest.approx(0.0, abs=0.02) and X[:, 5].std() == pytest.approx(1.0, abs=0.02)       # Rauschmerkmal


def test_log_dwell_time_is_standardized_normal_and_standgeld_is_dwell_above_seven_days():
    d = G.make_dataset(100_000, 4)
    z = (np.log(d.d) - d.mu) / d.s
    assert z.mean() == pytest.approx(0.0, abs=0.015) and z.std() == pytest.approx(1.0, abs=0.015)
    assert np.array_equal(d.y, (d.d > 7.0).astype(float))
    assert abs(d.y.mean() - d.p.mean()) < 0.01


def test_noise_feature_is_independent_of_the_outcome():
    d = G.make_dataset(100_000, 5)
    assert abs(np.corrcoef(d.X[:, 5], d.y)[0, 1]) < 0.02


def test_design_matrix_has_19_columns_with_exclusive_art_dummies():
    X = G.draw_features(500, 8)
    F = G.design(X)
    assert F.shape == (500, 19) and np.all(F[:, 0] == 1)
    assert np.all(F[:, 1:4].sum(axis=1) <= 1)                        # Import ist der Bezug: keine Art-Spalte gesetzt
    assert np.all(F[:, 8:].sum(axis=1) <= 1) and G.design(X, with_customer=False).shape == (500, 8)


def test_population_without_noise_feature_uses_the_same_probabilities():
    p1, _ = G.population(1000, 12)
    d = G.make_dataset(1000, 12)
    assert np.array_equal(p1, d.p)
