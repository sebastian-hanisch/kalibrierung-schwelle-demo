"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Aussagen, die für Schwelle, Rekalibrierung und Gütemaße gelten müssen, ohne dass das richtige Ergebnis bekannt ist.
Je Eigenschaft 20 bis 50 Zufallsinstanzen mit festem Zufalls-Seed (numpy-Strom nur in den Tests)."""

import numpy as np
import pytest
from scipy import special

import ksw_core as K
import ksw_generator as G
import ksw_models as M


def _sample(rng, n=None):
    n = n or int(rng.integers(80, 600))
    z = rng.standard_normal(n) * 1.3
    p = special.expit(-1.0 + z)
    return special.expit(0.3 + 0.7 * z + rng.normal(0, 0.15, n)), p, (rng.random(n) < p).astype(float)


def test_regret_is_never_negative_and_zero_at_true_probability():
    rng = np.random.default_rng(10)
    for _ in range(30):
        f, p, _y = _sample(rng)
        for r in (1, 3, 9, 19):
            assert K.regret_per_1000(f, p, r) >= -1e-9
            assert K.regret_per_1000(p, p, r) == 0.0


def test_with_ratio_one_the_cost_threshold_and_the_fixed_threshold_are_the_same_rule():
    rng = np.random.default_rng(11)
    for _ in range(30):
        f, p, _y = _sample(rng)
        assert K.regret_per_1000(f, p, 1) == K.regret_per_1000(f, p, 1, t=0.5)


def test_platt_is_invariant_to_logit_affine_distortions_of_the_prediction():
    """Dieselbe Rekalibrierung nach jeder Verzerrung p̃ = expit(a + b logit p̂) (b ≠ 0): die rekalibrierten Vorhersagen sind gleich (die Platt-Familie ist unter logit-affinen Abbildungen abgeschlossen)."""
    rng = np.random.default_rng(12)
    for _ in range(30):
        f, _p, y = _sample(rng)
        a, b = float(rng.uniform(-1, 1)), float(rng.choice([0.5, 2.0, 3.0]))
        base = K.platt_apply(f, *K.fit_platt(f, y))
        distorted = special.expit(a + b * K.logit(f))
        np.testing.assert_allclose(K.platt_apply(distorted, *K.fit_platt(distorted, y)), base, atol=1e-5)


def test_auc_is_unchanged_by_a_strictly_monotone_distortion():
    rng = np.random.default_rng(13)
    for _ in range(30):
        f, _p, y = _sample(rng)
        assert K.auc(special.expit(1.0 + 0.5 * K.logit(f)), y) == pytest.approx(K.auc(f, y), abs=1e-12)


def test_row_order_changes_neither_platt_nor_isotonic_nor_the_measures():
    rng = np.random.default_rng(14)
    for _ in range(25):
        f, p, y = _sample(rng)
        perm = rng.permutation(len(f))
        assert K.fit_platt(f[perm], y[perm]) == pytest.approx(K.fit_platt(f, y), abs=1e-9)
        for a, b in zip(K.isotonic_fit(f[perm], y[perm]), K.isotonic_fit(f, y)):
            np.testing.assert_allclose(a, b, atol=1e-12)
        assert K.ece(f[perm], y[perm]) == pytest.approx(K.ece(f, y), abs=1e-12)
        assert K.brier(f[perm], y[perm]) == pytest.approx(K.brier(f, y), abs=1e-12)
        assert K.auc(f[perm], y[perm]) == pytest.approx(K.auc(f, y), abs=1e-12)


def test_pav_keeps_the_sum_and_is_nondecreasing():
    rng = np.random.default_rng(15)
    for _ in range(40):
        y = rng.integers(0, 2, int(rng.integers(2, 60))).astype(float)
        fit = K.pav(y)
        assert fit.sum() == pytest.approx(y.sum(), abs=1e-9) and np.all(np.diff(fit) >= -1e-12)


def test_mean_calibration_in_training_holds_with_an_unpenalized_intercept_also_with_ridge():
    rng = np.random.default_rng(16)
    for lam in (1e-4, 1.0, 33.3):
        for _ in range(8):
            n = 150
            X = rng.standard_normal((n, 8))
            y = (rng.random(n) < special.expit(X[:, 0] - 0.5)).astype(float)
            F = np.column_stack([np.ones(n), X])
            beta = K.fit_logit(F, y, penalty=lam)
            assert np.sum(y - special.expit(F @ beta)) == pytest.approx(0.0, abs=1e-7)


def test_mean_calibration_does_not_hold_with_a_penalized_intercept():
    """Gegenprobe: bestraft man den Achsenabschnitt mit, gilt die Gleichung nicht mehr (hier λ = 50 auf allen Spalten)."""
    rng = np.random.default_rng(17)
    n = 100
    y = (rng.random(n) < 0.8).astype(float)
    F = np.ones((n, 1))
    beta = K.fit_logit(F, y, penalty=np.array([50.0]))
    assert abs(np.sum(y - special.expit(F @ beta))) > 1.0


def test_replacing_the_prediction_by_the_true_probability_gives_zero_regret_and_a_floor_ece():
    p, _ = G.population(20_000, 21)
    y = (np.random.default_rng(22).random(len(p)) < p).astype(float)
    assert K.regret_per_1000(p, p, 9) == 0.0
    assert 0.0 < K.ece(p, y, "10 gleich breit") < 0.02            # nicht 0, aber Rauschen (Boden bei 20 000 Containern: etwa 0,006)


def test_more_recalibration_containers_shrink_the_spread_of_the_tuned_threshold():
    """Mittel über 30 Wiederholungen: die auf den Kosten gewählte Schwelle streut bei m = 3000 weniger um 1/(1+r) als bei m = 300 (r = 9, Vorhersage = wahres p)."""
    spread = {}
    for m in (300, 3000):
        chosen = []
        for rep in range(30):
            ds = G.make_dataset(m, 5000 + rep)
            chosen.append(K.tuned_threshold(ds.p, ds.y, 9, np.linspace(0.02, 0.98, 49)))
        spread[m] = float(np.std(chosen))
    assert spread[3000] < spread[300]


def test_same_seed_same_data_and_different_seed_different_data():
    a, b, c = G.make_dataset(200, 5), G.make_dataset(200, 5), G.make_dataset(200, 6)
    assert np.array_equal(a.X, b.X) and np.array_equal(a.y, b.y) and np.array_equal(a.d, b.d)
    assert not np.array_equal(a.X, c.X)


def test_first_rows_do_not_depend_on_the_number_of_rows():
    small, large = G.make_dataset(50, 9), G.make_dataset(400, 9)
    assert np.array_equal(small.X, large.X[:50]) and np.array_equal(small.y, large.y[:50])


def test_distorting_models_keep_the_ranking_and_only_change_the_scale():
    train = G.make_dataset(500, 500)
    test = G.make_dataset(3000, 999)
    base = M.fit_model("haupt", train.X, train.d, train.y)(test.X)
    for key in ("verdichtet", "ueberkonfident", "verschoben"):
        dist = M.fit_model(key, train.X, train.d, train.y)(test.X)
        assert K.auc(dist, test.y) == pytest.approx(K.auc(base, test.y), abs=1e-9)
        assert np.all(np.diff(dist[np.argsort(base)]) >= -1e-12)
