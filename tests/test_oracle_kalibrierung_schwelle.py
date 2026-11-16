"""Orakel-Tests (ORACLE-PLAYBOOK §2, §5; DEMO-PLAYBOOK §12): jede Kernroutine gegen einen anderen Rechenweg.

* PAV: Aufzählung aller Blockzerlegungen in Brüchen (`fractions`) und scikit-learn `IsotonicRegression`.
* Isotonische Vorhersage zwischen den Stützstellen: scikit-learn `predict` mit `out_of_bounds="clip"`.
* Platt-Newton und Logit mit Ridge: scikit-learn `LogisticRegression(solver="newton-cholesky")` (kein BFGS: ein Löser ohne Gradient streut, ORACLE-PLAYBOOK §2).
* Regret: Formel „Σ |(1 + r) p − 1| über die abweichenden Entscheidungen“ gegen die direkte Kostenrechnung; Schwelle 1/(1+r) gegen die Aufzählung über ein Schwellenraster.
* Brier-Zerlegung in Brüchen; ECE gegen eine Schleife mit `np.digitize`; AUC gegen scikit-learn; abgestimmte Schwelle gegen die Aufzählung der Kosten.
* Platt stellt eine reine Logit-Verzerrung wieder her (Orakel der Rekalibrierung); Schwelle gleich τ-Quantil von `scipy.stats.bernoulli.ppf`.

Fließkomma-Vergleiche mit Toleranz (Platt-Koeffizienten 1e-6), Brüche exakt (DEMO-PLAYBOOK §5: Rechnungen mit scipy und BLAS hängen an der letzten Binärstelle)."""

import itertools
from fractions import Fraction as Fr

import numpy as np
import pytest
from scipy import special, stats

import ksw_core as K
import ksw_generator as G
import ksw_hand as H


def _pav_fractions(y):
    vals, wts, cnt = [], [], []
    for yi in y:
        vals.append(Fr(int(yi)))
        wts.append(Fr(1))
        cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
            vals[-1] = (vals[-1] * wts[-1] + v2 * w2) / (wts[-1] + w2)
            wts[-1] += w2
            cnt[-1] += c2
    out = []
    for v, c in zip(vals, cnt):
        out += [v] * c
    return out


def _pav_enumeration(y):
    """Orakel: alle 2^(n−1) Zerlegungen in aufeinanderfolgende Blöcke; nichtfallende Blockmittel; kleinste Quadratsumme (in Brüchen)."""
    n = len(y)
    best, fit = None, None
    for mask in range(1 << (n - 1)):
        blocks, start = [], 0
        for i in range(n - 1):
            if mask >> i & 1:
                blocks.append((start, i + 1))
                start = i + 1
        blocks.append((start, n))
        means = [sum(Fr(int(v)) for v in y[a:b]) / (b - a) for a, b in blocks]
        if any(means[i] > means[i + 1] for i in range(len(means) - 1)):
            continue
        sse = sum(sum((Fr(int(v)) - m) ** 2 for v in y[a:b]) for (a, b), m in zip(blocks, means))
        if best is None or sse < best:
            best, fit = sse, [m for (a, b), m in zip(blocks, means) for _ in range(b - a)]
    return fit


def test_pav_equals_enumeration_of_all_block_partitions_for_every_zero_one_sequence_of_length_7():
    for y in itertools.product((0, 1), repeat=7):
        exact = _pav_enumeration(y)
        assert _pav_fractions(y) == exact
        np.testing.assert_allclose(K.pav(list(y)), [float(v) for v in exact], atol=1e-12)


def test_pav_equals_scikit_learn_isotonic_regression_on_random_sequences():
    sk = pytest.importorskip("sklearn.isotonic")
    rng = np.random.default_rng(1)
    for _ in range(60):
        n = int(rng.integers(5, 80))
        y = rng.integers(0, 2, n).astype(float)
        ref = sk.IsotonicRegression().fit(np.arange(n, dtype=float), y).predict(np.arange(n, dtype=float))
        np.testing.assert_allclose(K.pav(y), ref, atol=1e-12)


def test_isotonic_prediction_equals_scikit_learn_with_clipping_and_ties():
    sk = pytest.importorskip("sklearn.isotonic")
    rng = np.random.default_rng(2)
    for _ in range(25):
        n = int(rng.integers(30, 300))
        f = np.round(rng.random(n), 2)                              # Gleichstände in der Vorhersage
        y = (rng.random(n) < f).astype(float)
        model = K.isotonic_fit(f, y)
        ref = sk.IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(f, y)
        grid = np.concatenate([np.linspace(-0.2, 1.2, 57), rng.random(30)])
        np.testing.assert_allclose(K.isotonic_apply(model, grid), ref.predict(grid), atol=1e-12)


def test_platt_newton_equals_scikit_learn_logistic_regression_without_penalty():
    sk = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(3)
    for _ in range(30):
        n = int(rng.integers(60, 400))
        z = rng.standard_normal(n) * 1.5
        y = (rng.random(n) < special.expit(-0.4 + 0.8 * z)).astype(float)
        a, b = K.fit_platt(special.expit(z), y)
        clf = sk.LogisticRegression(C=np.inf, solver="newton-cholesky", tol=1e-12, max_iter=200).fit(z[:, None], y)
        assert a == pytest.approx(clf.intercept_[0], abs=1e-6) and b == pytest.approx(clf.coef_[0, 0], abs=1e-6)


@pytest.mark.parametrize("c", [1e4, 1.0, 0.03])
def test_fit_logit_with_ridge_equals_scikit_learn(c):
    sk = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(4)
    for _ in range(8):
        n, p = 120, 6
        X = rng.standard_normal((n, p))
        y = (rng.random(n) < special.expit(X @ rng.standard_normal(p) * 0.8)).astype(float)
        beta = K.fit_logit(np.column_stack([np.ones(n), X]), y, penalty=1.0 / c)
        clf = sk.LogisticRegression(C=c, solver="newton-cholesky", tol=1e-12, max_iter=300).fit(X, y)
        np.testing.assert_allclose(beta, np.concatenate([clf.intercept_, clf.coef_[0]]), atol=2e-6)


def test_regret_formula_equals_direct_cost_difference():
    rng = np.random.default_rng(5)
    for _ in range(40):
        n = int(rng.integers(50, 2000))
        p = rng.beta(0.6, 1.8, n)
        score = np.clip(p + rng.normal(0, 0.1, n), 0, 1)
        r = float(rng.choice([1, 2, 3, 9, 19]))
        t = float(rng.random())
        assert K.regret_per_1000(score, p, r, t=t) == pytest.approx(K.regret_formula(score, p, r, t), abs=1e-9)
        assert K.regret_per_1000(score, p, r, t=t) >= -1e-9


def test_cost_threshold_is_the_minimum_over_a_threshold_grid_at_true_probability():
    p, _ = G.population(30_000, 7)
    grid = np.linspace(0.01, 0.99, 99)
    for r in (1, 3, 9, 19):
        costs = [K.expected_cost(p > t, p, r) for t in grid]
        assert abs(grid[int(np.argmin(costs))] - 1 / (1 + r)) <= 0.0101
        assert min(costs) >= K.expected_cost(p > K.threshold(r), p, r) - 1e-12


def test_brier_decomposition_equals_the_exact_fractions_of_the_hand_example(hand_groups):
    f, y = [], []
    for g in hand_groups:
        f += [float(g.p_hat)] * g.n
        y += [1.0] * g.events + [0.0] * (g.n - g.events)
    brier, rel, res, unc = K.brier_decomposition(np.array(f), np.array(y))
    exact = H.brier_decomposition(hand_groups)
    assert (brier, rel, res, unc) == pytest.approx(tuple(float(v) for v in exact), abs=1e-12)


def test_ece_equals_an_independent_loop_with_digitize():
    rng = np.random.default_rng(6)
    for nb in (5, 10, 20):
        for _ in range(10):
            n = int(rng.integers(100, 1500))
            f = rng.random(n)
            y = (rng.random(n) < f ** 1.3).astype(float)
            idx = np.digitize(f, np.linspace(0, 1, nb + 1)[1:-1])            # Fach 0 … nb − 1
            ref = sum(abs(f[idx == k].mean() - y[idx == k].mean()) * np.sum(idx == k) / n for k in range(nb) if np.any(idx == k))
            assert K.ece(f, y, f"{nb} gleich breit") == pytest.approx(ref, abs=1e-12)


def test_auc_equals_scikit_learn_also_with_ties():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(7)
    for _ in range(20):
        n = int(rng.integers(40, 500))
        f = np.round(rng.random(n), 1)
        y = (rng.random(n) < f).astype(float)
        if y.min() == y.max():
            continue
        assert K.auc(f, y) == pytest.approx(sk.roc_auc_score(y, f), abs=1e-12)


def test_tuned_threshold_equals_enumeration_of_the_realized_costs():
    rng = np.random.default_rng(8)
    cands = np.linspace(0.02, 0.98, 49)
    for _ in range(30):
        m = int(rng.integers(20, 600))
        pc = np.round(rng.random(m), 2)                                       # Gleichstände
        yc = (rng.random(m) < pc).astype(float)
        r = float(rng.choice([1, 3, 9, 19]))
        costs = [K.realized_cost(pc > t, yc, r) for t in cands]
        assert K.tuned_threshold(pc, yc, r, cands) == cands[int(np.argmin(costs))]


@pytest.mark.parametrize("a_, b_", [(0.5, 1.8), (-0.4, 0.5)])
def test_platt_restores_a_pure_logit_distortion(a_, b_):
    """p̂ = expit(a + b logit p) wird durch y ~ α + β logit p̂ mit β = 1/b, α = −a/b wiederhergestellt (Toleranz 0,05 bei 100 000 Containern)."""
    data = G.make_dataset(100_000, 7)
    f = special.expit(a_ + b_ * K.logit(data.p))
    alpha, beta = K.fit_platt(f, data.y)
    assert alpha == pytest.approx(-a_ / b_, abs=0.05) and beta == pytest.approx(1 / b_, abs=0.05)


def test_share_of_events_equals_mean_true_probability():
    data = G.make_dataset(100_000, 11)
    assert abs(data.y.mean() - data.p.mean()) < 0.01


def test_threshold_rule_equals_scipy_bernoulli_quantile_without_ties():
    count = 0
    for k in range(1, 40):
        for j in range(1, 40):
            p, tau = k / 40, j / 40
            if abs((1 - p) - tau) < 1e-12:
                continue                                                      # Gleichstand: Quantilkonvention, nicht die Kosten
            count += 1
            assert (stats.bernoulli.ppf(tau, p) == 1) == (p > 1 - tau)
    assert count == 1482
