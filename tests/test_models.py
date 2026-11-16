"""Die acht Modelle (`ksw_models.py`): Wertebereich, Gleichungen der Anpassung, Randfälle der Trainingsmenge."""

import numpy as np
import pytest
from scipy import special

import ksw_core as K
import ksw_generator as G
import ksw_models as M


@pytest.fixture(scope="module")
def data():
    return G.make_dataset(500, 500), G.make_dataset(4000, 999)


@pytest.mark.parametrize("key", M.MODEL_KEYS)
def test_every_model_returns_probabilities_inside_the_clip_range(data, key):
    train, test = data
    p = M.fit_model(key, train.X, train.d, train.y)(test.X)
    assert p.shape == (len(test.y),) and p.min() >= M.CLIP and p.max() <= 1 - M.CLIP and np.all(np.isfinite(p))


@pytest.mark.parametrize("key", ["haupt", "ohne_art", "geglaettet", "je_art"])
def test_logit_models_are_calibrated_in_the_mean_on_the_training_set(key):
    """Score-Gleichung des Logit mit unbestraftem Achsenabschnitt: Σ (y − p̂) = 0, auch mit Ridge; für „je Art“ gilt sie je Art (also auch im Ganzen)."""
    for seed in range(500, 505):
        train = G.make_dataset(500, seed)
        p = M.fit_model(key, train.X, train.d, train.y)(train.X)
        assert abs(p.mean() - train.y.mean()) < 1e-7


def test_distorted_models_are_the_logit_transform_of_the_main_effects(data):
    train, test = data
    base = M.fit_model("haupt", train.X, train.d, train.y)(test.X)
    for key, (a, b) in M.DISTORTIONS.items():
        got = M.fit_model(key, train.X, train.d, train.y)(test.X)
        np.testing.assert_allclose(got, np.clip(special.expit(a + b * K.logit(base)), M.CLIP, 1 - M.CLIP), atol=1e-12)


def test_the_ridge_model_is_flatter_than_the_main_effects(data):
    """Starke Strafe drückt die Koeffizienten: die Vorhersagen streuen weniger als bei den Haupteffekten (Standardabweichung der Logits)."""
    train, test = data
    flat = K.logit(M.fit_model("geglaettet", train.X, train.d, train.y)(test.X)).std()
    free = K.logit(M.fit_model("haupt", train.X, train.d, train.y)(test.X)).std()
    assert flat < free


def test_the_model_without_art_cannot_separate_the_container_types(data):
    train, test = data
    with_art = K.auc(M.fit_model("haupt", train.X, train.d, train.y)(test.X), test.y)
    without = K.auc(M.fit_model("ohne_art", train.X, train.d, train.y)(test.X), test.y)
    assert without < with_art - 0.1


def test_lognormal_model_recovers_the_true_probability_with_a_large_training_set():
    """Die richtige Familie: bei 40 000 Trainingscontainern liegt die Vorhersage dicht an der wahren Wahrscheinlichkeit (mittlere Abweichung unter 0,03)."""
    train, test = G.make_dataset(40_000, 31), G.make_dataset(3000, 999)
    p = M.fit_model("lognormal", train.X, train.d, train.y)(test.X)
    assert np.sqrt(np.mean((p - test.p) ** 2)) < 0.03


def test_per_art_logit_handles_a_group_without_any_event():
    """Hat eine Containerart im Training kein Standgeld, bekommt sie ihren Anteil (0) statt eines Fehlers; die Vorhersage wird auf 1e-9 gekappt."""
    train = G.make_dataset(500, 500)
    y = train.y.copy()
    y[train.X[:, 0] == 3] = 0.0
    p = M.fit_model("je_art", train.X, train.d, y)(G.draw_features(300, 4))
    assert np.all(np.isfinite(p)) and p[G.draw_features(300, 4)[:, 0] == 3].max() == pytest.approx(M.CLIP)


def test_small_training_sets_do_not_break_the_fit():
    """Weniger Container als Spalten (30 Container, 19 Spalten): der Logit mit minimaler Strafe bleibt bestimmt und liefert endliche Werte."""
    train = G.make_dataset(30, 3)
    for key in M.MODEL_KEYS:
        p = M.fit_model(key, train.X, train.d, train.y)(G.draw_features(50, 5))
        assert np.all(np.isfinite(p)) and p.min() >= M.CLIP


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        M.fit_model("gibt_es_nicht", np.zeros((3, 6)), np.ones(3), np.zeros(3))


def test_every_model_has_a_label():
    assert set(M.MODEL_LABELS) == set(M.MODEL_KEYS) and len(M.MODEL_KEYS) == 8
