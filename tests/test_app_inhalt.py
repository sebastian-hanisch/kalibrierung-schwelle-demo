"""Inhalte der App (AppTest): Verdikte und Kennzahlen je Einstellung, Beispiel von Hand mit Eingaben, Permalink, Randfälle der Regler."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _app(**query) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in query.items():
        at.query_params[k] = v
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at, kind):
    return [x.value for x in getattr(at, kind)]


def _all_verdicts(at):
    return _texts(at, "success") + _texts(at, "info") + _texts(at, "warning")


def test_default_run_shows_three_key_questions_and_all_experiment_verdicts():
    at = _app()
    texts = _all_verdicts(at)
    assert any("die feste Schwelle 0,5 kostet im Mittel" in t for t in texts)                  # Kernfrage 1
    assert any("Steigung der Kalibrierung auf den Testdaten" in t for t in texts)              # Kernfrage 2
    assert any("Rekalibriermenge m = 2000" in t for t in texts)                                # Kernfrage 3
    assert any(t.startswith("r = 9 (Schwelle 1/10), 30 Container: roh") for t in texts)         # Experiment 1
    assert any("Wahres r = 9 (bei wahrem p" in t for t in texts)                               # Experiment 2
    assert any(t.startswith("ECE (10 gleich breit)") for t in texts)                           # Experiment 3
    assert any("Platt ist ab m = 1000" in t for t in texts)                                    # Experiment 4
    assert any("„ohne Containerart“ (AUC" in t for t in texts)                                 # Experiment 5


def test_ratio_one_reports_no_difference_between_the_fixed_and_the_cost_threshold():
    at = _app()
    at.sidebar.select_slider(key="r_select").set_value(1).run()
    assert not at.exception
    texts = _all_verdicts(at)
    assert any("keinen Unterschied" in t and "Haupteffekte" in t for t in texts)
    assert not any("0,5 ist teurer" in t for t in texts)


def test_ratio_three_gives_the_hand_costs_of_the_example():
    at = _app()
    at.sidebar.select_slider(key="r_select").set_value(3).run()
    assert any("roh (nach der Vorhersage) 11, feste Schwelle 1/2: 27, nach den beobachteten Häufigkeiten 9" in t for t in _all_verdicts(at))


def test_every_model_runs_and_names_itself_in_the_key_questions():
    for key in ("lognormal", "je_art", "geglaettet", "ohne_art", "verdichtet", "ueberkonfident", "verschoben"):
        at = _app()
        at.sidebar.selectbox(key="model_select").set_value(key).run()
        assert not at.exception, key
        assert any("Steigung der Kalibrierung" in t for t in _all_verdicts(at))


def test_small_recalibration_set_warns_about_few_events():
    at = _app()
    at.sidebar.select_slider(key="m_select").set_value(100).run()
    assert not at.exception
    assert any("Weniger als 200 Ereignisse" in w for w in _texts(at, "warning"))
    at.sidebar.select_slider(key="m_select").set_value(3000).run()
    assert not any("Weniger als 200 Ereignisse" in w for w in _texts(at, "warning"))


def test_equal_mass_bins_have_no_ece_floor_statement():
    at = _app()
    at.sidebar.selectbox(key="bins_select").set_value("10 gleiche Masse").run()
    assert not at.exception
    assert any("ECE-Boden nicht gemessen" in t for t in _texts(at, "info"))
    at.sidebar.selectbox(key="bins_select").set_value("5 gleich breit").run()
    assert any("ECE-Boden" in t and "5 gleich breit" in t for t in _texts(at, "info"))


def test_hand_example_inputs_change_the_costs_and_the_group_count_stays_three():
    at = _app()
    assert len(at.slider) == 3
    at.slider(key="hand_p2").set_value(60).run()
    assert not at.exception
    texts = _texts(at, "success") + _texts(at, "info") + _texts(at, "warning")
    assert any("30 Container" in t and "roh (nach der Vorhersage)" in t for t in texts)


def test_events_input_is_clamped_to_the_group_size():
    at = _app()
    at.number_input(key="hand_n2").set_value(5).run()          # Gruppe 3 hat 7 Ereignisse: auf 5 Container begrenzt
    assert not at.exception
    assert at.number_input(key="hand_e2").value == 5
    assert any("25 Container" in t for t in _all_verdicts(at))


def test_reset_button_restores_the_hand_example():
    at = _app()
    at.slider(key="hand_p0").set_value(80).run()
    next(b for b in at.button if "Beispiel von Hand wiederherstellen" in b.label).click().run()
    assert at.slider(key="hand_p0").value == 20 and not at.exception


def test_cost_matrix_expander_gives_the_threshold_and_rejects_degenerate_matrices():
    at = _app()
    assert any("Schwelle = **1/4**" in t for t in _texts(at, "info"))
    at.number_input(key="mat_c11").set_value(3).run()          # b − c11 = 3 − 3 = 0
    assert not at.exception and any("keine Schwelle" in w for w in _texts(at, "warning"))


def test_permalink_sets_the_controls_and_ignores_invalid_values():
    at = _app(r="19", modell="ohne_art", m="300", faecher="20 gleich breit", seed="777")
    assert at.sidebar.select_slider(key="r_select").value == 19
    assert at.sidebar.selectbox(key="model_select").value == "ohne_art"
    assert at.sidebar.select_slider(key="m_select").value == 300 and at.sidebar.number_input(key="seed_input").value == 777
    at = _app(r="4", modell="gibt_es_nicht", m="abc", seed="-5")
    assert at.sidebar.select_slider(key="r_select").value == 9 and at.sidebar.selectbox(key="model_select").value == "haupt"
    assert at.sidebar.select_slider(key="m_select").value == 2000 and at.sidebar.number_input(key="seed_input").value == 0     # Seed auf die Untergrenze begrenzt


def test_random_seed_button_changes_the_training_set_without_error():
    at = _app()
    before = at.sidebar.number_input(key="seed_input").value
    next(b for b in at.sidebar.button if "Neue Trainingsmenge" in b.label).click().run()
    assert not at.exception and at.sidebar.number_input(key="seed_input").value != before


def test_formula_expander_has_the_reading_aid_and_no_unescaped_percent():
    at = _app()
    expander = next(e for e in at.expander if e.label == "📐 Mathematische Formulierung")
    text = " ".join(m.value for m in expander.markdown)
    assert "| Symbol | Bedeutung |" in text and "\\frac{1}{1 + r}" in text and "%" not in text


@pytest.mark.parametrize("seed", [0, 99999])
def test_seed_range_ends(seed):
    at = _app(seed=str(seed))
    assert at.sidebar.number_input(key="seed_input").value == seed and not at.exception
