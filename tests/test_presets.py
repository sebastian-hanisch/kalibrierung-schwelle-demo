"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; jedes Preset hat die erwartete Verdikt-Zeile (kein Preset ohne Wirkung meldet eine Wirkung)."""

import pytest

import ksw_constants as C
import ksw_evaluation as E
import ksw_frozen as FZ
import ksw_hand as H
import ksw_models as M
import ksw_presets as P

FZ_DATA = FZ.load()
REPS = E.reps(FZ_DATA)


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_control_ranges():
    for preset in C.PRESETS.values():
        assert preset["r"] in C.RATIOS and preset["model"] in M.MODEL_KEYS and preset["m"] in C.CAL_SIZES and 0 <= preset["seed"] <= C.SEED_MAX
        assert preset["bins"] in P.SETTING_SPECS["bins_select"].options


def test_default_preset_equals_the_default_settings():
    first = C.PRESETS[next(iter(C.PRESETS))]
    assert E.Settings(first["r"], first["model"], first["m"], first["seed"], first["bins"]) == E.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_presets_are_pairwise_different():
    seen = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(seen)) == len(seen)


def test_the_four_ratios_and_six_sizes_are_those_of_the_precomputed_measurements():
    assert list(C.RATIOS) == FZ_DATA["meta"]["ratios"] and list(C.CAL_SIZES) == FZ_DATA["meta"]["cal_sizes"]
    assert set(FZ_DATA["C"]) == set(M.MODEL_KEYS) == set(FZ_DATA["E"])


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_gives_a_verdict_for_each_key_question(name):
    pr = C.PRESETS[name]
    label = M.MODEL_LABELS[pr["model"]]
    st = E.c_stats(FZ_DATA, pr["model"], pr["r"])
    kind, text = E.verdict_threshold(pr["r"], label, st, 100.0, REPS)
    assert kind in ("info", "warning") and label in text and f"r = {pr['r']}" in text
    kind, text = E.verdict_recalibration(pr["r"], label, pr["m"], E.e_stats(FZ_DATA, pr["model"], pr["m"], pr["r"]), REPS)
    assert kind in ("success", "info", "warning") and f"m = {pr['m']}" in text


def test_preset_with_ratio_one_reports_no_difference_between_the_fixed_and_the_cost_threshold():
    """Ein Preset ohne Effekt darf keinen Effekt melden (DEMO-PLAYBOOK §11a Nr. 7): bei r = 1 sind feste Schwelle und Kostenschwelle dieselbe Regel."""
    st = E.c_stats(FZ_DATA, "haupt", 1)
    assert st["n_t05_gt_roh"] == 0 and st["t05"] == st["roh"]
    kind, text = E.verdict_threshold(1, "Haupteffekte", st, 0.0, REPS)
    assert kind == "info" and "keinen Unterschied" in text and "teurer" not in text


def test_hand_preset_ratio_three_reports_the_hand_costs():
    pr = C.PRESETS["Von Hand (30 Container)"]
    s = H.summary(H.DEFAULT_GROUPS, pr["r"])
    kind, text = E.verdict_hand(s, 30)
    assert kind == "success" and "roh (nach der Vorhersage) 11" in text and "feste Schwelle 1/2: 27" in text and "nach den beobachteten Häufigkeiten 9;" in text


def test_distorted_preset_models_show_the_slope_away_from_one():
    for key, low in (("verdichtet", False), ("ueberkonfident", True)):
        slope = E.good_stats(FZ_DATA, key)["slope"][0]
        assert (slope < 1) == low and abs(slope - 1) > 0.3
        assert E.verdict_calibration(M.MODEL_LABELS[key], E.good_stats(FZ_DATA, key), E.max_mean_gap(FZ_DATA, key), 5, 4, 7, 9, REPS)[0] == "warning"


def test_many_events_preset_versus_few_events_preset_differ_in_the_events_named():
    few = E.e_stats(FZ_DATA, "haupt", 100, 9)
    many = E.e_stats(FZ_DATA, "haupt", 2000, 9)
    assert few["events"] < 40 < 300 < many["events"]
