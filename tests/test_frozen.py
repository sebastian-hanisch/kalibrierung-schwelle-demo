"""Vorgerechnete Messungen (`ksw_frozen.json`): Aufbau, Zählungen aus den Einzelwerten, und der Abgleich eines Ausschnitts mit einer Neuberechnung (als Band).

Die Datei wird von `tools/ksw_make_frozen.py` erzeugt; `ksw_frozen.verify_subset` rechnet Trainingsmenge 0 (alle acht Modelle), die Rekalibriermengen zu Trainingsmenge 0 (zwei Modelle,
zwei Größen) und je drei Wiederholungen des ECE-Bodens neu und vergleicht mit der Datei (relative Toleranz 1e-6)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import ksw_constants as C
import ksw_evaluation as E
import ksw_frozen as FZ
import ksw_measure as MS
import ksw_models as M

DATA = FZ.load()
ROOT = Path(__file__).resolve().parent.parent


def test_structure_of_the_frozen_file():
    meta = DATA["meta"]
    assert meta["reps"] == 40 and meta["n_train"] == 500 and meta["n_cal"] == 2000 and meta["n_test"] == 100_000
    assert set(DATA) == {"meta", "oracle_cost_per_1000", "C", "D", "E", "F", "ridge"}
    for key in M.MODEL_KEYS:
        assert set(DATA["C"][key]["gute"]) == set(MS.GOODNESS)
        for r in C.RATIOS:
            assert set(DATA["C"][key]["regret"][str(r)]) == set(MS.METHODS)
            assert all(len(v) == 40 for v in DATA["C"][key]["regret"][str(r)].values())
        for m in C.CAL_SIZES:
            assert len(DATA["E"][key][str(m)]["ereignisse"]) == 40
    assert [row["n"] for row in DATA["D"]] == [200, 2000, 20_000, 100_000] and [row["reps"] for row in DATA["D"]] == [200, 200, 100, 20]
    assert len(DATA["ridge"]) == 5 and set(DATA["F"]["modelle"]) == set(MS.FROZEN_MODELS_F)


def test_regret_is_never_negative_in_any_measurement():
    for key in M.MODEL_KEYS:
        for r in C.RATIOS:
            for meth, values in DATA["C"][key]["regret"][str(r)].items():
                assert min(values) > -1e-6, (key, r, meth)
        for m in C.CAL_SIZES:
            for r in C.RATIOS:
                for meth, values in DATA["E"][key][str(m)]["regret"][str(r)].items():
                    assert min(values) > -1e-4, (key, m, r, meth)


def test_with_ratio_one_the_raw_rule_and_the_fixed_threshold_are_identical_in_every_set_and_model():
    for key in M.MODEL_KEYS:
        r1 = DATA["C"][key]["regret"]["1"]
        assert r1["roh"] == r1["t05"]


def test_counts_equal_a_direct_count_over_the_single_values():
    st = E.c_stats(DATA, "haupt", 9)
    roh, t05 = FZ.series_c(DATA, "haupt", "roh", 9), FZ.series_c(DATA, "haupt", "t05", 9)
    assert st["n_t05_gt_roh"] == sum(1 for a, b in zip(t05, roh) if a > b) and st["reps"] == 40
    assert st["roh"] == (float(np.mean(roh)), float(np.std(roh)))


def test_mean_calibration_in_the_training_set_for_the_logit_models():
    for key in ("haupt", "ohne_art", "geglaettet", "je_art"):
        assert E.max_mean_gap(DATA, key) < 1e-7


def test_oracle_costs_are_increasing_with_the_ratio_and_the_test_set_is_the_stated_one():
    costs = [DATA["oracle_cost_per_1000"][str(r)] for r in C.RATIOS]
    assert costs == sorted(costs) and abs(DATA["meta"]["test_mean_p"] - DATA["meta"]["test_event_share"]) < 0.005


def test_the_file_matches_the_code_as_a_band():
    """Neuberechnung eines Ausschnitts: Datei und Code dürfen sich nur im Rahmen der Plattformunterschiede unterscheiden (relative Toleranz 1e-6)."""
    assert FZ.verify_subset(DATA) == []


def test_the_make_frozen_tool_checks_the_file_from_the_command_line():
    out = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "tools" / "ksw_make_frozen.py"), "--check"], capture_output=True, text=True, cwd=ROOT, stdin=subprocess.DEVNULL, timeout=300)
    assert out.returncode == 0 and "Abgleich ok" in out.stdout


def test_the_file_is_valid_json_with_finite_numbers():
    text = (ROOT / "ksw_frozen.json").read_text(encoding="utf-8")
    json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))        # NaN und Infinity wären hier ein Fehler
