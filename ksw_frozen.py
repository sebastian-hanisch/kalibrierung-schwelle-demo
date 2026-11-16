"""Vorgerechnete Messungen: Laden der eingefrorenen Datei `ksw_frozen.json`, Neuberechnung (`build`) und der Abgleich eines Ausschnitts (`verify_subset`).

Die Datei enthält je Messung die Werte **je Trainingsmenge** (nicht nur Mittel), damit App und Tests Mittel, Streuung und Zählungen („in k von 40 Trainingsmengen“)
aus denselben Zahlen bilden. Erzeugt mit `python tools/ksw_make_frozen.py` (Minuten); `python tools/ksw_make_frozen.py --check` rechnet einen Ausschnitt neu und vergleicht."""

import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

import ksw_generator as G
import ksw_measure as MS
import ksw_models as M

PATH = Path(__file__).with_name("ksw_frozen.json")
VERSION = 1


def _round(x, nd=9):
    return round(float(x), nd)


def _arr(values, nd=9):
    return [_round(v, nd) for v in values]


@lru_cache(maxsize=1)
def load() -> dict:
    with open(PATH, encoding="utf-8") as fh:
        return json.load(fh)


def series_c(frozen: dict, model: str, method: str, r: int) -> np.ndarray:
    return np.array(frozen["C"][model]["regret"][str(r)][method])


def series_e(frozen: dict, model: str, m: int, method: str, r: int) -> np.ndarray:
    return np.array(frozen["E"][model][str(m)]["regret"][str(r)][method])


def build(progress=print) -> dict:
    """Rechnet alle Messungen (Minuten). Zufall nur über den SplitMix64 aus ksw_rng.py."""
    t0 = time.time()
    keys = M.MODEL_KEYS
    test = G.make_dataset(MS.N_TEST, MS.SEED_TEST)
    orc = MS.oracle_costs(test)
    frozen = {"meta": {"version": VERSION, "reps": MS.REPS, "n_train": MS.N_TRAIN, "n_cal": MS.N_CAL, "n_test": MS.N_TEST, "ratios": list(MS.RATIOS),
                       "cal_sizes": list(MS.CAL_SIZES), "seeds": {"train": MS.SEED_TRAIN, "cal": MS.SEED_CAL, "test": MS.SEED_TEST, "train_f": MS.SEED_TRAIN_F,
                                                                  "cal_f": MS.SEED_CAL_F, "test_f": MS.SEED_TEST_F, "cal_e": MS.SEED_CAL_E},
                       "test_event_share": _round(test.y.mean(), 6), "test_mean_p": _round(test.p.mean(), 6), "test_auc_true_p": _round(MS.K.auc(test.p, test.y), 6)},
              "oracle_cost_per_1000": {str(r): _round(1000 * c, 6) for r, c in orc.items()}}
    # C: Modelle auf Standgeld
    reps = [MS.measure_rep(rep, test) for rep in range(MS.REPS)]
    progress(f"C fertig ({time.time() - t0:.0f} s)")
    frozen["C"] = {}
    for k in keys:
        frozen["C"][k] = {
            "gute": {g: ([float(rp[k]["gute"][g]) for rp in reps] if g == "mitt" else _arr([rp[k]["gute"][g] for rp in reps])) for g in MS.GOODNESS},
            "regret": {str(r): {meth: _arr([rp[k]["regret"][r][meth] for rp in reps]) for meth in MS.METHODS} for r in MS.RATIOS},
            "tuned_t": {str(r): _arr([rp[k]["tuned_t"][r] for rp in reps]) for r in MS.RATIOS},
        }
    # D: ECE des wahren p
    frozen["D"] = []
    for n, nrep in MS.ECE_FLOOR_SIZES:
        vals = np.array([MS.ece_floor(n, k) for k in range(nrep)])
        frozen["D"].append({"n": n, "reps": nrep, "ece5": _arr(vals[:, 0]), "ece10": _arr(vals[:, 1]), "ece20": _arr(vals[:, 2])})
    progress(f"D fertig ({time.time() - t0:.0f} s)")
    # E: Größe der Rekalibriermenge
    erep = [MS.measure_cal_sizes(rep, test) for rep in range(MS.REPS)]
    frozen["E"] = {}
    for k in keys:
        frozen["E"][k] = {}
        for m in MS.CAL_SIZES:
            frozen["E"][k][str(m)] = {
                "ereignisse": _arr([rp[k][m]["ereignisse"] for rp in erep]),
                "regret": {str(r): {meth: _arr([rp[k][m]["regret"][r][meth] for rp in erep], 7) for meth in MS.METHODS} for r in MS.RATIOS},
            }
    progress(f"E fertig ({time.time() - t0:.0f} s)")
    # F: Gegenprobe mit anderen Zufalls-Seeds
    test_f = G.make_dataset(MS.N_TEST, MS.SEED_TEST_F)
    orc_f = MS.oracle_costs(test_f)
    frep = [MS.measure_rep(rep, test_f, models=MS.FROZEN_MODELS_F, train_seed=MS.SEED_TRAIN_F, cal_seed=MS.SEED_CAL_F) for rep in range(MS.REPS)]
    frozen["F"] = {"oracle_cost_per_1000": {str(r): _round(1000 * c, 6) for r, c in orc_f.items()}, "modelle": {}}
    for k in MS.FROZEN_MODELS_F:
        frozen["F"]["modelle"][k] = {str(r): {meth: _arr([rp[k]["regret"][r][meth] for rp in frep]) for meth in MS.METHODS} for r in MS.RATIOS}
    progress(f"F fertig ({time.time() - t0:.0f} s)")
    # Ridge-Stärke
    frozen["ridge"] = [MS.ridge_row(c) for c in MS.RIDGE_C]
    progress(f"Ridge fertig ({time.time() - t0:.0f} s)")
    frozen["meta"]["laufzeit_s"] = int(time.time() - t0)
    return frozen


def write(frozen: dict, path: Path = PATH) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(frozen, fh, ensure_ascii=False, separators=(",", ":"))
        fh.write("\n")


def verify_subset(frozen: dict, rtol: float = 1e-6) -> list[str]:
    """Rechnet Trainingsmenge 0 (Messung C, alle acht Modelle), Rekalibriermengen zu Trainingsmenge 0 (Messung E, zwei Modelle) und die ersten Wiederholungen des ECE-Bodens neu.
    Gibt die Abweichungen (leer = Datei passt zum Code) zurück; verglichen wird als Band (relative Toleranz, kleine absolute Toleranz)."""
    bad = []

    def near(a, b, label, atol=1e-6):
        if not (abs(a - b) <= atol + rtol * abs(b)):
            bad.append(f"{label}: neu {a!r}, Datei {b!r}")

    test = G.make_dataset(MS.N_TEST, MS.SEED_TEST)
    rp = MS.measure_rep(0, test)
    for k in M.MODEL_KEYS:
        for g in MS.GOODNESS:
            near(rp[k]["gute"][g], frozen["C"][k]["gute"][g][0], f"C {k} {g}", atol=1e-6)
        for r in MS.RATIOS:
            for meth in MS.METHODS:
                near(rp[k]["regret"][r][meth], frozen["C"][k]["regret"][str(r)][meth][0], f"C {k} r={r} {meth}", atol=1e-6)
    er = MS.measure_cal_sizes(0, test, models=("haupt", "ohne_art"), sizes=(100, 1000))
    for k in ("haupt", "ohne_art"):
        for m in (100, 1000):
            for r in MS.RATIOS:
                for meth in MS.METHODS:
                    near(er[k][m]["regret"][r][meth], frozen["E"][k][str(m)]["regret"][str(r)][meth][0], f"E {k} m={m} r={r} {meth}", atol=1e-5)
    for row in frozen["D"][:2]:
        for k in range(3):
            e5, e10, e20 = MS.ece_floor(row["n"], k)
            near(e10, row["ece10"][k], f"D n={row['n']} Wdh. {k}")
    return bad
