"""Messungen der Demo: 40 Trainingsmengen, 100 000 Testcontainer, Rekalibriermengen, ECE-Boden, Ridge-Stärke, Gegenprobe mit anderen Zufalls-Seeds.

Die Reihen sind rechenintensiv (Minuten) und werden von `tools/ksw_make_frozen.py` vorgerechnet und in `ksw_frozen.json` eingefroren; die App liest nur diese Datei.
`tests/test_frozen.py` rechnet je einen Ausschnitt neu und vergleicht ihn als Band mit der Datei.

Zähleinheiten: Entscheidungen (Container) für Kosten und ECE, Trainingsmengen (je 500 Container) für Mittel, Streuung und Zählungen, Wiederholungen für den ECE-Boden.
Kosten sind erwartete Kosten bei der wahren Wahrscheinlichkeit der Testcontainer (kein Rauschen der Ergebnisse); Regret je 1000 Entscheidungen."""

import numpy as np

import ksw_core as K
import ksw_generator as G
import ksw_models as M
import ksw_rng as R

RATIOS = (1, 3, 9, 19)
METHODS = ("roh", "t05", "platt", "iso", "tuned")        # roh = Schwelle 1/(1+r) auf p̂; t05 = feste 0,5; platt, iso = nach Rekalibrierung; tuned = auf den Kosten der Rekalibriermenge gewählt
GOODNESS = ("brier", "rmse", "auc", "ece5", "ece10", "ece20", "ece10m", "icpt", "slope", "mitt", "rel", "res")
N_TRAIN, N_CAL, N_TEST, REPS = 500, 2000, 100_000, 40
SEED_TRAIN, SEED_CAL, SEED_TEST = 500, 20_000, 999
SEED_TRAIN_F, SEED_CAL_F, SEED_TEST_F = 700, 40_000, 1999
SEED_CAL_E = 30_000
CAL_SIZES = (100, 300, 1000, 2000, 3000, 10_000)
ECE_FLOOR_SIZES = ((200, 200), (2000, 200), (20_000, 100), (100_000, 20))       # (Container, Wiederholungen)
RIDGE_C = (0.003, 0.01, 0.03, 0.1, 0.3)
RIDGE_REPS = 20
THRESHOLD_CANDIDATES = np.linspace(0.02, 0.98, 49)
FROZEN_MODELS_F = ("haupt", "geglaettet", "ohne_art")


def oracle_costs(test: G.Dataset) -> dict:
    return {r: K.expected_cost(test.p > K.threshold(r), test.p, r) for r in RATIOS}


def recalibrate(pf, pc, yc):
    """Rohe Vorhersage der Testmenge nach Platt und nach isotonischer Rekalibrierung auf der Rekalibriermenge (pc, yc). Gibt (Platt, isotonisch, (a, b))."""
    a, b = K.fit_platt(pc, yc)
    pp = pf if np.isnan(b) else K.platt_apply(pf, a, b)
    return pp, K.isotonic_apply(K.isotonic_fit(pc, yc), pf), (a, b)


def regrets(pf, pp, pi, pc, yc, p, r):
    """Regret je 1000 der fünf Regeln bei Verhältnis r; Gibt (Wörterbuch, gewählte Schwelle)."""
    ts = K.threshold(r)
    tb = K.tuned_threshold(pc, yc, r, THRESHOLD_CANDIDATES)
    return {"roh": K.regret_per_1000(pf, p, r), "t05": K.regret_per_1000(pf, p, r, t=0.5), "platt": K.regret_per_1000(pp, p, r),
            "iso": K.regret_per_1000(pi, p, r), "tuned": K.regret_per_1000(pf, p, r, t=tb)}, tb


def goodness(pf, test: G.Dataset, train_pred, train: G.Dataset) -> dict:
    a, b = K.fit_platt(pf, test.y)
    bz = K.brier_decomposition(pf, test.y)
    return {"brier": K.brier(pf, test.y), "rmse": float(np.sqrt(np.mean((pf - test.p) ** 2))), "auc": K.auc(pf, test.y),
            "ece5": K.ece(pf, test.y, "5 gleich breit"), "ece10": K.ece(pf, test.y, "10 gleich breit"), "ece20": K.ece(pf, test.y, "20 gleich breit"),
            "ece10m": K.ece(pf, test.y, "10 gleiche Masse"), "icpt": a, "slope": b,
            "mitt": float(np.mean(train_pred) - train.y.mean()), "rel": bz[1], "res": bz[2]}


def measure_rep(rep: int, test: G.Dataset, models=M.MODEL_KEYS, train_seed=SEED_TRAIN, cal_seed=SEED_CAL, n_train=N_TRAIN, n_cal=N_CAL):
    """Eine Trainingsmenge (Zufalls-Seed train_seed + rep) und eine Rekalibriermenge: Güte und Regret je Modell. Gibt {Modell: {"gute": …, "regret": {r: {…}}, "tuned_t": {r: t}}}."""
    train = G.make_dataset(n_train, train_seed + rep)
    cal = G.make_dataset(n_cal, cal_seed + rep)
    out = {}
    for key in models:
        pred = M.fit_model(key, train.X, train.d, train.y)
        pf = pred(test.X)
        pc = pred(cal.X)
        pp, pi, _ = recalibrate(pf, pc, cal.y)
        reg, tun = {}, {}
        for r in RATIOS:
            reg[r], tun[r] = regrets(pf, pp, pi, pc, cal.y, test.p, r)
        out[key] = {"gute": goodness(pf, test, pred(train.X), train), "regret": reg, "tuned_t": tun}
    return out


def measure_cal_sizes(rep: int, test: G.Dataset, models=M.MODEL_KEYS, sizes=CAL_SIZES):
    """Rekalibriermengen verschiedener Größe zu derselben Trainingsmenge (Zufalls-Seed 500 + rep). Gibt {Modell: {m: {"ereignisse": …, "regret": {r: {Regel: Wert}}}}}."""
    train = G.make_dataset(N_TRAIN, SEED_TRAIN + rep)
    out = {}
    for key in models:
        pred = M.fit_model(key, train.X, train.d, train.y)
        pf = pred(test.X)
        out[key] = {}
        for m in sizes:
            cal = G.make_dataset(m, SEED_CAL_E + 100 * rep + CAL_SIZES.index(m))        # Seed hängt an der Größe, nicht an der Stelle in `sizes`
            pc = pred(cal.X)
            pp, pi, _ = recalibrate(pf, pc, cal.y)
            out[key][m] = {"ereignisse": float(cal.y.sum()), "regret": {r: regrets(pf, pp, pi, pc, cal.y, test.p, r)[0] for r in RATIOS}}
    return out


def ece_floor(n: int, k: int):
    """ECE (5, 10, 20 gleich breite Fächer) der wahren Wahrscheinlichkeit auf n Containern mit eigenen Ergebnissen (Wiederholung k): perfekt kalibriert, trotzdem nicht 0."""
    seed = 3000 + 7 * k + n // 100
    p, _ = G.population(n, seed)
    y = (R.uniform_array(G.stream_seed(seed, 7), n) < p).astype(float)
    return K.ece(p, y, "5 gleich breit"), K.ece(p, y, "10 gleich breit"), K.ece(p, y, "20 gleich breit")


def ridge_row(c: float):
    """Modell „stark geglättet“ mit Ridge-Stärke C auf 20 Trainingsmengen: AUC, Steigung, ECE, Anteil Container mit p̂ > 1/10, Regret roh und Platt bei r = 9 (Mittelwerte)."""
    test = G.make_dataset(N_TEST, SEED_TEST)
    v = {k: [] for k in ("auc", "slope", "ece", "akt", "roh", "platt")}
    for rep in range(RIDGE_REPS):
        train = G.make_dataset(N_TRAIN, SEED_TRAIN + rep)
        cal = G.make_dataset(N_CAL, SEED_CAL + rep)
        pred = M.fit_smooth(train.X, train.y, c)
        pf, pc = pred(test.X), pred(cal.X)
        pp, _, _ = recalibrate(pf, pc, cal.y)
        v["auc"].append(K.auc(pf, test.y))
        v["slope"].append(K.fit_platt(pf, test.y)[1])
        v["ece"].append(K.ece(pf, test.y))
        v["akt"].append(float(np.mean(pf > 0.1)))
        v["roh"].append(K.regret_per_1000(pf, test.p, 9))
        v["platt"].append(K.regret_per_1000(pp, test.p, 9))
    return {"C": c, **{k: float(np.mean(x)) for k, x in v.items()}, "roh_sd": float(np.std(v["roh"]))}
