"""Die acht Modelle für Standgeld: fünf angepasste Modelle und drei kontrolliert verzerrte Fassungen desselben Modells.

* Haupteffekte: Logit auf allen 18 Merkmalsspalten (Art, Zoll, Wochenende, log(1 + Vorlauf), Gewicht, 11 Kunden), nahezu unbestraft (λ = 1/C = 1e-4).
* lognormal je Art: die richtige Familie; je Containerart OLS auf die log-Verweilzeit, Streuung je Art, p = 1 − Φ((ln 7 − μ̂) / ŝ). Braucht die Verweilzeit, nicht nur das Standgeld.
* je Art: je Containerart ein eigenes Logit auf Zoll, Wochenende, log(1 + Vorlauf), Gewicht und Kunden (Wechselwirkung Art × alles, bei n = 500 überangepasst).
* stark geglättet: Haupteffekte mit Ridge auf standardisierten Spalten, λ = 1/C = 1/0,03.
* ohne Art: Haupteffekte ohne die drei Art-Spalten (die Rangfolge fehlt zum Teil).
* verdichtet / überkonfident / verschoben: Haupteffekte, danach p̃ = expit(a + b · logit p̂) mit (a; b) = (0; 0,5) / (0; 2) / (1; 1): die Rangfolge bleibt, nur die Kalibrierung ist verzerrt.

Jedes Modell wird mit `fit_model(key, X, d, y)` angepasst und liefert eine Funktion Merkmale → p̂ (auf 1e-9 bis 1 − 1e-9 gekappt)."""

import numpy as np
from scipy import special

import ksw_core as K
import ksw_generator as G

MODEL_KEYS = ("haupt", "lognormal", "je_art", "geglaettet", "ohne_art", "verdichtet", "ueberkonfident", "verschoben")
MODEL_LABELS = {
    "haupt": "Haupteffekte (Logit, 18 Spalten)",
    "lognormal": "lognormal je Art (richtige Familie)",
    "je_art": "je Art (Wechselwirkung, überangepasst)",
    "geglaettet": "Ridge „stark geglättet“",
    "ohne_art": "ohne Containerart (Rangfolge fehlt)",
    "verdichtet": "Haupteffekte „verdichtet“ (Steigung 0,5)",
    "ueberkonfident": "Haupteffekte „überkonfident“ (Steigung 2)",
    "verschoben": "Haupteffekte „verschoben“ (Achsenabschnitt +1)",
}
DISTORTIONS = {"verdichtet": (0.0, 0.5), "ueberkonfident": (0.0, 2.0), "verschoben": (1.0, 1.0)}   # (a, b) in p̃ = expit(a + b logit p̂)
C_NEAR_UNPENALIZED = 1e4
C_SMOOTH = 0.03
PENALTY_FREE = 1.0 / C_NEAR_UNPENALIZED
CLIP = 1e-9


def _clip(p):
    return np.clip(p, CLIP, 1.0 - CLIP)


def _fit_main(X, y, columns=None, penalty=PENALTY_FREE):
    F = G.design(X)
    cols = list(range(F.shape[1])) if columns is None else columns
    beta = K.fit_logit(F[:, cols], y, penalty=penalty)
    return lambda Xn: _clip(special.expit(G.design(Xn)[:, cols] @ beta))


def fit_smooth(X, y, c=C_SMOOTH):
    F = G.design(X)[:, 1:]
    mu, sd = F.mean(0), F.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    beta = K.fit_logit(np.column_stack([np.ones(len(F)), (F - mu) / sd]), y, penalty=1.0 / c)
    return lambda Xn: _clip(special.expit(beta[0] + ((G.design(Xn)[:, 1:] - mu) / sd) @ beta[1:]))


def _fit_per_art(X, y):
    art = X[:, 0].astype(int)
    F = G.design(X)[:, 4:]                      # Zoll, Wochenende, log(1 + Vorlauf), Gewicht, 11 Kunden
    fits = {}
    for a in range(4):
        m = art == a
        if m.sum() == 0 or y[m].min() == y[m].max():
            fits[a] = float(y[m].mean()) if m.sum() else float(y.mean())     # Gruppe ohne Ereignis oder ohne Nichtereignis: Anteil
        else:
            fits[a] = K.fit_logit(np.column_stack([np.ones(m.sum()), F[m]]), y[m], penalty=PENALTY_FREE)

    def predict(Xn):
        out = np.zeros(len(Xn))
        Fn = G.design(Xn)[:, 4:]
        an = Xn[:, 0].astype(int)
        for a in range(4):
            m = an == a
            if not m.any():
                continue
            out[m] = fits[a] if isinstance(fits[a], float) else special.expit(np.column_stack([np.ones(m.sum()), Fn[m]]) @ fits[a])
        return _clip(out)
    return predict


def _fit_lognormal(X, d):
    art = X[:, 0].astype(int)
    F = G.design(X)[:, 4:]
    par = {}
    for a in range(4):
        m = art == a
        Fa = F[m]
        keep = [j for j in range(Fa.shape[1]) if m.any() and np.std(Fa[:, j]) > 0]
        A = np.column_stack([np.ones(m.sum()), Fa[:, keep]])
        ly = np.log(d[m])
        beta = np.linalg.lstsq(A, ly, rcond=None)[0]
        dof = max(len(ly) - A.shape[1], 1)
        par[a] = (keep, beta, float(np.sqrt(np.sum((ly - A @ beta) ** 2) / dof)))

    def predict(Xn):
        out = np.zeros(len(Xn))
        Fn = G.design(Xn)[:, 4:]
        an = Xn[:, 0].astype(int)
        for a in range(4):
            m = an == a
            if not m.any():
                continue
            keep, beta, s = par[a]
            mu = np.column_stack([np.ones(m.sum()), Fn[m][:, keep]]) @ beta
            out[m] = special.ndtr((mu - G.LN_STAND) / max(s, 1e-6))
        return _clip(out)
    return predict


def fit_model(key: str, X, d, y):
    """Passt das Modell `key` an Trainingsdaten (Merkmale X, Verweilzeit d, Standgeld y) an und gibt die Vorhersagefunktion zurück."""
    if key == "haupt":
        return _fit_main(X, y)
    if key == "ohne_art":
        return _fit_main(X, y, columns=[j for j in range(19) if j not in (1, 2, 3)])
    if key == "geglaettet":
        return fit_smooth(X, y)
    if key == "je_art":
        return _fit_per_art(X, y)
    if key == "lognormal":
        return _fit_lognormal(X, d)
    if key in DISTORTIONS:
        a, b = DISTORTIONS[key]
        base = _fit_main(X, y)
        return lambda Xn: _clip(special.expit(a + b * K.logit(base(Xn))))
    raise ValueError(f"unbekanntes Modell: {key}")
