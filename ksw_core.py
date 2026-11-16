"""Kern: Logit (Newton), Platt-Skalierung, isotonische Rekalibrierung (PAV), Kosten und Regret einer Schwelle, Zuverlässigkeitsdiagramm, ECE, Brier-Zerlegung, AUC.

Nur numpy und scipy; jede Einheit ist eine eigene Funktion mit expliziten Ein- und Ausgaben. Die Orakel-Tests (`tests/test_oracle_kalibrierung_schwelle.py`) rechnen jede Einheit
auf einem anderen Weg nach (Brüche, Aufzählung, scikit-learn, Formel).

Konventionen (stehen auch in der README):
* Kosten: Maßnahme bei Nichtereignis (Fehlalarm) kostet 1, keine Maßnahme bei Ereignis (verpasst) kostet r; richtig entschieden kostet 0. r = Kosten verpasst : Kosten unnötig.
* Schwelle 1/(1+r); Maßnahme bei p̂ > Schwelle (bei Gleichstand kosten beide Entscheidungen im Erwartungswert dasselbe).
* Regret = erwartete Kosten bei wahrem p der Regel minus erwartete Kosten der Orakelregel (wahres p, Schwelle 1/(1+r)), je 1000 Entscheidungen; nie negativ.
* ECE = Σ (Anteil im Fach) · |mittleres p̂ − beobachtete Häufigkeit| im Fach; Fach k = ⌊p̂ · B + 1e-9⌋ (links geschlossen, die Randregel steht im Test in Brüchen).
* Brier = Mittel von (p̂ − y)²; Murphy-Zerlegung Brier = REL − RES + UNC auf zwei Stellen gerundeten Vorhersagen.
* Achsenabschnitt und Steigung = Zwei-Parameter-Anpassung y ~ a + b · logit(p̂) (Platt ohne geglättete Zielwerte)."""

import numpy as np
from scipy import special, stats

EPS_P = 1e-9      # Kappung von p̂ vor dem logit
BINNINGS = ("5 gleich breit", "10 gleich breit", "20 gleich breit", "10 gleiche Masse")


def logit(p):
    p = np.clip(p, EPS_P, 1.0 - EPS_P)
    return np.log(p / (1.0 - p))


def threshold(r: float) -> float:
    """Kostenschwelle 1/(1 + r); r ≤ 0 ist keine gültige Kostenrelation."""
    if not r > 0:
        raise ValueError("Das Kostenverhältnis r muss größer als 0 sein.")
    return 1.0 / (1.0 + r)


# --- Logit (Newton mit Schritthalbierung) -------------------------------------------------------------------------------------------------

def fit_logit(F: np.ndarray, y: np.ndarray, penalty=1e-4, max_iter: int = 200, tol: float = 1e-10) -> np.ndarray:
    """Logistische Regression: minimiert Σ [log(1 + e^η) − y η] + ½ Σ_j λ_j β_j². `F` enthält den Achsenabschnitt als erste Spalte (unbestraft bei `penalty` als Zahl);
    `penalty` ist eine Zahl (gilt für alle Spalten außer der ersten) oder ein Feld je Spalte. Entspricht scikit-learn `LogisticRegression(C=1/λ)`."""
    F = np.asarray(F, float)
    y = np.asarray(y, float)
    if np.ndim(penalty) == 0:
        lam = np.full(F.shape[1], float(penalty))
        lam[0] = 0.0
    else:
        lam = np.asarray(penalty, float)

    def nll(b):
        eta = F @ b
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta) + 0.5 * np.sum(lam * b * b))

    beta = np.zeros(F.shape[1])
    f0 = nll(beta)
    for _ in range(max_iter):
        mu = special.expit(F @ beta)
        grad = F.T @ (mu - y) + lam * beta
        H = (F.T * (mu * (1.0 - mu))) @ F + np.diag(lam)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H + 1e-8 * np.eye(len(beta)), grad, rcond=None)[0]
        t = 1.0
        while t > 1e-10:
            f1 = nll(beta - t * step)
            if f1 <= f0 + 1e-14 * max(1.0, abs(f0)):
                break
            t /= 2.0
        else:
            break
        beta = beta - t * step
        f0 = f1
        if np.max(np.abs(t * step)) < tol:
            break
    return beta


def fit_platt(f, y, ridge: float = 1e-8):
    """Platt-Skalierung: y ~ a + b · logit(p̂), zwei Parameter, derselbe Newton-Code wie das Logit. Gibt (a, b); ist p̂ (fast) konstant, ist b nicht bestimmt (NaN)."""
    z = logit(np.asarray(f, float))
    if len(z) == 0 or np.ptp(z) < 1e-9:
        return float("nan"), float("nan")
    beta = fit_logit(np.column_stack([np.ones(len(z)), z]), y, penalty=np.array([ridge, ridge]))
    return float(beta[0]), float(beta[1])


def platt_apply(f, a: float, b: float):
    return special.expit(a + b * logit(np.asarray(f, float)))


# --- Isotonische Rekalibrierung (PAV) -------------------------------------------------------------------------------------------------------

def pav(values, weights=None) -> np.ndarray:
    """Pool Adjacent Violators: nichtfallende Anpassung der (gewichteten) Mittelwerte in der gegebenen Reihenfolge (bereits nach der Vorhersage sortiert)."""
    n = len(values)
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    vals, wts, cnt = [], [], []
    for i in range(n):
        vals.append(float(values[i]))
        wts.append(float(w[i]))
        cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
            vals[-1] = (vals[-1] * wts[-1] + v2 * w2) / (wts[-1] + w2)
            wts[-1] += w2
            cnt[-1] += c2
    out = np.empty(n)
    k = 0
    for v, c in zip(vals, cnt):
        out[k:k + c] = v
        k += c
    return out


def isotonic_fit(f, y):
    """Isotonische Regression y ~ f: gleiche Vorhersagen werden gewichtet zusammengefasst, danach PAV. Gibt (Stützstellen, angepasste Werte)."""
    f = np.asarray(f, float)
    y = np.asarray(y, float)
    order = np.argsort(f, kind="stable")
    fs, ys = f[order], y[order]
    ux, start = np.unique(fs, return_index=True)
    cnt = np.diff(np.append(start, len(fs))).astype(float)
    mean = np.add.reduceat(ys, start) / cnt
    return ux, pav(mean, cnt)


def isotonic_apply(model, f):
    """Lineare Interpolation zwischen den Stützstellen, außerhalb gekappt (wie scikit-learn `IsotonicRegression(out_of_bounds="clip")`)."""
    ux, fit = model
    return np.interp(np.asarray(f, float), ux, fit)


# --- Kosten, Regret, Schwellenwahl ----------------------------------------------------------------------------------------------------------

def expected_cost(act, p, r: float) -> float:
    """Erwartete Kosten je Entscheidung bei wahrer Wahrscheinlichkeit p: Maßnahme (act = True) kostet 1 − p, Unterlassung r · p."""
    return float(np.mean(np.where(act, 1.0 - p, r * p)))


def realized_cost(act, y, r: float) -> float:
    """Kosten je Entscheidung auf beobachteten Ergebnissen."""
    return float(np.mean(np.where(act, 1.0 - y, r * y)))


def regret_per_1000(score, p, r: float, t: float | None = None) -> float:
    """Regret der Regel „Maßnahme bei score > t“ (Standard t = 1/(1+r)) gegen die Orakelregel, je 1000 Entscheidungen."""
    t = threshold(r) if t is None else t
    return 1000.0 * (expected_cost(score > t, p, r) - expected_cost(p > threshold(r), p, r))


def regret_formula(score, p, r: float, t: float) -> float:
    """Zweiter Rechenweg: Regret = Mittel von |(1 + r) p − 1| über die Container, bei denen die Regel anders entscheidet als die Orakelregel (je 1000)."""
    anders = (score > t) != (p > threshold(r))
    return 1000.0 * float(np.mean(np.abs((1.0 + r) * p - 1.0) * anders))


def tuned_threshold(pc, yc, r: float, candidates) -> float:
    """Schwelle aus der Kandidatenliste mit den kleinsten Kosten auf den beobachteten Ergebnissen der Rekalibriermenge (erster Treffer bei Gleichstand)."""
    pc = np.asarray(pc, float)
    yc = np.asarray(yc, float)
    order = np.argsort(pc, kind="stable")
    ps, ys = pc[order], yc[order]
    cum = np.cumsum(ys)
    cands = np.asarray(candidates, float)
    k = np.searchsorted(ps, cands, side="right")           # Zahl der Container mit p̂ ≤ t
    ev_le = np.where(k > 0, cum[np.maximum(k - 1, 0)], 0.0)
    n_gt = len(ps) - k
    ev_gt = cum[-1] - ev_le
    cost = (n_gt - ev_gt) + r * ev_le                       # Fehlalarme unter p̂ > t plus verpasste Ereignisse unter p̂ ≤ t
    return float(cands[int(np.argmin(cost))])


# --- Güte der Wahrscheinlichkeiten -----------------------------------------------------------------------------------------------------------

def _bin_index(f, nb: int):
    return np.minimum(np.floor(np.asarray(f, float) * nb + 1e-9).astype(int), nb - 1)


def reliability_bins(f, y, binning: str = "10 gleich breit"):
    """Zuverlässigkeitsdiagramm: Liste (mittleres p̂, beobachtete Häufigkeit, Zahl Container) je besetztem Fach. `binning` aus BINNINGS."""
    f = np.asarray(f, float)
    y = np.asarray(y, float)
    if binning not in BINNINGS:
        raise ValueError(binning)
    rows = []
    if binning == "10 gleiche Masse":
        for blk in np.array_split(np.argsort(f, kind="stable"), 10):
            if len(blk):
                rows.append((float(f[blk].mean()), float(y[blk].mean()), len(blk)))
        return rows
    nb = int(binning.split()[0])
    idx = _bin_index(f, nb)
    for k in range(nb):
        m = idx == k
        if m.any():
            rows.append((float(f[m].mean()), float(y[m].mean()), int(m.sum())))
    return rows


def ece(f, y, binning: str = "10 gleich breit") -> float:
    n = len(y)
    return float(sum(c / n * abs(mf - of) for mf, of, c in reliability_bins(f, y, binning)))


def brier(f, y) -> float:
    return float(np.mean((np.asarray(f, float) - np.asarray(y, float)) ** 2))


def brier_decomposition(f, y, digits: int = 2):
    """Murphy: Brier = REL − RES + UNC auf gerundeten Vorhersagen. Gibt (Brier der gerundeten Vorhersagen, REL, RES, UNC)."""
    fr = np.round(np.asarray(f, float), digits)
    y = np.asarray(y, float)
    vals, inv = np.unique(fr, return_inverse=True)
    nk = np.bincount(inv).astype(float)
    ok = np.bincount(inv, weights=y) / nk
    ybar = float(y.mean())
    rel = float(np.sum(nk * (vals - ok) ** 2) / len(y))
    res = float(np.sum(nk * (ok - ybar) ** 2) / len(y))
    return float(np.mean((fr - y) ** 2)), rel, res, ybar * (1.0 - ybar)


def auc(score, y) -> float:
    """Fläche unter der ROC-Kurve über die mittleren Ränge (Gleichstände zählen halb); NaN, wenn nur eine Klasse vorkommt."""
    y = np.asarray(y, float)
    n1 = float(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = stats.rankdata(score)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))
