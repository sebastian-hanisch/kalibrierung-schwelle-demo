"""Synthetischer Containerstrom mit dem Ergebnis „Standgeld fällig“ (Verweilzeit über 7 Tage). Annahmen, kein Echtdatenbezug.

Herkunft: die Modellformeln und Konstanten (Anteile der Containerarten, Kundeneffekte, Streuung je Art, Zollhalt, Mittelfunktion `log_median_and_s`) sind aus `verweilzeit-demo/vwz_generator.py`
und `vwz_constants.py` übernommen und mit Präfix umbenannt, ohne die Stapelplanung und ohne Import. Zwei Unterschiede zur Quelle: (1) Der Zufall kommt aus dem ganzzahligen
SplitMix64 (`ksw_rng.py`), nicht aus numpy; die zwölf Kundeneffekte stehen deshalb als Zahlen im Quelltext (die Quelle zieht sie mit `numpy.random.default_rng(12345).normal(0, 0.3, 12)`).
(2) Das Rauschmerkmal „Gewicht“ (Wirkung null) und der Verweilzeit-Rauschterm sind Normalzahlen aus `ksw_rng.normal_array`. Eine einzelne Messung nimmt also andere Container als die
Quelle; die Verteilung der Merkmale und der Ergebnisse ist dieselbe (`tests/test_generator.py` prüft sie gegen Messwerte der Quelle als Band).

Verweilzeit d = exp(mu + s z) mit z ~ N(0, 1), Standgeld y = 1 genau dann, wenn d > 7. Die **wahre Wahrscheinlichkeit** p = 1 − Φ((ln 7 − mu) / s) folgt aus den Merkmalen allein
und ist der Orakelwert, an dem Kosten ohne Rauschen der Ergebnisse gerechnet werden.

Merkmale je Container (Spalten von X): 0 Art (0 Import, 1 Export, 2 Umschlag, 3 Leer), 1 Kunde (0 bis 11), 2 Zollhalt (nur Import), 3 Wochenende, 4 Vorlauf in Tagen
(nur Export und Umschlag), 5 Gewicht (Rauschen)."""

from dataclasses import dataclass

import numpy as np
from scipy import special

import ksw_rng as R

ART_LABELS = ("Import", "Export", "Umschlag", "Leer")
ART_P = (0.35, 0.30, 0.25, 0.10)             # Anteile der Containerarten
K_KUNDEN = 12
U_SCALE = (1.0, 0.3, 0.3, 1.0)               # Kundeneffekt je Art
S_BASE = (0.55, 0.25, 0.35, 0.90)            # Streuung der log-Verweilzeit je Art
P_ZOLL = 0.25                                # Zollhalt bei Import
# Kundeneffekte der Quelle: numpy.random.default_rng(12345).normal(0.0, 0.30, 12) (als Zahlen übernommen)
UK = np.array([-0.42714751093638936, 0.3791185374387331, -0.2611985213877257, -0.07775197048031927, -0.02260299210315629, -0.2222653956256827,
               -0.410337810534883, 0.19466784065791196, 0.10831743391646849, -0.5858589189036569, 0.7042228963136555, 0.2905490717255771])
STAND_TAGE = 7.0                             # Standgeld fällig, wenn die Verweilzeit diese Tage übersteigt (Annahme dieser Demo)
LN_STAND = float(np.log(STAND_TAGE))
N_FEATURES = 6


def stream_seed(seed: int, k: int) -> int:
    """Eigener Zustand je Merkmalsspalte und Datensatz: ein Zufalls-Seed gibt über k getrennte Ströme (die ersten n Zeilen hängen nicht von n ab)."""
    g = R.SplitMix64((int(seed) & R.MASK) * 1_000_003 + k + 1)
    g.next_u64()
    return g.next_u64()


def log_median_and_s(art, kunde, zoll, we, vorlauf):
    """Bedingte Verteilung der log-Verweilzeit: N(mu, s²). Vektorisiert. Das ist der Orakelwert (aus der Quelle übernommen)."""
    art = np.asarray(art).astype(int)
    kunde = np.asarray(kunde).astype(int)
    u = UK[kunde] * np.take(np.array(U_SCALE), art)
    med = np.select(
        [art == 0, art == 1, art == 2],
        [3.5 * np.exp(0.7 * zoll + 0.30 * we), 0.6 + 0.85 * vorlauf, 0.3 + 0.8 * vorlauf],
        default=7.0 * np.exp(0.30 * we),
    )
    mu = np.log(med) + u
    s = np.take(np.array(S_BASE), art)
    return mu, s


def true_probability(mu, s):
    """p = P(Verweilzeit > 7 Tage) = 1 − Φ((ln 7 − mu) / s)."""
    return special.ndtr((np.asarray(mu) - LN_STAND) / np.asarray(s))


def draw_features(n: int, seed: int, noise: bool = True) -> np.ndarray:
    """n Container (Spalten siehe Modulkopf). Jede Spalte hat einen eigenen Strom; `noise=False` lässt das Rauschmerkmal weg (Spalte 5 bleibt 0)."""
    u = lambda k: R.uniform_array(stream_seed(seed, k), n)
    art = np.searchsorted(np.cumsum(ART_P), u(0), side="right").clip(0, 3)
    kunde = np.floor(u(1) * K_KUNDEN).astype(int).clip(0, K_KUNDEN - 1)
    zoll = ((u(2) < P_ZOLL) & (art == 0)).astype(float)
    tag = np.floor(u(3) * 7000.0).astype(int)
    we = ((tag % 7) >= 5).astype(float)
    uv = u(4)
    vorlauf = np.where(art == 1, 1.0 + 9.0 * uv, np.where(art == 2, 0.5 + 5.5 * uv, 0.0))
    gewicht = R.normal_array(stream_seed(seed, 5), n) if noise else np.zeros(n)
    return np.column_stack([art.astype(float), kunde.astype(float), zoll, we, vorlauf, gewicht])


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray        # (n, 6)
    mu: np.ndarray
    s: np.ndarray
    p: np.ndarray        # wahre Wahrscheinlichkeit
    d: np.ndarray        # Verweilzeit in Tagen
    y: np.ndarray        # Standgeld 0/1

    @property
    def n(self) -> int:
        return len(self.y)


def make_dataset(n: int, seed: int) -> Dataset:
    """n abgeschlossene Container (Merkmale, Verweilzeit, Standgeld) aus dem Zufalls-Seed `seed`."""
    X = draw_features(n, seed)
    mu, s = log_median_and_s(X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4])
    z = R.normal_array(stream_seed(seed, 6), n)
    log_d = mu + s * z
    return Dataset(X, mu, s, true_probability(mu, s), np.exp(log_d), (log_d > LN_STAND).astype(float))


def population(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Wahre Wahrscheinlichkeiten und Containerart von n Containern ohne Ergebnisse und ohne Rauschmerkmal (schnell, für Kosten bei wahrem p)."""
    X = draw_features(n, seed, noise=False)
    mu, s = log_median_and_s(X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4])
    return true_probability(mu, s), X[:, 0].astype(int)


def design(X: np.ndarray, with_customer: bool = True) -> np.ndarray:
    """Entwurfsmatrix des Logit: 1, drei Art-Dummies (Export, Umschlag, Leer), Zoll, Wochenende, log(1 + Vorlauf), Gewicht, 11 Kunden-Dummies (Kunde 0 = Bezug): 19 Spalten."""
    art = X[:, 0].astype(int)
    cols = [np.ones(len(X)), art == 1, art == 2, art == 3, X[:, 2], X[:, 3], np.log1p(X[:, 4]), X[:, 5]]
    if with_customer:
        cols += [X[:, 1] == k for k in range(1, K_KUNDEN)]
    return np.column_stack(cols).astype(float)
