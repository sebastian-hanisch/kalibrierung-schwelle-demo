"""Das Beispiel von Hand: wenige Gruppen von Containern mit gleicher Vorhersage, alles in Brüchen (`fractions`), exakt.

Eine Gruppe hat eine Vorhersage p̂ (Bruch), eine Zahl Container n und eine Zahl e davon mit Standgeld. Kosten: Maßnahme bei Container ohne Standgeld kostet 1, keine
Maßnahme bei Container mit Standgeld kostet r (ganzzahlig). Eine Gruppenregel legt für jede Gruppe fest, ob eine Maßnahme ergriffen wird; bei wenigen Gruppen lassen sich
alle 2^g Gruppenregeln aufzählen und mit der Regel „nach beobachteter Häufigkeit und Schwelle 1/(1+r)“ vergleichen.

Die Zahlen des Standardbeispiels (drei Gruppen, 30 Container) sind in `tests/test_hand.py` von Hand nachgerechnet."""

import itertools
from dataclasses import dataclass
from fractions import Fraction as Fr


@dataclass(frozen=True)
class Group:
    p_hat: Fr        # Vorhersage der Gruppe, 0 < p̂ < 1
    n: int           # Zahl Container
    events: int      # davon mit Standgeld (0 bis n)

    def __post_init__(self):
        if not (0 < self.p_hat < 1):
            raise ValueError("Die Vorhersage p̂ muss zwischen 0 und 1 liegen.")
        if self.n < 1 or not 0 <= self.events <= self.n:
            raise ValueError("Jede Gruppe braucht mindestens einen Container, und die Zahl mit Standgeld liegt zwischen 0 und n.")

    @property
    def observed(self) -> Fr:
        return Fr(self.events, self.n)


# Standardbeispiel: das Modell ist zur Mitte gedrückt (Vorhersage 1/5, 3/10, 2/5; beobachtet 0, 1/5, 7/10)
DEFAULT_GROUPS = (Group(Fr(1, 5), 10, 0), Group(Fr(3, 10), 10, 2), Group(Fr(2, 5), 10, 7))


def total_containers(groups) -> int:
    return sum(g.n for g in groups)


def total_events(groups) -> int:
    return sum(g.events for g in groups)


def mean_prediction(groups) -> Fr:
    return sum(g.p_hat * g.n for g in groups) / total_containers(groups)


def event_share(groups) -> Fr:
    return Fr(total_events(groups), total_containers(groups))


def cost(groups, act, r: int) -> Fr:
    """Gesamtkosten einer Gruppenregel: `act[i]` sagt, ob für Gruppe i eine Maßnahme ergriffen wird (Fehlalarme n − e, sonst verpasste e · r)."""
    return sum((g.n - g.events) * 1 if a else g.events * r for a, g in zip(act, groups))


def rule_threshold(groups, t: Fr, by: str = "p_hat"):
    """Gruppenregel „Maßnahme bei Größe > t“ nach der Vorhersage (`by='p_hat'`) oder nach der beobachteten Häufigkeit (`by='observed'`)."""
    return tuple((g.p_hat if by == "p_hat" else g.observed) > t for g in groups)


def all_rules_costs(groups, r: int) -> dict:
    """Kosten aller 2^g Gruppenregeln (Aufzählung)."""
    return {act: cost(groups, act, r) for act in itertools.product((False, True), repeat=len(groups))}


def summary(groups, r: int) -> dict:
    """Kosten der Regeln bei Verhältnis r: roh (Schwelle 1/(1+r) auf p̂), fest 1/2 auf p̂, nach beobachteten Häufigkeiten mit Schwelle 1/(1+r), und das beste aller Gruppenregeln."""
    if r < 1:
        raise ValueError("Das Kostenverhältnis r muss mindestens 1 sein.")
    t = Fr(1, 1 + r)
    allc = all_rules_costs(groups, r)
    best = min(allc.values())
    return {
        "r": r, "t": t,
        "roh": cost(groups, rule_threshold(groups, t), r),
        "fest": cost(groups, rule_threshold(groups, Fr(1, 2)), r),
        "beobachtet": cost(groups, rule_threshold(groups, t, "observed"), r),
        "bestes": best,
        "beste_regeln": sorted(act for act, c in allc.items() if c == best),
        "alle": allc,
    }


def brier_decomposition(groups):
    """Brier = REL − RES + UNC in Brüchen (Murphy): gibt (Brier, REL, RES, UNC); Gruppen gleicher Vorhersage sind die Klassen der Zerlegung."""
    n = total_containers(groups)
    ybar = event_share(groups)
    brier = sum(g.events * (1 - g.p_hat) ** 2 + (g.n - g.events) * g.p_hat ** 2 for g in groups) / n
    rel = sum(g.n * (g.p_hat - g.observed) ** 2 for g in groups) / n
    res = sum(g.n * (g.observed - ybar) ** 2 for g in groups) / n
    return brier, rel, res, ybar * (1 - ybar)


def ece(groups, bins: int, left_closed: bool = True) -> Fr:
    """ECE mit `bins` gleich breiten Fächern in Brüchen. Liegt p̂ genau auf einer Fachgrenze, entscheidet die Randregel: links geschlossen (Fach ⌊p̂ · B⌋) oder rechts geschlossen."""
    cells = {}
    for g in groups:
        x = g.p_hat * bins
        k = int(x) if (left_closed or x.denominator != 1) else int(x) - 1
        cells.setdefault(min(max(k, 0), bins - 1), []).append(g)
    n = total_containers(groups)
    out = Fr(0)
    for members in cells.values():
        nn = sum(g.n for g in members)
        mean_p = sum(g.p_hat * g.n for g in members) / nn
        freq = Fr(sum(g.events for g in members), nn)
        out += Fr(nn, n) * abs(mean_p - freq)
    return out


def threshold_from_matrix(a, b, c00=0, c11=0) -> Fr:
    """Schwelle aus einer 2×2-Kostenmatrix: a = Maßnahme ohne Ereignis, b = keine Maßnahme bei Ereignis, c00 = keine Maßnahme ohne Ereignis, c11 = Maßnahme bei Ereignis.
    Maßnahme ist besser genau bei p ≥ (a − c00) / ((a − c00) + (b − c11)); nur die Differenzen je Entscheidung zählen."""
    num, den = Fr(a) - Fr(c00), (Fr(a) - Fr(c00)) + (Fr(b) - Fr(c11))
    if num <= 0 or den <= num:
        raise ValueError("Die Matrix braucht a − c00 > 0 und b − c11 > 0.")
    return num / den


def cost_curve(groups, r: int, steps: int = 100):
    """Gesamtkosten der Gruppenregel „Maßnahme bei p̂ > t“ für t = 0, 1/steps, …, 1 (Liste von (t als Float, Kosten als Float))."""
    return [(k / steps, float(cost(groups, rule_threshold(groups, Fr(k, steps)), r))) for k in range(steps + 1)]
