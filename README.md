# Kalibrierung und Schwelle – aus einer Wahrscheinlichkeit wird eine Entscheidung – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-kalibrierung-schwelle-demo.streamlit.app/)** -->

Eine Demo zu **Regression und verallgemeinerten linearen Modellen** für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Sie zeigt an einem Containerstrom (Ergebnis **Standgeld fällig**: Verweilzeit über 7 Tage), dass aus einer Vorhersage *p̂* erst durch das **Kostenverhältnis r** (ein verpasstes Standgeld kostet das *r*-Fache einer unnötigen Maßnahme) eine Entscheidung wird: Die beste Schwelle ist 1/(1+*r*), nicht 0,5, **vorausgesetzt *p̂* stimmt als Häufigkeit**. Ob es stimmt, zeigen weder die Kalibrierung im Mittel noch ein kleiner ECE; Rekalibrierung behebt eine verzerrte Skala, nicht eine fehlende Rangfolge, und braucht Daten.

Verwandte Demos zu Regression und Modellen: [`glm-familie-demo`](https://github.com/sebastian-hanisch/glm-familie-demo) (Familie und Link, ein IRLS-Code), [`logit-separation-demo`](https://github.com/sebastian-hanisch/logit-separation-demo) (wann der Logit-Schätzer existiert), [`quantil-ueberdeckung-demo`](https://github.com/sebastian-hanisch/quantil-ueberdeckung-demo) (Quantile statt Mittelwert), [`modellwahl-demo`](https://github.com/sebastian-hanisch/modellwahl-demo) (AIC, BIC, Kreuzvalidierung). Dieselbe Schwelle ist das τ-Quantil eines 0/1-Ergebnisses: [`newsvendor-direkt-demo`](https://github.com/sebastian-hanisch/newsvendor-direkt-demo) lernt dagegen eine Menge bei stetiger Nachfrage (Pinball-Verlust, Quantilregression), [`regret-statt-fehler-demo`](https://github.com/sebastian-hanisch/regret-statt-fehler-demo) erklärt den Regret als Kosten der Entscheidung minus Kosten der besten Entscheidung bei bekannter Verteilung (hier: bei bekanntem *p*). Das Vehikel (Container, Verweilzeit, Merkmale) stammt aus [`verweilzeit-demo`](https://github.com/sebastian-hanisch/verweilzeit-demo). [`isolation-forest-demo`](https://github.com/sebastian-hanisch/isolation-forest-demo) sagt dieselbe Lehre für einen Anomalie-Score (kein kalibrierter Wert, die Rangfolge bleibt gut); [`forecast-interval-demo`](https://github.com/sebastian-hanisch/forecast-interval-demo) benutzt „Kalibrierung“ für die Menge, mit der Überdeckung von Intervallen eingestellt wird (hier heißt das **Rekalibriermenge** und meint etwas anderes). Zahlen aus diesen Demos werden hier weder nachgerechnet noch verglichen.

## Warum dieses Problem

Wer eine Wahrscheinlichkeit vorhersagt, entscheidet noch nichts. **Welche Schwelle gehört zu welchem Kostenverhältnis, stimmt *p̂* als Häufigkeit, und was repariert Rekalibrierung?** Im Containerstrom dieser Demo (Anteil Standgeld 23,5 % nach der wahren Wahrscheinlichkeit, 100 000 Container, Zufalls-Seed 7) kostet die feste Schwelle 0,5 bei r = 9 und bekanntem wahrem *p* 561,5 Einheiten je 1000 Entscheidungen mehr als die Schwelle 1/10. Ein Logit auf allen Merkmalen, angepasst auf 500 Containern, liegt der Regret im Mittel über 40 Trainingsmengen bei 41,7 ± 18,8 mit der Schwelle 1/10 und bei 625,3 ± 101,0 mit 0,5 (0,5 ist teurer in 40 von 40 Trainingsmengen). Dass die Vorhersage dabei nicht stimmt, sieht man nicht am Mittel (im Training höchstens 2,1e-10 Abstand von mittlerem *p̂* und Anteil Standgeld bei den vier Logit-Modellen), sondern an der Steigung der Kalibrierung (0,81 ± 0,10 auf den Testdaten, 1 hieße kalibriert), und ein kleiner ECE schützt nicht vor einem schlechten Modell: „ohne Containerart (Rangfolge fehlt)“ hat den ECE-Rang 3 von 8 und bei r = 9 den höchsten Regret (406,8; bei r = 19 hat „überkonfident“ den höchsten, 403,7).
Alle Formeln und Befunde sind **eigene Nachrechnung in diesem Modell, bei diesen Parametern** (Container-Generator, Standgeld nach 7 Tagen, 500 Trainingscontainer, gesetzte Kostenverhältnisse). Die **Kosten sind angenommen** (1 für eine unnötige Maßnahme, *r* für ein verpasstes Standgeld), kein Geldbetrag. Eine Neuheitsaussage gibt es nicht: Schwelle 1/(1+*r*), Murphy-Zerlegung, Platt-Skalierung und PAV sind Lehrbuchstoff (siehe Quellen); neu ist nur die Vorführbarkeit mit Zähleinheiten in einem wachsenden Beispiel.

## Modell

**Vehikel.** Container mit Art (Import, Export, Umschlag, Leer), Kunde (12), Zollhalt (nur Import), Wochenende, Vorlauf (Export, Umschlag) und einem Rauschmerkmal (Gewicht, Wirkung null). Die Verweilzeit ist lognormal, log d ~ N(μ, s²) mit μ und s aus den Merkmalen; **Standgeld y = 1 genau dann, wenn d > 7 Tage** (die Frist ist eine Annahme dieser Demo). Die **wahre Wahrscheinlichkeit** p = 1 − Φ((ln 7 − μ)/s) folgt aus den Merkmalen allein; sie macht Kosten ohne Rauschen der Ergebnisse möglich (`ksw_generator.py`).
**Entscheidung und Kosten.** Eine Maßnahme (Kunde vorwarnen, Abholtermin nachfassen; was sie ist, ist eine Annahme) bei y = 0 kostet 1, keine Maßnahme bei y = 1 kostet *r*, richtig entschieden kostet 0. **Konvention:** r = Kosten verpasst : Kosten unnötig (andere Quellen schreiben das Verhältnis umgekehrt; dann lautet die Schwelle ρ/(1+ρ) mit ρ = 1/r). Bei wahrer Wahrscheinlichkeit *p* kostet die Maßnahme im Mittel 1 − *p*, die Unterlassung *r*·*p*: die Maßnahme ist besser genau bei **p > 1/(1+r)**; bei Gleichstand kosten beide dasselbe (Konvention der Demo: Maßnahme bei *p̂* > Schwelle). Eine allgemeine 2×2-Kostenmatrix hat dafür einen Freiheitsgrad (Elkan 2001, Quellenbericht Stufe W; Gleichung (2) dort nicht neu geöffnet, die Umformung ist selbst nachgerechnet): Schwelle (a − c₀₀)/((a − c₀₀) + (b − c₁₁)).
**Regret** = erwartete Kosten der Regel bei wahrer Wahrscheinlichkeit der Testcontainer minus die der Orakelregel (wahres *p*, Schwelle 1/(1+*r*)), **je 1000 Entscheidungen**, nie negativ. Das ist eine andere Zählung als „Kosten auf den beobachteten Ergebnissen“; Zahlen sind deshalb nur innerhalb dieser Demo vergleichbar.
**Kalibrierung** heißt Vorhersage und beobachtete Häufigkeit passen zusammen (probabilistische Kalibrierung). **Konventionen:** **ECE** = Σ (Anteil im Fach)·|mittleres *p̂* − beobachtete Häufigkeit| mit der angegebenen Fächerzahl (gleich breit, links geschlossen: Fach ⌊*p̂*·B⌋; oder gleiche Masse) – die Randregel ändert den Wert (Beispiel von Hand); **Brier** = Mittel von (*p̂* − y)² für ein 0/1-Ergebnis; **REL, RES, UNC** nach Murphy auf zwei Stellen gerundeten Vorhersagen (nicht die MCB und DSC der CORP-Zerlegung von Dimitriadis u. a.); **Achsenabschnitt und Steigung** = Zwei-Parameter-Anpassung y ~ a + b·logit *p̂* (nicht der Achsenabschnitt bei auf 1 festgehaltener Steigung; diese Variante ist nicht gemessen); **Platt-Skalierung** ohne geglättete Zielwerte (die Originalarbeit nutzt nach meiner Erinnerung welche, Stufe G, nicht nachgelesen); **isotonisch** = nichtfallende Anpassung (PAV) mit linearer Interpolation zwischen den Stützstellen und Kappung außerhalb (eine Stufenfunktion gäbe an den Zwischenwerten andere Zahlen).
**Acht Modelle** (`ksw_models.py`): fünf angepasste (Logit auf allen Merkmalen = „Haupteffekte“, 18 Spalten, nahezu unbestraft; lognormal je Art = die richtige Familie; je Art = ein eigenes Logit je Containerart, überangepasst; Ridge „stark geglättet“; ohne Containerart) und drei kontrolliert verzerrte Fassungen der Haupteffekte („verdichtet“, „überkonfident“, „verschoben“: p̃ = expit(a + b·logit *p̂*) mit (a; b) = (0; 0,5), (0; 2), (1; 1)). Bei den verzerrten Fassungen bleibt die Rangfolge der Container gleich, nur die Skala ist verzerrt (AUC gleich bis auf höchstens 2,0·10⁻⁴: die Kappung der Vorhersage bei 10⁻⁹ erzeugt beim Modell „überkonfident“ einzelne Gleichstände; geprüft).
**Gilt** für diesen Generator, 500 Trainingscontainer, Standgeld nach 7 Tagen, die genannten Modelle und ein angenommenes *r*; nicht gemessen sind andere Fristen, andere Trainingsgrößen, Kosten je Container und mehr als zwei Ausgänge.

## Methodik

| Verfahren | Wie es arbeitet | Grenze |
|---|---|---|
| **Logit** (`ksw_core.fit_logit`) | Newton-Verfahren mit Schritthalbierung; Achsenabschnitt unbestraft, Ridge λ = 1/C auf den übrigen Spalten (C = 10⁴ für nahezu unbestraft, C = 0,03 auf standardisierten Spalten für „stark geglättet“) | ohne Strafe bei Trennung nicht bestimmt (die kleine Strafe hält es endlich); Gleichung Σ (y − p̂) = 0 gilt nur bei unbestraftem Achsenabschnitt |
| **Platt** (`ksw_core.fit_platt`) | dasselbe Newton-Verfahren auf (1, logit *p̂*), zwei Parameter, Ridge 10⁻⁸ | bei (fast) konstanter Vorhersage ist die Steigung nicht bestimmt (die App meldet es); trennt bei kleiner Rekalibriermenge |
| **Isotonisch** (`ksw_core.isotonic_fit`) | gleiche Vorhersagen gewichtet zusammenfassen, Pool Adjacent Violators, lineare Interpolation, Kappung | verliert bei wenig Daten (siehe Befunde); Blöcke erzeugen Gleichstände |
| **Abgestimmte Schwelle** (`ksw_core.tuned_threshold`) | die Schwelle aus 49 Kandidaten (0,02 bis 0,98) mit den kleinsten Kosten auf den beobachteten Ergebnissen der Rekalibriermenge | streut stark bei wenig Ereignissen (Befunde) |
| **Kosten und Regret** (`ksw_core.expected_cost`, `regret_per_1000`) | erwartete Kosten bei wahrem *p*; zweiter Rechenweg über Σ \|(1+*r*)*p* − 1\| der abweichenden Entscheidungen | setzt die wahre Wahrscheinlichkeit voraus (nur im Generator bekannt) |
| **Gütemaße** | ECE, Brier-Zerlegung, AUC (mittlere Ränge), Zuverlässigkeitsdiagramm | ECE hängt an Fächerzahl und Randregel und hat einen Boden (Befunde) |

**Zufall.** Nur der ganzzahlige SplitMix64 (`ksw_rng.py`, aus `plan-stoerung-demo` kopiert, Normalzahlen aus Grundrechenarten und Wurzel, auf Windows und Linux bitgleich); kein `hash()`, kein NumPy-Zufall im Programm. Die zwölf Kundeneffekte stehen als Zahlen im Quelltext (die Quelle zieht sie mit NumPys Generator; ein Test prüft die Gleichheit). **Vorgerechnet:** die Messungen über 40 Trainingsmengen (500 Container; Zufalls-Seeds 500 bis 539; Rekalibriermengen von 2000 Containern; 100 000 Testcontainer, Zufalls-Seed 999) stehen in `ksw_frozen.json` und werden von `tools/ksw_make_frozen.py` (etwa zwei Minuten) erzeugt; App und README nennen sie „vorgerechnet“. **Live** rechnet die App eine Trainingsmenge (Zufalls-Seed einstellbar), die Rekalibrierung und 20 000 Testcontainer (rund 0,05 Sekunden je Eingabe für die Rechnung).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Container-Generator, Standgeld nach 7 Tagen, 500 Trainingscontainer, Zähleinheit **Entscheidungen** (Container) für Kosten und ECE, **Trainingsmengen** (je 500 Container, 40 Trainingsmengen, Zufalls-Seeds 500 bis 539) für Mittel (± Standardabweichung über die Trainingsmengen) und Zählungen („in k von 40“), **Wiederholungen** für den ECE-Boden. Regret in Einheiten je 1000 Entscheidungen bei wahrer Wahrscheinlichkeit der 100.000 Testcontainer (Zufalls-Seed 999; Standgeld-Anteil 23,2 %, Mittel der wahren Wahrscheinlichkeiten 23,4 %). Orakel-Kosten je 1000 Entscheidungen (Testcontainer): 160,4 / 277,7 / 393,9 / 462,5 bei r = 1 / 3 / 9 / 19. Die Kosten sind angenommen, kein Geldbetrag.

**1. Schwelle bei bekanntem wahrem *p*** (100 000 Container, Zufalls-Seed 7, ohne Rauschen der Ergebnisse; Anteil Standgeld 23,5 %; je Art 26,5 % / 26,7 % / 3,8 % / 52,1 % für Import / Export / Umschlag / Leer). Regret je 1000 Entscheidungen; Zeile = wahres r, Spalte = angenommenes r′ (Schwelle 1/(1+r′)); die Spalte r′ = 1 ist die feste Schwelle 0,5:

| wahres r | r′ = 1 (= 0,5) | r′ = 3 | r′ = 9 | r′ = 19 |
|---|---|---|---|---|
| 1 | 0,0 | 41,6 | 139,4 | 216,6 |
| 3 | 81,3 | 0,0 | 45,4 | 108,4 |
| 9 | 561,5 | 111,8 | 0,0 | 20,2 |
| 19 | 1488,8 | 424,9 | 51,1 | 0,0 |

Kostenkurve bei r = 9 (je 1000 Entscheidungen, bei wahrem *p*) für t = 0,02 / 0,05 / 0,1 / 0,15 / 0,2 / 0,3 / 0,5: 494,4 / 416,3 / 396,1 / 414,8 / 448,0 / 566,4 / 957,6; das Minimum liegt bei t = 0,1. Bei r = 1 ist 0,5 die Orakelschwelle (Regret 0).

**2. Was die feste 0,5 im Modell kostet** (Haupteffekte-Logit, 40 Trainingsmengen; Regret je 1000, Mittel ± Streuung):

| r | Schwelle 1/(1+r) | feste Schwelle 0,5 | 0,5 teurer in |
|---|---|---|---|
| 1 | 11,6 ± 4,7 | 11,6 ± 4,7 | 0 von 40 |
| 3 | 25,4 ± 8,5 | 106,1 ± 26,5 | 40 von 40 |
| 9 | 41,7 ± 18,8 | 625,3 ± 101,0 | 40 von 40 |
| 19 | 52,1 ± 30,0 | 1615,8 ± 225,7 | 40 von 40 |

**Ausnahme im selben Satz:** beim Modell „verschoben“ (Achsenabschnitt +1) ist 0,5 bei r = 3 nur in 3 von 40 Trainingsmengen teurer als 1/(1+r) (28,2 ± 9,5 gegen 53,2 ± 13,7), bei r = 9 in 40 von 40; beim Modell „verdichtet“ ist es bei r = 3 in 37 von 40 teurer; „ohne Containerart“ bei r = 3 in 39 von 40. Bei r = 1 sind beide Regeln dieselbe. **Nicht als Vielfaches:** das Verhältnis der Mittel (0,5 durch 1/(1+r)) ist für die Haupteffekte 4,2 / 15,0 / 31,0 bei r = 3 / 9 / 19, aber der Nenner streut mit ± 8,5 / 18,8 / 30,0; die Demo nennt deshalb die Differenz in Einheiten je 1000.

**3. Stimmt *p̂* als Häufigkeit?** (acht Modelle, Mittel über 40 Trainingsmengen, Testcontainer; ECE mit 10 gleich breiten Fächern; Rang 1 = bester Wert; Steigung = Zwei-Parameter-Anpassung auf den Testdaten):

| Modell | AUC | ECE | Rang ECE | Brier | Rang Brier | Steigung | Regret roh (r = 9) | Rang Regret |
|---|---|---|---|---|---|---|---|---|
| Haupteffekte (Logit, 18 Spalten) | 0,877 | 0,0301 | 2 | 0,1175 | 1 | 0,81 ± 0,10 | 41,7 | 2 |
| lognormal je Art (richtige Familie) | 0,876 | 0,0271 | 1 | 0,1179 | 2 | 0,80 ± 0,05 | 33,6 | 1 |
| je Art (Wechselwirkung, überangepasst) | 0,838 | 0,0742 | 6 | 0,1371 | 5 | 0,30 ± 0,06 | 165,4 | 5 |
| Ridge „stark geglättet“ | 0,805 | 0,0462 | 4 | 0,1418 | 7 | 1,47 ± 0,15 | 220,4 | 7 |
| ohne Containerart (Rangfolge fehlt) | 0,680 | 0,0412 | 3 | 0,1627 | 8 | 0,76 ± 0,13 | 406,8 | 8 |
| Haupteffekte „verdichtet“ (Steigung 0,5) | 0,877 | 0,0672 | 5 | 0,1222 | 3 | 1,62 ± 0,21 | 117,7 | 4 |
| Haupteffekte „überkonfident“ (Steigung 2) | 0,877 | 0,0907 | 7 | 0,1297 | 4 | 0,40 ± 0,05 | 199,3 | 6 |
| Haupteffekte „verschoben“ (Achsenabschnitt +1) | 0,877 | 0,1132 | 8 | 0,1379 | 6 | 0,81 ± 0,10 | 63,4 | 3 |

- **Kalibrierung im Mittel sagt nichts über die Steigung.** Mittleres *p̂* minus Anteil Standgeld im Training, größter Betrag über 40 Trainingsmengen: Haupteffekte 7,5e-12, je Art 2,1e-10, stark geglättet 5,6e-17, ohne Containerart 5,6e-17; die Steigung auf den Testdaten ist dabei 0,81 ± 0,10 / 0,30 ± 0,06 / 1,47 ± 0,15 / 0,76 ± 0,13. Steigung unter 1 heißt zu extreme Vorhersagen (Van Calster u. a. 2019, Stufe W); sie tritt schon beim Modell der richtigen Familie (lognormal je Art (richtige Familie): 0,80 ± 0,05) und beim nahezu unbestraften Logit mit 18 Spalten auf 500 Containern auf. Deutung, nicht eigens getestet: zu viele Spalten für zu wenig Daten; zur Verzerrung des Maximum-Likelihood-Schätzers bei vielen Spalten siehe Sur und Candès 2019 (Stufe W).
- **ECE sagt nicht, welches Modell entscheidet.** Der Regret ist bei „ohne Containerart (Rangfolge fehlt)“ der höchste der acht Modelle (406,8 bei r = 9), sein ECE hat den Rang 3 von 8 (Brier und RMSE: Rang 8 und 8). Die vier Fächerschemata (5, 10, 20 gleich breit, 10 gleiche Masse) ändern die ECE-Werte (stark geglättet 0,0416 / 0,0462 / 0,0504 / 0,0489), aber nicht die Reihenfolge der acht Modelle; die Reihenfolge ändert sich zwischen ECE und Brier oder RMSE, und für 8 von 8 Modellen ändert sich der Platz zwischen den sechs Maßen. Die Aussage gilt für dieses Paar von Maßen und diese acht Modelle, nicht für Maße im Allgemeinen.
- **ECE hat einen Boden** (die wahre Wahrscheinlichkeit, perfekt kalibriert, mit eigenen gezogenen Ergebnissen; Wiederholungen: 200, 200, 100, 20): bei n = 200 / 2000 / 20 000 / 100 000 Containern und 10 gleich breiten Fächern 0,0562 ± 0,0145 / 0,0178 ± 0,0048 / 0,0057 ± 0,0013 / 0,0025 ± 0,0005; bei 5 Fächern 0,0125 ± 0,0048, bei 20 Fächern 0,0249 ± 0,0050 (n = 2000). Ein ECE von 0,02 auf 2000 Containern ist von „perfekt“ nicht zu trennen.

**4. Rekalibrieren** (Rekalibriermenge von 2000 Containern, r = 9; Regret je 1000, Mittel ± Streuung über 40 Trainingsmengen; Zählungen aus denselben Trainingsmengen; Gleichstände zählen nicht als billiger):

| Modell | roh | feste 0,5 | Platt | isotonisch | abgestimmt |
|---|---|---|---|---|---|
| Haupteffekte (Logit, 18 Spalten) | 41,7 ± 18,8 | 625,3 ± 101,0 | 38,8 ± 18,3 | 37,6 ± 17,1 | 38,9 ± 16,3 |
| lognormal je Art (richtige Familie) | 33,6 ± 16,3 | 605,9 ± 83,7 | 29,6 ± 12,2 | 30,9 ± 10,4 | 31,4 ± 12,0 |
| je Art (Wechselwirkung, überangepasst) | 165,4 ± 56,5 | 640,6 ± 100,3 | 178,3 ± 40,5 | 140,8 ± 40,8 | 146,6 ± 49,0 |
| Ridge „stark geglättet“ | 220,4 ± 19,7 | 1309,6 ± 130,9 | 184,1 ± 22,5 | 176,4 ± 24,4 | 181,6 ± 24,7 |
| ohne Containerart (Rangfolge fehlt) | 406,8 ± 37,5 | 1349,6 ± 103,2 | 374,7 ± 13,1 | 375,2 ± 17,1 | 375,5 ± 14,0 |
| Haupteffekte „verdichtet“ (Steigung 0,5) | 117,7 ± 25,5 | 625,3 ± 101,0 | 38,8 ± 18,3 | 37,6 ± 17,1 | 36,8 ± 14,0 |
| Haupteffekte „überkonfident“ (Steigung 2) | 199,3 ± 49,3 | 625,3 ± 101,0 | 38,8 ± 18,3 | 37,6 ± 17,1 | 55,4 ± 22,2 |
| Haupteffekte „verschoben“ (Achsenabschnitt +1) | 63,4 ± 17,3 | 227,3 ± 56,2 | 38,8 ± 18,3 | 37,6 ± 17,1 | 38,3 ± 15,5 |

| Modell | 0,5 teurer als roh | Platt billiger als roh | isotonisch billiger als roh | Platt billiger als isotonisch | abgestimmt billiger als Platt |
|---|---|---|---|---|---|
| Haupteffekte (Logit, 18 Spalten) | 40 | 24 | 29 | 16 | 19 |
| lognormal je Art (richtige Familie) | 40 | 28 | 20 | 25 | 18 |
| je Art (Wechselwirkung, überangepasst) | 40 | 17 | 32 | 3 | 35 |
| Ridge „stark geglättet“ | 40 | 35 | 35 | 12 | 22 |
| ohne Containerart (Rangfolge fehlt) | 40 | 31 | 32 | 19 | 20 |
| Haupteffekte „verdichtet“ (Steigung 0,5) | 40 | 40 | 40 | 16 | 23 |
| Haupteffekte „überkonfident“ (Steigung 2) | 40 | 40 | 40 | 16 | 9 |
| Haupteffekte „verschoben“ (Achsenabschnitt +1) | 40 | 35 | 36 | 16 | 22 |

- **Rekalibrierung behebt eine falsche Skala, kaum eine fehlende Rangfolge.** Bei den drei verzerrten Fassungen (Rangfolge gleich: AUC gleich auf 10⁻⁹ in 40 von 40 Trainingsmengen bei „verdichtet“ und „verschoben“, in 34 von 40 bei „überkonfident“ mit höchstens 2,0·10⁻⁴ Unterschied wegen der Kappung) fällt der Regret bei r = 9 von 117,7 („verdichtet“) auf 38,8 (Platt) und 37,6 (isotonisch), von 199,3 („überkonfident“) und von 63,4 („verschoben“) auf denselben Wert; die Haupteffekte selbst gehen von 41,7 auf 38,8. Platt ist logit-affin invariant: das Ergebnis ist für alle Verzerrungen der Haupteffekte gleich (38,8 bei „verdichtet“, „überkonfident“, „verschoben“ und den Haupteffekten). Bei „ohne Containerart (Rangfolge fehlt)“ (AUC 0,680) geht der Regret nur von 406,8 auf 374,7 (Platt) und 375,2 (isotonisch). Beim Ridge-Modell „stark geglättet“ (Rangfolge und Skala verändern sich zugleich) von 220,4 auf 184,1 (Platt).
- **Negativbefunde.** Beim überangepassten Modell „je Art“ ist Platt im Mittel **teurer** als roh (178,3 gegen 165,4; Platt billiger in 17 von 40), die isotonische Rekalibrierung billiger (140,8, in 32 von 40). Die auf den Kosten der Rekalibriermenge abgestimmte Schwelle ist nicht besser als Platt (Haupteffekte, r = 1 / 3 / 9 / 19: billiger als Platt in 13 / 20 / 19 / 17 von 40); die gewählte Schwelle streut (Haupteffekte, r = 9: 0,078 ± 0,025 gegen 0,100; r = 19: 0,040 ± 0,021 gegen 0,050). Beim Modell „überkonfident“ ist die abgestimmte Schwelle bei r = 19 im Mittel sehr teuer (200,9 gegen 43,6 nach Platt), Deutung (nur nachgerechnet, nicht getestet an den Messwerten): die Orakelschwelle 1/20 auf *p̂* entspricht nach der Verzerrung expit(2·logit 0,05) ≈ 0,003 und liegt unter der kleinsten Kandidatin des Rasters (0,02).

**5. Wie viele Ereignisse braucht Rekalibrierung?** (Haupteffekte, r = 9; zu jeder Trainingsmenge eine neue Rekalibriermenge der Größe m, Zufalls-Seeds 30 000 und aufwärts; Regret je 1000, Mittel ± Streuung über 40 Trainingsmengen; in Klammern: in wie vielen Trainingsmengen billiger als roh 41,7):

| m | Ereignisse im Mittel | roh | Platt | isotonisch | abgestimmt |
|---|---|---|---|---|---|
| 100 | 24 | 41,7 ± 18,8 | 52,8 ± 28,7 (16) | 73,6 ± 46,2 (5) | 69,0 ± 41,9 |
| 300 | 69 | 41,7 ± 18,8 | 41,5 ± 16,5 (20) | 56,7 ± 35,0 (15) | 54,6 ± 32,0 |
| 1000 | 231 | 41,7 ± 18,8 | 38,2 ± 18,8 (26) | 39,0 ± 16,7 (24) | 39,1 ± 17,0 |
| 2000 | 471 | 41,7 ± 18,8 | 38,3 ± 18,3 (28) | 37,7 ± 15,6 (25) | 40,4 ± 18,6 |
| 3000 | 702 | 41,7 ± 18,8 | 38,3 ± 18,3 (23) | 37,1 ± 16,0 (27) | 36,7 ± 15,6 |
| 10000 | 2347 | 41,7 ± 18,8 | 38,6 ± 17,6 (22) | 33,4 ± 14,9 (35) | 34,5 ± 15,5 |

Bei r = 3 (roh 25,4 ± 8,5) steht Platt für m = 100 / 300 / 1000 bei 29,6 / 27,0 / 23,2. Dieselbe Rekalibrierung bei der starken Verzerrung „verdichtet“ (r = 9, roh 117,7 ± 25,5) hilft schon mit m = 100 (im Mittel 24 Ereignisse): Platt 52,8 ± 28,7, billiger als roh in 39 von 40. **Die Antwort hängt also an der Größe der Verzerrung:** bei den Haupteffekten (Steigung 0,81) schadet eine Rekalibriermenge von 100 Containern (Platt 52,8 und isotonisch 73,6 gegen roh 41,7), Platt ist erst ab m = 1000 in mehr als der Hälfte der Trainingsmengen billiger als roh; die isotonische Rekalibrierung liegt bei knappen Daten hinter Platt (m = 100: 73,6 gegen 52,8; m = 300: 56,7 gegen 41,5) und ist bei m = 2000, 3000 und 10 000 im Mittel nicht mehr teurer als Platt. Die Richtung passt zu Niculescu-Mizil und Caruana 2005 (nur Stufe S, nicht nachgelesen). Van Calster u. a. nennen als Orientierung für ein Kalibrierdiagramm mindestens 200 Ereignisse und 200 Nichtereignisse (keine Schranke); die Demo warnt unter diesen Zahlen.

**6. Gegenprobe mit anderen Zufalls-Seeds** (Trainingsmengen 700 bis 739, Rekalibriermengen 40 000 bis 40 039, Test-Zufalls-Seed 1999, Orakel-Kosten je 1000: 160,9 / 278,9 / 395,3 / 463,4 bei r = 1 / 3 / 9 / 19; Regret je 1000, Mittel ± Streuung über 40 Trainingsmengen):

| Modell | r | roh | feste 0,5 | Platt | isotonisch | 0,5 teurer als roh |
|---|---|---|---|---|---|---|
| Haupteffekte (Logit, 18 Spalten) | 1 | 10,2 ± 3,3 | 10,2 ± 3,3 | 10,0 ± 3,4 | 10,2 ± 3,2 | 0 von 40 |
| Haupteffekte (Logit, 18 Spalten) | 3 | 24,4 ± 7,7 | 98,8 ± 23,2 | 23,3 ± 7,1 | 23,5 ± 8,1 | 40 von 40 |
| Haupteffekte (Logit, 18 Spalten) | 9 | 41,8 ± 20,2 | 602,0 ± 90,2 | 37,8 ± 22,2 | 36,3 ± 18,0 | 40 von 40 |
| Haupteffekte (Logit, 18 Spalten) | 19 | 53,7 ± 33,2 | 1566,7 ± 202,1 | 45,4 ± 33,1 | 44,9 ± 27,5 | 40 von 40 |
| Ridge „stark geglättet“ | 1 | 47,0 ± 4,6 | 47,0 ± 4,6 | 43,9 ± 3,9 | 43,5 ± 3,3 | 0 von 40 |
| Ridge „stark geglättet“ | 3 | 119,4 ± 15,4 | 300,6 ± 29,1 | 117,1 ± 12,5 | 113,0 ± 12,4 | 40 von 40 |
| Ridge „stark geglättet“ | 9 | 218,1 ± 19,4 | 1299,0 ± 104,6 | 178,6 ± 25,0 | 171,8 ± 28,2 | 40 von 40 |
| Ridge „stark geglättet“ | 19 | 270,9 ± 11,5 | 3089,1 ± 230,8 | 210,2 ± 17,4 | 196,6 ± 18,6 | 40 von 40 |
| ohne Containerart (Rangfolge fehlt) | 1 | 60,6 ± 4,9 | 60,6 ± 4,9 | 57,7 ± 3,8 | 56,7 ± 2,7 | 0 von 40 |
| ohne Containerart (Rangfolge fehlt) | 3 | 261,6 ± 10,4 | 323,7 ± 16,4 | 262,2 ± 11,4 | 255,9 ± 12,7 | 40 von 40 |
| ohne Containerart (Rangfolge fehlt) | 9 | 400,1 ± 32,2 | 1350,7 ± 70,4 | 375,1 ± 14,4 | 365,8 ± 12,1 | 40 von 40 |
| ohne Containerart (Rangfolge fehlt) | 19 | 362,1 ± 60,3 | 3188,4 ± 161,1 | 307,2 ± 16,1 | 313,6 ± 18,1 | 40 von 40 |

Die Kopfzahlen der ersten Messung (Haupteffekte: 25,4 ± 8,5 / 41,7 ± 18,8 / 52,1 ± 30,0 roh bei r = 3 / 9 / 19) liegen innerhalb einer Streuung der zweiten Messung.

**7. Ridge-Stärke des Modells „stark geglättet“** (20 Trainingsmengen, Zufalls-Seeds 500 bis 519; C = 0,03 ist die Wahl der Demo):

| C | AUC | Steigung | ECE (10 Fächer) | Anteil mit p̂ > 1/10 | Regret roh (r = 9) | Regret nach Platt |
|---|---|---|---|---|---|---|
| 0,003 | 0,781 | 4,40 | 0,080 | 1,000 | 372,1 | 225,0 |
| 0,01 | 0,792 | 2,15 | 0,070 | 0,940 | 323,5 | 204,9 |
| 0,03 | 0,804 | 1,43 | 0,042 | 0,788 | 218,0 | 183,5 |
| 0,1 | 0,820 | 1,14 | 0,030 | 0,677 | 166,0 | 155,6 |
| 0,3 | 0,840 | 1,06 | 0,026 | 0,621 | 125,4 | 118,6 |

Mit schwächerem Ridge (größerem C) steigt die AUC (von 0,781 auf 0,840), und die Steigung nähert sich von oben der 1 (von 4,40 auf 1,06): Rangfolge und Kalibrierung lassen sich über Ridge nicht trennen, darum gibt es die kontrolliert verzerrten Modelle. Bei C = 0,003 bekommt jeder Container bei r = 9 eine Maßnahme (Anteil p̂ > 1/10 gleich 1, die Regel „immer Maßnahme“, Streuung 0).

**Beispiel von Hand** (Experiment 1): 30 Container in drei Gruppen mit Vorhersage 1/5, 3/10, 2/5 (je 10 Container) und 0, 2, 7 Containern mit Standgeld (beobachtet 0, 1/5, 7/10); alle Zahlen exakt in Brüchen (`tests/test_hand.py`, von Hand nachgerechnet). Mittleres *p̂* = Anteil Standgeld = 3/10: im Mittel kalibriert, in der Steigung nicht. Brier 17/100 = REL 7/150 − RES 13/150 + UNC 21/100. Gesamtkosten roh (Schwelle 1/(1+r) auf *p̂*) / feste 1/2 / nach beobachteter Häufigkeit (= bestes aller acht Gruppenregeln): r = 3: 11 / 27 / 9; r = 1: 9 / 9 / 5; r = 9: 21 / 81 / 11. Die Verdichtung wirkt in beide Richtungen (bei kleinem r zu wenig Maßnahmen, bei großem zu viele). ECE mit 2 / 4 / 5 / 10 gleich breiten Fächern: 0 / 2/15 / 1/5 / 1/5, bei 5 rechts geschlossenen Fächern 2/15 (die Vorhersage 1/5 liegt genau auf einer Fachgrenze). Schwellen aus der Kostenmatrix (a, b, c₀₀, c₁₁): (1, 3, 0, 0) und (11, 7, 10, 4) geben 1/4, (3, 3, 0, 2) gibt 3/4, (1, 3, 0, −2) gibt 1/6. Bei Container-Gruppen mit der Quote der Wahrheit hoch ½ stellt das Quadrieren der Quote p wieder her (1/5 → 1/17, 4/5 → 16/17); bei r = 9 kostet die unnötige Maßnahme in Gruppe 1 dann 7/17 je Container.

### Hinweise zu den Zahlen

- **Zahl → Test.** Die Tabellen der Abschnitte 2 bis 7 und die Sätze dazu rechnet `tests/test_claims.py` aus `ksw_frozen.json` neu und sucht sie als Text in dieser README (exakt, die Datei ist fest). **Neu gerechnet und mit der Datei verglichen** (`tests/test_frozen.py`, Band mit relativer Toleranz 10⁻⁶) wird nur ein Ausschnitt: Trainingsmenge 0 aller acht Modelle (Güte und Regret), die Rekalibriermengen zu Trainingsmenge 0 für „Haupteffekte“ und „ohne Containerart“ mit m = 100 und 1000, und je drei Wiederholungen des ECE-Bodens für n = 200 und 2000. **Messung dieser README, nicht im Test neu gerechnet:** die übrigen 39 Trainingsmengen, die Gegenprobe mit anderen Zufalls-Seeds (Abschnitt 6) und die Ridge-Zeilen (Abschnitt 7); sie stehen nur als Zeilen der Datei, `python tools/ksw_make_frozen.py` erzeugt sie in etwa zwei Minuten neu. Die Zahlen bei wahrem *p* (Abschnitt 1) rechnet `tests/test_claims.py` neu (Band 0,05 je 1000). Exakt sind nur die Brüche des Beispiels von Hand.
- **Plattform.** Die Datei wurde unter Windows erzeugt. Unter Linux können Gleitkommazahlen in der letzten Stelle abweichen (Bibliotheks-Mathematik, BLAS); die Neuberechnung im Test wird deshalb als Band verglichen, und die Zählungen („in k von 40“) der Datei bleiben fest.
- **Zeilen mit Gleichstand.** „Billiger“ heißt strikt kleiner; Gleichstände zählen nicht (bei r = 1 sind roh und feste 0,5 gleich, die Zählung „0,5 teurer“ ist dort 0).
- **Live gegen vorgerechnet.** Die App rechnet eine Trainingsmenge mit 20.000 Testcontainern; ihre Zahlen weichen von den Mitteln über 40 Trainingsmengen ab (Einzelfall). Die Voreinstellung (Zufalls-Seed 500) ist Trainingsmenge 0 der vorgerechneten Messung.

## Befunde und Korrekturen gegenüber den früheren Messungen

Eine frühere Messung in einer einfacheren Welt (drei Merkmale, 300 Trainingsfälle, nicht dieser Containerstrom) und eine erste Rechnung in diesem Containerstrom (anderer Zufallsstrom: NumPys Generator) lagen vor; die Zahlen dieser README sind neu gemessen. Abweichungen:

- **„Die feste 0,5 kostet bei r = 19 das 112-Fache“ gilt hier nicht.** In der früheren Welt war das Verhältnis der Mittel 112, im Containerstrom ist es 31,0 (Haupteffekte, r = 19; 4,2 und 15,0 bei r = 3 und 9), und der Nenner streut so stark, dass ein Vielfaches nicht angebracht ist; die Demo nennt die Differenz.
- **„Die feste 0,5 ist bei r ≥ 3 die teuerste Regel“** gilt für die richtig spezifizierten und überangepassten Modelle in 40 von 40 Trainingsmengen, nicht für das verschobene Modell bei r = 3 (3 von 40).
- **„Rekalibrierung hilft nur bei richtiger Rangfolge“** gilt sauber nur für eine reine Verzerrung (gleiche AUC). Beim Ridge-Modell fällt der Regret nur von 220,4 auf 184,1; die frühere Zahl (von 205,9 auf 29,3) wird nicht reproduziert, weil Ridge die Rangfolge mitsenkt (Deutung, nicht eigens getestet: mit schwächerem Ridge steigt die AUC und die Steigung nähert sich von oben der 1, Abschnitt 7).
- **Ridge-Stärke.** Die frühere Welt rechnete mit C = 0,003; im Containerstrom macht das die Vorhersagen so flach, dass jeder Container bei r = 9 eine Maßnahme bekommt (Abschnitt 7). Gewählt ist C = 0,03.
- **Kalibrierung im Mittel.** Die erste Rechnung fand im Training höchstens 4·10⁻⁸ (Abbruchtoleranz des Lösers); der Newton-Löser dieser Demo gibt höchstens 2,1e-10.
- **Zufallsstrom.** Zählungen („in k von 40“) und Mittel unterscheiden sich von der ersten Rechnung um Streuung (zum Beispiel „verschoben“, r = 3: 0,5 teurer in 3 statt 7 von 40; Haupteffekte, r = 9, m = 100: Platt 52,8 statt 60,5); alle Aussagen halten in der Richtung.
- **Haupteffekte, r = 9, Platt gegenüber roh:** Platt ist im Mittel billiger (38,8 gegen 41,7), aber nur in 24 von 40 Trainingsmengen; der Unterschied ist klein gegen die Streuung (18,8).

## Quellen

Gelesen im Wortlaut (nach dem Quellenbericht, Stufe W): Elkan, *The Foundations of Cost-Sensitive Learning*, IJCAI 2001 (Schwelle auf die Wahrscheinlichkeit; die 2×2-Kostenmatrix hat einen Freiheitsgrad; Gleichung (2) dort nicht neu geöffnet, die Umformung selbst nachgerechnet); Van Calster u. a., *Calibration: the Achilles heel of predictive analytics*, BMC Medicine 17:230 (2019) (schwache Kalibrierung = Achsenabschnitt ungefähr 0 und Steigung ungefähr 1, Steigung unter 1 heißt zu extreme Risiken; Orientierung 200 Ereignisse und 200 Nichtereignisse); Dimitriadis, Gneiting, Jordan, *Evaluating probabilistic classifiers: Reliability diagrams and score decompositions revisited*, arXiv 2008.03033 (das Binning-Diagramm ist instabil, die CORP-Fassung über PAV ist hier nicht umgesetzt); Sur und Candès, *A modern maximum-likelihood theory for high-dimensional logistic regression* (2019, Verzerrung des Schätzers bei vielen Spalten; hier nur als Hinweis). Nur Abstract oder Suchtreffer (Stufe A oder S): Nixon u. a., *Measuring Calibration in Deep Learning* (ECE hat Schwächen; die Rangfolge von Verfahren hängt vom Maß ab), Niculescu-Mizil und Caruana 2005 (isotonische Rekalibrierung bei knappen Daten überanpassungsanfälliger als Platt, Stufe S). Aus dem Gedächtnis (Stufe G, nicht nachgelesen): Platt 1999, Zadrozny und Elkan, Murphy 1973, Brier 1950. Guo u. a. (ICML 2017) betrifft tiefe Netze; nach dem Quellenbericht gilt der Befund für ein Logit mit Achsenabschnitt nicht, hier ist es nur als Gegenstück genannt.

## Ehrliche Grenzen

- **Ein Generator, eine Frist (7 Tage), eine Trainingsgröße (500), ein Satz Merkmale.** Andere Fristen, andere Trainingsgrößen und andere Anteile von Standgeld sind nicht gemessen; ob andere Konfigurationen dazwischen liegen (zum Beispiel beim Verhältnis aus 0,5 und 1/(1+r)), ist nicht geprüft.
- **Kosten angenommen** (r = 1, 3, 9, 19), keine Messung am Terminal; nur das einfache Schema (richtig entschieden kostet 0); ein Nutzen bei richtiger Maßnahme steht nur im Beispiel von Hand.
- **Das Vehikel ist synthetisch:** die Merkmale, die lognormale Verweilzeit und die Größenordnungen sind erfunden; die wahre Wahrscheinlichkeit gibt es nur, weil der Generator sie kennt. Bei echten Daten gäbe es nur eine Rekalibriermenge.
- **Nicht gemessen:** Kalibrierung je Teilgruppe (zum Beispiel je Containerart; „ohne Containerart“ ist darin sicher schief), Mehrklassen, Temperature Scaling, Beta-Kalibrierung, geglättete Platt-Zielwerte, das CORP-Diagramm mit Unsicherheitsband, MCB und DSC statt REL und RES, AUC nach isotonischer Rekalibrierung (Blöcke erzeugen Gleichstände), die isotonische Stufenfunktion statt linearer Interpolation, der Achsenabschnitt bei auf 1 festgehaltener Steigung.
- **Ridge-Stärke:** nur die fünf Werte von Abschnitt 7; ein Modell mit richtiger Rangfolge und falscher Skala entsteht hier erst über die kontrollierte Verzerrung, nicht über Ridge.
- **Zufallsstrom:** Zählungen und Zahlen gelten für diesen Strom (SplitMix64); eine zweite Zufalls-Seed-Folge gibt es nur für drei Modelle und die Regeln roh, 0,5, Platt, isotonisch (Abschnitt 6).
- **Live gegen vorgerechnet:** die App rechnet eine Trainingsmenge mit 20 000 Testcontainern (die vorgerechneten Mittel nutzen 100 000); die App zeigt die Mittel über 40 Trainingsmengen nur aus der Datei.
- **Namensfalle HC3 und Ridge-Wahl** sind nicht Thema: die λ-Wahl bei Ridge liegt bei `parameterwahl-demo` und `tikhonov-demo` (Verweis, hier nichts nachgerechnet).

## Verifikation

- **Kern** (`tests/test_core.py`, `tests/test_hand.py`): Mini-Instanzen von Hand (erwartete Kosten, Regret, PAV, isotonische Interpolation, Logit mit einer 0/1-Spalte, Platt, ECE mit Randwerten, AUC, Brier-Zerlegung, abgestimmte Schwelle) und das Beispiel von Hand in Brüchen (Brier 17/100, Kosten 11 / 27 / 9, 9 / 9 / 5, 21 / 81 / 11, ECE 0 / 2/15 / 1/5 / 1/5, Schwellen 1/4, 1/4, 3/4, 1/6, τ-Quantil über 1599 Paare).
- **Orakel** (`tests/test_oracle_kalibrierung_schwelle.py`): PAV gegen die Aufzählung aller Blockzerlegungen in Brüchen (alle 128 Folgen der Länge 7) und gegen scikit-learn `IsotonicRegression`; isotonische Vorhersage gegen scikit-learn mit Gleichständen und Kappung; Platt und Logit mit Ridge gegen `LogisticRegression` (Löser `newton-cholesky`, kein BFGS); Regret über die Formel Σ |(1+*r*)*p* − 1| gegen die direkte Kostenrechnung; Schwelle 1/(1+*r*) gegen die Aufzählung eines Schwellenrasters; Brier-Zerlegung in Brüchen; ECE gegen eine Schleife mit `np.digitize`; AUC gegen scikit-learn; abgestimmte Schwelle gegen die Aufzählung; Platt stellt eine reine Logit-Verzerrung wieder her; Schwelle gegen `scipy.stats.bernoulli.ppf` (1482 Paare ohne Gleichstand).
- **Eigenschaften** (`tests/test_eigenschaften_kalibrierung_schwelle.py`): Regret nie negativ und 0 bei wahrem *p*; bei r = 1 sind 0,5 und Kostenschwelle dieselbe Regel; Platt ist invariant gegen logit-affine Verzerrungen; AUC gleich nach streng monotoner Verzerrung; Zeilenreihenfolge ändert weder Platt noch PAV noch Maße; PAV erhält die Summe; Kalibrierung im Mittel im Training auch mit Ridge (und die Gegenprobe mit bestraftem Achsenabschnitt); mehr Rekalibriercontainer verkleinern die Streuung der abgestimmten Schwelle; gleicher Zufalls-Seed, gleiche Daten.
- **Generator und Modelle** (`tests/test_generator.py`, `tests/test_models.py`): Kundeneffekte gleich dem NumPy-Strom der Quelle, wahre Wahrscheinlichkeit von Hand für zwei Container, Anteile und Verteilungen gegen Messwerte der Quelle als Band, Wertebereich, Score-Gleichung, verzerrte Modelle als Logit-Transformation, Randfälle (Art ohne Ereignis, weniger Container als Spalten).
- **Vorgerechnet** (`tests/test_frozen.py`): Aufbau, kein negativer Regret, Zählungen aus den Einzelwerten, Abgleich eines Ausschnitts mit einer Neuberechnung (Band), Aufruf `tools/ksw_make_frozen.py --check`.
- **Claims** (`tests/test_claims.py`): die Zahlen der Abschnitte 1 bis 7 und der Hinweise stehen in dieser README (Text wird gesucht).
- **Verdikte, Presets, Randfälle** (`tests/test_verdicts.py`, `tests/test_presets.py`, `tests/test_edge_cases.py`): jede Zahl im Text an handgesetzten Zeilen; je Preset die erwartete Zeile (ein Preset ohne Effekt meldet keinen Effekt: bei r = 1 meldet die App keinen Unterschied zwischen 0,5 und 1/(1+r)); konstante Vorhersage, nur eine Klasse, Gleichstand an der Schwelle, Fachgrenzen, r ≤ 0.
- **Zufall** (`tests/test_rng.py`): Folge der Klasse gleich der des Feldes, Referenzwerte von SplitMix64, Reihen für Logarithmus und Kosinus gegen die exakte Fassung, Momente. Einige Tests ziehen ihre Zufallsfälle mit NumPys `default_rng` und festen Zufalls-Seeds (nur in den Tests, als unabhängiger Weg).
- **Oberfläche und Layout** (`tests/test_app.py`, `tests/test_app_inhalt.py`, `tests/test_layout.py`, `tests/test_footer.py`, `tests/test_runtime_imports.py`): AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum), Inhalte und Verdikte der App, Schnellstart oben in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben, jede Regleränderung hebt die Markierung auf, Footer, Laufzeit-Importe.
- **Rechenzeit.** Live-Rechnung der App je Eingabe 0,02 bis 0,06 Sekunden (Trainingsmenge, Rekalibrierung, 20 000 Testcontainer; gemessen für drei Modelle und m = 100, 2000, 10 000), die Seite mit allen Abbildungen im AppTest rund 2 Sekunden; das Erzeugen der vorgerechneten Datei dauert rund zwei Minuten.
- **Linux-Lauf** (Docker, Python 3.12.14 wie die CI, frische Installation aus `requirements-dev.txt`, `tools/demo_linux_check.py` des Website-Repos; Stand des Baus, 2026-11-16): alle 216 Tests grün in 24 Sekunden.
- **Testzahl:** 216 Tests (`pytest --collect-only -q tests`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: fünf Experimente, Kernfragen, Grenzen, Formeln |
| `ksw_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `ksw_core.py` | Kern: Logit (Newton), Platt, PAV, isotonische Rekalibrierung, Kosten und Regret, ECE, Brier-Zerlegung, AUC |
| `ksw_evaluation.py` | Live-Rechnung, Kennzahlen aus der vorgerechneten Datei, alle Verdikte als reine Funktionen |
| `ksw_frozen.py`, `ksw_frozen.json` | Laden, Neuberechnen und Abgleichen der vorgerechneten Messungen; die Datei selbst |
| `ksw_generator.py` | Container-Generator (aus `verweilzeit-demo` übernommen, mit SplitMix64) und wahre Wahrscheinlichkeit |
| `ksw_hand.py` | Beispiel von Hand in Brüchen |
| `ksw_layout.py` | Fester Kopf (Stylesheet) |
| `ksw_measure.py` | Messungen (Trainingsmengen, Rekalibriermengen, ECE-Boden, Ridge-Stärke) |
| `ksw_models.py` | Die acht Modelle |
| `ksw_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `ksw_rng.py` | SplitMix64 mit exakten Normalzahlen (kopiert aus `plan-stoerung-demo`) |
| `ksw_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest und scikit-learn (nur für die Orakel-Tests) |
| `tools/ksw_make_frozen.py` | erzeugt `ksw_frozen.json` (`--check` gleicht einen Ausschnitt ab) |
| `tests/` | Kern, Beispiel von Hand, Orakel, Eigenschaften, Generator, Modelle, vorgerechnete Datei, Claims, Verdikte, Presets, Randfälle, Zufall, AppTest, Layout, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **PDF-Export** der Ergebnisse (andere Demos haben ihn; hier steht der Platz für die fünf Experimente).
- **Standgeld-Frist als Regler** (5, 7, 10 Tage): nicht gemessen.
- **Teilung in zwei Seiten** (Schwelle, Kalibrierung): eine Seite mit fünf Experimenten.
- **CORP-Diagramm mit Unsicherheitsband** statt Fächern (Dimitriadis u. a.): nicht umgesetzt, die Fächer sind wählbar, um die Abhängigkeit von der Fächerzahl zu zeigen.
- **Temperature Scaling, Beta-Kalibrierung, Mehrklassen, Kalibrierung je Teilgruppe.**
- **Echtdaten:** der Containerstrom ist erfunden.
- **Ridge-Stärke und λ-Wahl:** Verweis auf `parameterwahl-demo` und `tikhonov-demo`.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
