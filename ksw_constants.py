"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

RATIOS = (1, 3, 9, 19)                          # Kostenverhältnis r = Kosten verpasst : Kosten unnötig (gesetzt, nicht gemessen)
DEFAULT_RATIO = 9
CAL_SIZES = (100, 300, 1000, 2000, 3000, 10_000)  # Rekalibriermengen (Container); die vorgerechneten Reihen gelten für genau diese Größen
DEFAULT_CAL = 2000
DEFAULT_MODEL = "haupt"
SEED_MAX, DEFAULT_SEED = 99_999, 500            # Zufalls-Seed der Trainingsmenge; die vorgerechneten Reihen nutzen 500 bis 539
DEFAULT_BINNING = "10 gleich breit"
N_LIVE_TEST = 20_000                            # Testcontainer der Live-Rechnung (die vorgerechneten Reihen nutzen 100 000)
N_TRAIN = 500                                   # Trainingscontainer
N_CAL = 2000                                    # Rekalibriercontainer der vorgerechneten Hauptmessung
POPULATION_N, POPULATION_SEED = 100_000, 7      # Container für die Kosten bei wahrem p (Experiment 2)
LIVE_CAL_SEED = 20_000                          # Zufalls-Seed der Rekalibriermenge der Live-Rechnung = LIVE_CAL_SEED + Zufalls-Seed der Trainingsmenge
LIVE_TEST_SEED = 999

COLOR_ORACLE = "#2F6B65"
COLOR_RAW = "#c77700"
COLOR_FIXED = "#b03a3a"
COLOR_PLATT = "#2a6fb0"
COLOR_ISO = "#7a3fb0"
COLOR_TUNED = "#5b6470"
COLOR_BAND = "rgba(76,120,168,0.18)"

PRESETS = {
    "Standgeld, Logit mit allen Merkmalen": dict(r=9, model="haupt", m=2000, seed=DEFAULT_SEED, bins=DEFAULT_BINNING),
    "Von Hand (30 Container)": dict(r=3, model="haupt", m=2000, seed=DEFAULT_SEED, bins=DEFAULT_BINNING),
    "Verdichtet (Steigung 0,5)": dict(r=9, model="verdichtet", m=2000, seed=DEFAULT_SEED, bins=DEFAULT_BINNING),
    "Überkonfident (Steigung 2)": dict(r=9, model="ueberkonfident", m=2000, seed=DEFAULT_SEED, bins=DEFAULT_BINNING),
    "Ohne Containerart (Rangfolge fehlt)": dict(r=9, model="ohne_art", m=2000, seed=DEFAULT_SEED, bins=DEFAULT_BINNING),
    "Wenig Kalibrierdaten (m = 100)": dict(r=9, model="haupt", m=100, seed=DEFAULT_SEED, bins=DEFAULT_BINNING),
}
PRESET_HELP = {
    "Standgeld, Logit mit allen Merkmalen": "Ein Logit auf allen Merkmalen, Kostenverhältnis r = 9 (ein verpasstes Standgeld kostet das Neunfache einer unnötigen Maßnahme), Rekalibriermenge 2000 Container.",
    "Von Hand (30 Container)": "Kostenverhältnis r = 3: im Beispiel von Hand (30 Container in drei Gruppen) ist die Schwelle 1/4; die Regler der Standgeld-Rechnung bleiben auf ihren Voreinstellungen.",
    "Verdichtet (Steigung 0,5)": "Dasselbe Logit, danach die Vorhersage zur Mitte gedrückt (Achsenabschnitt 0, Steigung 0,5 im Logit). Die Rangfolge der Container bleibt, nur die Wahrscheinlichkeiten stimmen nicht mehr.",
    "Überkonfident (Steigung 2)": "Dasselbe Logit, danach die Vorhersage zu den Rändern gezogen (Steigung 2 im Logit). Die Rangfolge bleibt, die Wahrscheinlichkeiten sind zu extrem.",
    "Ohne Containerart (Rangfolge fehlt)": "Das Logit ohne die drei Art-Spalten: die Wahrscheinlichkeiten stimmen im Mittel gut, aber die Containerarten lassen sich nicht mehr trennen.",
    "Wenig Kalibrierdaten (m = 100)": "Rekalibriermenge von nur 100 Containern (im Mittel rund 23 mit Standgeld): zeigt, dass Rekalibrieren mit wenig Daten schaden kann.",
}
