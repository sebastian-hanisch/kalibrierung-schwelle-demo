"""Rechnet die Messungen der Demo vor und friert sie in `ksw_frozen.json` ein; `--check` rechnet einen Ausschnitt neu und vergleicht ihn mit der Datei.

Aufruf (Wurzelordner der Demo):  python tools/ksw_make_frozen.py            schreibt ksw_frozen.json (mehrere Minuten, nur numpy/scipy)
                                  python tools/ksw_make_frozen.py --check    prüft, dass die Datei zum Code passt (Sekunden), Exit-Code 1 bei Abweichung
Die Zahlen der README stammen aus dieser Datei; ein Test (tests/test_frozen.py) ruft den Abgleich auf."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ksw_frozen as F  # noqa: E402


def main(argv) -> int:
    if "--check" in argv:
        bad = F.verify_subset(F.load())
        print("Abgleich ok" if not bad else "\n".join(bad))
        return 1 if bad else 0
    frozen = F.build()
    F.write(frozen)
    print(f"geschrieben: {F.PATH} ({F.PATH.stat().st_size} Bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
