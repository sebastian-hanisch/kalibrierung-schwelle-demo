"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ksw_hand as H  # noqa: E402


@pytest.fixture
def hand_groups():
    """Drei Gruppen zu je 10 Containern: Vorhersage 1/5, 3/10, 2/5, davon mit Standgeld 0, 2, 7 (beobachtet 0, 1/5, 7/10)."""
    return H.DEFAULT_GROUPS
