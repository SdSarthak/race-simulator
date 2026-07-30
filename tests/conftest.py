"""Shared fixtures. Tests are deterministic, need no display and download nothing."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import STATE_DIM, ACTION_DIM   # noqa: E402
from track import Track                    # noqa: E402
from car import Car                        # noqa: E402


@pytest.fixture
def track():
    return Track()


@pytest.fixture
def oval():
    return Track("oval")


@pytest.fixture
def car(track):
    return Car(track.start_pos, track.start_angle)


@pytest.fixture
def dims():
    return STATE_DIM, ACTION_DIM
