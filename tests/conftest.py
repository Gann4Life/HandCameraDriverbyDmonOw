"""Puts the repo root on sys.path so tests import the app's modules as the app does."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
