"""
run_ingestion.py — Convenience script to run ingestion from the repo root.

Usage:
    python run_ingestion.py

This is equivalent to:
    python -m src.ingestion
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from src.ingestion import run_ingestion

if __name__ == "__main__":
    run_ingestion()
