#!/usr/bin/env python3
"""
Launcher script - run from main project folder
"""
import sys
import os

# Add simulations folder to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'simulations'))

# Now import and run simulations
from main import run_complete_simulation  # Adjust based on your main.py

if __name__ == "__main__":
    print("🚀 Launching simulations from project root...")
    results = run_complete_simulation()
    print("✅ Simulations complete!")