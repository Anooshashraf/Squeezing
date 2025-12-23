#!/usr/bin/env python3
"""
Launcher for visualizations
"""
import subprocess
import os
import sys

# Get current directory
project_root = os.path.dirname(os.path.abspath(__file__))
viz_folder = os.path.join(project_root, 'visualizations')

print("🎬 Launching quantum animation...")
os.chdir(viz_folder)  # Change to visualizations folder
subprocess.run([sys.executable, "animation_simulation.py"])