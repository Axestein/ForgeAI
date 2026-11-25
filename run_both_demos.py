#!/usr/bin/env python3
"""
run_both_demos.py – Runs both ForgeAI demo pipelines sequentially.
Produces committed artifacts in runs/hr_assistant/ and runs/ship_spares/.
"""
import sys
import subprocess
from pathlib import Path

root = Path(__file__).parent

def run(script: str, name: str):
    print(f"\n{'='*70}")
    print(f"Starting: {name}")
    print(f"{'='*70}\n")
    result = subprocess.run(
        [sys.executable, script],
        cwd=str(root),
    )
    return result.returncode == 0

success1 = run(str(root / "demos" / "hr_assistant" / "run_demo.py"), "HR Assistant Demo")
success2 = run(str(root / "demos" / "ship_spares" / "run_demo.py"), "Ship Spares Demo")

print("\n" + "="*70)
print("DEMO RUN SUMMARY")
print("="*70)
print(f"HR Assistant: {'✅ PASSED' if success1 else '❌ FAILED'}")
print(f"Ship Spares:  {'✅ PASSED' if success2 else '❌ FAILED'}")
print("\nArtifacts in: runs/hr_assistant/ and runs/ship_spares/")
print("View report:  streamlit run app.py")
