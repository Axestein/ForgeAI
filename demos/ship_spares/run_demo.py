#!/usr/bin/env python3
"""
Demo Run 2 – Ship Spares Procurement Assistant
Runs ForgeAI end-to-end on the ship spares procurement brief.

Synthesis prompts used to create the demo documents:
1. CSV catalogue:
   "Generate a realistic ship spare parts catalogue as a CSV with 30 rows covering
   propulsion, electrical, hull, engine, navigation, and safety categories. Include
   columns: part_number, description, category, unit, unit_price_usd, lead_time_days,
   supplier_code, stock_status, min_order_qty, notes. Use realistic marine part names
   and prices."

2. Procurement policy:
   "Generate a maritime procurement policy document covering: approval authority thresholds,
   approved supplier list (8 suppliers with codes), standard and emergency ordering processes,
   payment terms per supplier, quality standards (IMO/SOLAS compliance, certificates required),
   warranty/return policy, inventory management rules, and prohibited practices."

3. Technical specs:
   "Generate a ship technical specifications document covering maintenance intervals and
   replacement schedules for propulsion systems, electrical systems, hull coatings,
   engine consumables, navigation equipment calibration, and safety equipment service."
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # companyAssignment/

BRIEF = (
    "I need a procurement assistant for ship maintenance teams. It helps engineers "
    "look up spare parts by part number or description, check prices and lead times, "
    "find the right supplier, and understand procurement approval thresholds and payment terms. "
    "It should also answer questions about maintenance intervals from our technical specs. "
    "It must refuse questions about HR, crew welfare, cargo operations, or anything not in "
    "the procurement and technical documents. The audience is Chief Engineers and Fleet "
    "Managers who need fast, accurate information to avoid vessel downtime."
)

DOC_FOLDER = str(Path(__file__).parent / "docs")
OUTPUT_DIR = str(Path(__file__).parent.parent.parent / "runs" / "ship_spares")


def main():
    from forgeai.pipeline import run_pipeline

    print("=" * 70)
    print("ForgeAI Demo Run 2: Ship Spares Procurement Assistant")
    print("=" * 70)
    print(f"Brief: {BRIEF}")
    print(f"Docs: {DOC_FOLDER}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    result = run_pipeline(
        brief=BRIEF,
        doc_folder=DOC_FOLDER,
        output_dir=OUTPUT_DIR,
        interactive=True,
        max_iterations=3,
        tournament=False,
        target_eval_cases=22,
    )

    if "error" in result:
        print(f"\nERROR: {result}")
        sys.exit(1)

    print(f"\n✅ Demo 2 complete!")
    print(f"   Score: {result['best_score']:.1%}")
    print(f"   Output: {result['output_dir']}")
    print(f"\nTo chat with the agent:")
    print(f"  python -m forgeai.runner.chat --agent {result['artifacts']['agent_final']} --docs {DOC_FOLDER}")


if __name__ == "__main__":
    main()
