#!/usr/bin/env python3
"""
Demo Run 1 – HR Policy Assistant
Runs ForgeAI end-to-end on the HR policy brief.

Synthesis prompt used to create the demo documents:
"Generate a realistic corporate HR policy manual for a shipping company's shore staff,
covering leave policies (annual, sick, maternity/paternity, emergency, unpaid),
travel and reimbursement policy (domestic and international, per diems, hotels,
ground transport, expense submission deadlines, non-reimbursable expenses),
code of conduct (core principles, harassment policy, social media, gifts, disciplinary process),
performance management (review cycle, rating scale, bonus), and benefits (health insurance,
PF, gratuity, flexible working). Make values realistic for an Indian company (₹ per diems, etc.)
Also generate a companion travel booking procedures document with approved vendor lists,
hotel contracts, and forex guidelines."
"""
import os
import sys
from pathlib import Path

# Add project root to path  (demos/hr_assistant → demos → companyAssignment)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

BRIEF = (
    "I need an assistant that answers HR policy questions for shore staff. "
    "It should cover leave policies (annual, sick, maternity, emergency), travel and "
    "reimbursement procedures (per diems, hotel limits, expense submission), and the "
    "company code of conduct. It must refuse questions about vessel operations, cargo, "
    "or any topic not in the HR policy documents. The audience is shore-based employees "
    "who need quick, accurate policy answers without having to read the full manual."
)

DOC_FOLDER = str(Path(__file__).parent / "docs")
OUTPUT_DIR = str(Path(__file__).parent.parent.parent / "runs" / "hr_assistant")


def main():
    from forgeai.pipeline import run_pipeline

    print("=" * 70)
    print("ForgeAI Demo Run 1: HR Policy Assistant")
    print("=" * 70)
    print(f"Brief: {BRIEF}")
    print(f"Docs: {DOC_FOLDER}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    result = run_pipeline(
        brief=BRIEF,
        doc_folder=DOC_FOLDER,
        output_dir=OUTPUT_DIR,
        interactive=True,   # Will ask clarifying questions
        max_iterations=3,
        tournament=False,
        target_eval_cases=22,
    )

    if "error" in result:
        print(f"\nERROR: {result}")
        sys.exit(1)

    print(f"\n✅ Demo 1 complete!")
    print(f"   Score: {result['best_score']:.1%}")
    print(f"   Output: {result['output_dir']}")
    print(f"\nTo chat with the agent:")
    print(f"  python -m forgeai.runner.chat --agent {result['artifacts']['agent_final']} --docs {DOC_FOLDER}")


if __name__ == "__main__":
    main()
