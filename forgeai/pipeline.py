"""
ForgeAI – Main pipeline orchestrator.
Ties all stages together: intake → generate → eval → judge → critic → report.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

load_dotenv()
console = Console()


def run_pipeline(
    brief: str,
    doc_folder: str,
    output_dir: str,
    interactive: bool = True,
    max_iterations: int = 3,
    tournament: bool = False,
    target_eval_cases: int = 22,
) -> dict:
    """
    Full ForgeAI pipeline.
    Returns paths to all produced artifacts.
    """
    from forgeai.core import llm
    from forgeai.core.intake import run_intake
    from forgeai.core.generator import run_generator
    from forgeai.core.eval_generator import run_eval_generator
    from forgeai.core.judge import run_judge
    from forgeai.core.critic import run_critic_optimizer
    from forgeai.core.reporter import run_reporter

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()

    console.print(Panel.fit(
        f"[bold green]ForgeAI Pipeline Starting[/bold green]\n"
        f"Brief: {brief[:80]}...\n"
        f"Docs: {doc_folder}",
        border_style="green",
    ))

    # ── Stage 1: Intake ───────────────────────────────────────────────────
    console.rule("[bold]Stage 1: Intake & Requirements[/bold]")
    req = run_intake(
        brief=brief,
        doc_folder=doc_folder,
        interactive=interactive,
        output_path=str(out / "requirements.json"),
    )
    console.print(f"[green]✓ Groundability OK: {req.groundability_ok}[/green]")

    if not req.groundability_ok:
        console.print(f"[yellow]⚠ Issues: {req.groundability_issues}[/yellow]")

    # Budget check
    if llm.usage.over_budget():
        console.print("[red]⚠ Token budget exceeded after intake. Stopping.[/red]")
        return {"error": "budget_exceeded", "stage": "intake"}

    # ── Stage 2: Agent Generation ─────────────────────────────────────────
    console.rule("[bold]Stage 2: Agent Generation[/bold]")
    agent, candidates = run_generator(
        req=req,
        doc_folder=doc_folder,
        output_path=str(out / "agent_definition.json"),
        tournament=tournament,
    )

    if tournament and len(candidates) > 1:
        # Save all candidates
        for i, c in enumerate(candidates, 1):
            cand_path = out / f"candidate_v{i}.json"
            cand_path.write_text(c.model_dump_json(indent=2))
        console.print(f"[dim]Saved {len(candidates)} tournament candidates.[/dim]")

    # Budget check
    if llm.usage.over_budget():
        console.print("[red]⚠ Token budget exceeded after generation. Stopping.[/red]")
        return {"error": "budget_exceeded", "stage": "generation"}

    # ── Stage 3: Eval Generation ──────────────────────────────────────────
    console.rule("[bold]Stage 3: Eval Set Generation[/bold]")
    eval_set = run_eval_generator(
        req=req,
        doc_folder=doc_folder,
        agent_name=agent.name,
        output_path=str(out / "eval_set.json"),
        target_total=target_eval_cases,
    )
    console.print(f"[green]✓ Generated {len(eval_set.cases)} eval cases.[/green]")

    # Budget check
    if llm.usage.over_budget():
        console.print("[red]⚠ Token budget exceeded after eval generation. Stopping.[/red]")
        return {"error": "budget_exceeded", "stage": "eval_generation"}

    # ── Stages 4+5: Judge + Critic-Optimizer Loop ─────────────────────────
    console.rule("[bold]Stages 4–5: Judge & Critic-Optimizer Loop[/bold]")

    iter_dir = str(out / "iterations")

    final_agent, all_results = run_critic_optimizer(
        agent=agent,
        eval_set=eval_set,
        doc_folder=doc_folder,
        judge_fn=run_judge,
        max_iterations=max_iterations,
        output_dir=iter_dir,
    )

    # Save final agent
    final_agent_path = out / "agent_final.json"
    final_agent_path.write_text(final_agent.model_dump_json(indent=2))

    # ── Stage 6: Report ───────────────────────────────────────────────────
    console.rule("[bold]Stage 6: Reporting[/bold]")
    report = run_reporter(
        agent=final_agent,
        results=all_results,
        output_path=str(out / "final_report.json"),
    )

    elapsed = time.perf_counter() - start_time

    console.print(Panel.fit(
        f"[bold green]ForgeAI Pipeline Complete[/bold green]\n"
        f"Agent: {final_agent.name}\n"
        f"Best Score: {report.final_score:.1%} (iteration {report.best_iteration})\n"
        f"Tokens used: {llm.usage.total_tokens:,}\n"
        f"Time: {elapsed:.1f}s\n"
        f"Output: {output_dir}",
        border_style="green",
    ))

    return {
        "agent_name": final_agent.name,
        "best_score": report.final_score,
        "best_iteration": report.best_iteration,
        "total_eval_cases": len(eval_set.cases),
        "output_dir": output_dir,
        "artifacts": {
            "requirements": str(out / "requirements.json"),
            "agent_definition": str(out / "agent_definition.json"),
            "agent_final": str(out / "agent_final.json"),
            "eval_set": str(out / "eval_set.json"),
            "final_report": str(out / "final_report.json"),
            "final_report_md": str(out / "final_report.md"),
            "iterations_dir": iter_dir,
        },
        "usage": llm.usage.summary(),
        "elapsed_s": round(elapsed, 2),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="ForgeAI – The Agent That Builds Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m forgeai.pipeline --brief "HR policy assistant" --docs demos/hr_assistant/docs --out runs/hr_run
  python -m forgeai.pipeline --brief "Ship spares assistant" --docs demos/ship_spares/docs --out runs/ship_run --tournament
  python -m forgeai.pipeline --brief "..." --docs ./docs --out ./output --no-interactive
        """,
    )
    parser.add_argument("--brief", required=True, help="One-paragraph description of the agent to build")
    parser.add_argument("--docs", required=True, help="Path to folder containing documents")
    parser.add_argument("--out", required=True, help="Output directory for all artifacts")
    parser.add_argument("--iterations", type=int, default=3, help="Max critic-optimizer iterations (default: 3)")
    parser.add_argument("--eval-cases", type=int, default=22, help="Target number of eval cases (default: 22)")
    parser.add_argument("--tournament", action="store_true", help="Run prompt tournament (stretch goal)")
    parser.add_argument("--no-interactive", action="store_true", help="Skip clarifying questions (non-interactive mode)")

    args = parser.parse_args()

    result = run_pipeline(
        brief=args.brief,
        doc_folder=args.docs,
        output_dir=args.out,
        interactive=not args.no_interactive,
        max_iterations=args.iterations,
        tournament=args.tournament,
        target_eval_cases=args.eval_cases,
    )

    if "error" in result:
        console.print(f"[red]Pipeline failed: {result}[/red]")
        sys.exit(1)

    print("\nArtifact locations:")
    for key, path in result.get("artifacts", {}).items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
