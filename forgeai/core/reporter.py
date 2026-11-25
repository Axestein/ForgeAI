"""
Stage 6 – Reporter
Generates an honest final report: scorecards per iteration,
what still fails and why, honest quality assessment.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import List

from forgeai.core.models import AgentDefinition, FinalReport, IterationResult
from forgeai.core import llm


def _build_scorecard(results: List[IterationResult]) -> str:
    lines = []
    lines.append("| Iteration | Score | Passed | Total | Changes Made |")
    lines.append("|-----------|-------|--------|-------|--------------|")
    for r in results:
        changes = r.changes_made[:60] + "..." if len(r.changes_made) > 60 else r.changes_made
        lines.append(f"| {r.iteration} | {r.score:.1%} | {r.passed} | {r.total} | {changes or '(initial run)'} |")
    return "\n".join(lines)


def _remaining_failures(best_result: IterationResult) -> List[str]:
    failures = []
    for r in best_result.results:
        if not r.passed:
            cat = r.failure_category or "unknown"
            failures.append(f"[{cat}] {r.question[:80]}: {r.judge_reasoning[:100]}")
    return failures


def _generate_honest_assessment(
    agent_name: str,
    results: List[IterationResult],
    remaining: List[str],
) -> str:
    best = max(results, key=lambda r: r.score)
    worst = min(results, key=lambda r: r.score)

    failure_cats = Counter()
    for r in best.results:
        if not r.passed and r.failure_category:
            failure_cats[r.failure_category] += 1

    prompt = f"""Write an honest, professional quality assessment for an AI agent that was auto-generated.

Agent name: {agent_name}
Best score: {best.score:.1%} (iteration {best.iteration})
Starting score: {results[0].score:.1%}
Improvement: {best.score - results[0].score:+.1%}
Remaining failure categories: {dict(failure_cats)}
Number of remaining failures: {len(remaining)}

Sample remaining failures:
{chr(10).join(remaining[:5])}

Write 2-3 paragraphs:
1. What the agent does well and why the score is what it is
2. What still fails and the honest root causes
3. What a human engineer should do next to improve it further

Be honest. If the score is 78%, say it's 78% and explain why that's not 100%.
Do NOT claim success where there are known failures."""

    r = llm.chat(
        messages=[{"role": "user", "content": prompt}],
        system="You write honest technical quality reports. Never oversell.",
        temperature=0.3,
        max_tokens=600,
    )
    return r.text.strip()


def run_reporter(
    agent: AgentDefinition,
    results: List[IterationResult],
    output_path: str | None = None,
) -> FinalReport:
    """Generate and save the final report."""
    print("[Reporter] Generating final report...")

    best = max(results, key=lambda r: r.score)
    remaining = _remaining_failures(best)

    honest_assessment = _generate_honest_assessment(
        agent.name, results, remaining
    )

    from forgeai.core.llm import usage
    report = FinalReport(
        agent_name=agent.name,
        iterations=results,
        best_iteration=best.iteration,
        final_score=best.score,
        remaining_failures=remaining,
        honest_assessment=honest_assessment,
        usage_summary=usage.summary(),
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(report.model_dump_json(indent=2))

    # Also write human-readable markdown
    md_path = output_path.replace(".json", ".md") if output_path else None
    md = _render_markdown(report, results)
    if md_path:
        Path(md_path).write_text(md)
        print(f"[Reporter] Report saved → {md_path}")

    return report


def _render_markdown(report: FinalReport, results: List[IterationResult]) -> str:
    scorecard = _build_scorecard(results)
    failures_md = "\n".join(f"- {f}" for f in report.remaining_failures[:10]) or "_None remaining._"

    return f"""# ForgeAI Final Report: `{report.agent_name}`

## Summary

| Metric | Value |
|--------|-------|
| Best Score | **{report.final_score:.1%}** |
| Best Iteration | {report.best_iteration} |
| Total Iterations | {len(results)} |
| Total Tokens Used | {report.usage_summary.get('total_tokens', 'N/A')} |
| Budget Remaining | {report.usage_summary.get('budget_remaining', 'N/A')} |

## Per-Iteration Scorecard

{scorecard}

## Honest Assessment

{report.honest_assessment}

## Remaining Failures (Best Iteration)

{failures_md}

## Usage Breakdown

```json
{json.dumps(report.usage_summary, indent=2)}
```
"""
