"""
Stage 5 – Critic–Optimizer Loop
Runs evals → classifies failures → applies targeted fixes → re-runs.
Maximum 3 iterations. Every change is logged.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import List, Tuple

from forgeai.core import llm
from forgeai.core.models import (
    AgentDefinition,
    EvalSet,
    IterationResult,
)

_CRITIC_SYSTEM = """You are ForgeAI's self-critic.
You receive an agent definition and a list of test failures, categorized by failure type.
Your job: propose a minimal, targeted fix to the system prompt to address these failures.
Do NOT rewrite the whole prompt. Apply surgical changes.
Return your response as JSON: {
  "analysis": "...",
  "failure_categories_addressed": ["..."],
  "revised_system_prompt": "..."
}"""


def _summarize_failures(iteration: IterationResult) -> dict:
    """Summarize failures by category for the critic."""
    failures = [r for r in iteration.results if not r.passed]
    counter = Counter(r.failure_category for r in failures if r.failure_category)

    failure_details = []
    for r in failures[:8]:  # Show up to 8 examples
        failure_details.append({
            "question": r.question,
            "agent_answer_excerpt": r.agent_answer[:200],
            "failure_category": r.failure_category,
            "judge_reasoning": r.judge_reasoning,
        })

    return {
        "total_failures": len(failures),
        "by_category": dict(counter),
        "examples": failure_details,
    }


def _apply_targeted_fix(
    agent: AgentDefinition,
    failure_summary: dict,
    iteration: int,
) -> Tuple[AgentDefinition, str, List[str]]:
    """Ask the LLM critic to propose a targeted fix and return the updated agent."""

    dominant_failure = max(
        failure_summary["by_category"],
        key=lambda k: failure_summary["by_category"][k],
        default="prompt_defect",
    )

    fix_hints = {
        "retrieval_miss": "The agent is not finding relevant content. Add instructions to rephrase the question if the first search yields nothing, and to search for related synonyms.",
        "hallucination": "The agent is fabricating information. Strengthen the grounding instructions: add 'If the answer is not explicitly in the documents, say so.'",
        "prompt_defect": "The agent is finding content but not extracting the right facts. Add more specific instructions on how to structure the answer.",
        "wrong_refusal": "The agent is refusing questions it should answer. Clarify the scope boundaries and explicitly list topics that ARE in scope.",
    }

    hint = fix_hints.get(dominant_failure, "Review the system prompt for clarity.")

    prompt = f"""Current agent system prompt:
{agent.system_prompt}

Test failure analysis (iteration {iteration}):
{json.dumps(failure_summary, indent=2)}

Dominant failure type: {dominant_failure}
Fix hint: {hint}

Apply a minimal, targeted fix to the system prompt to address the dominant failure type.
Do NOT rewrite everything. Keep all correct behavior.

Return JSON only:
{{
  "analysis": "1-2 sentence diagnosis",
  "failure_categories_addressed": ["{dominant_failure}"],
  "revised_system_prompt": "the full revised system prompt"
}}"""

    r = llm.chat(
        messages=[{"role": "user", "content": prompt}],
        system=_CRITIC_SYSTEM,
        temperature=0.2,
        max_tokens=2048,
    )

    import re
    text = r.text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    result = json.loads(text)

    # Build new agent version
    new_agent = agent.model_copy(deep=True)
    new_agent.system_prompt = result["revised_system_prompt"]
    new_agent.version = agent.version + 1

    changes = f"[Iteration {iteration+1}] {result['analysis']}"
    addressed = result.get("failure_categories_addressed", [dominant_failure])

    return new_agent, changes, addressed


def run_critic_optimizer(
    agent: AgentDefinition,
    eval_set: EvalSet,
    doc_folder: str,
    judge_fn,
    max_iterations: int = 3,
    output_dir: str | None = None,
) -> Tuple[AgentDefinition, List[IterationResult]]:
    """
    Run the critic-optimizer loop.
    Returns (best_agent, all_iteration_results).
    """
    all_results: List[IterationResult] = []
    current_agent = agent

    # Initial eval
    print(f"\n{'='*60}")
    print(f"CRITIC-OPTIMIZER LOOP (max {max_iterations} iterations)")
    print("="*60)

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration}/{max_iterations} ---")

        iter_result = judge_fn(
            agent=current_agent,
            eval_set=eval_set,
            doc_folder=doc_folder,
            iteration=iteration,
        )
        all_results.append(iter_result)

        # Save iteration artifacts
        if output_dir:
            iter_path = Path(output_dir) / f"iteration_{iteration}.json"
            iter_path.parent.mkdir(parents=True, exist_ok=True)
            iter_path.write_text(iter_result.model_dump_json(indent=2))
            print(f"[Critic] Iteration {iteration} saved → {iter_path}")

        # Perfect score or last iteration → stop
        if iter_result.score >= 1.0:
            print(f"[Critic] Perfect score at iteration {iteration}. Stopping.")
            break

        if iteration == max_iterations:
            print(f"[Critic] Budget reached ({max_iterations} iterations). Stopping.")
            break

        # Check if improvement is worth another iteration
        if len(all_results) >= 2:
            prev_score = all_results[-2].score
            improvement = iter_result.score - prev_score
            if improvement <= 0.02:
                print(f"[Critic] Improvement {improvement:.1%} ≤ 2% – early stopping (not paying for itself).")
                break

        # Apply fix for next iteration
        print(f"[Critic] Analyzing failures and applying targeted fix...")
        failure_summary = _summarize_failures(iter_result)
        print(f"  Failures by category: {failure_summary['by_category']}")

        try:
            new_agent, changes, addressed = _apply_targeted_fix(
                current_agent, failure_summary, iteration
            )
            print(f"  Fix: {changes}")
            # Update the last result with change info
            all_results[-1].changes_made = changes
            all_results[-1].failures_addressed = addressed
            current_agent = new_agent

            # Save updated agent definition
            if output_dir:
                agent_path = Path(output_dir) / f"agent_v{new_agent.version}.json"
                agent_path.write_text(new_agent.model_dump_json(indent=2))
                print(f"[Critic] Updated agent saved → {agent_path}")

        except Exception as e:
            print(f"[Critic] Fix generation failed: {e}. Continuing with current agent.")

    # Return the best-scoring agent
    best_idx = max(range(len(all_results)), key=lambda i: all_results[i].score)
    best_iteration = all_results[best_idx].iteration

    # Find corresponding agent version
    # Version = initial (1) + iterations applied before that result
    best_version = min(best_iteration, current_agent.version)
    print(f"\n[Critic] Best result: Iteration {best_iteration} (score={all_results[best_idx].score:.1%})")

    return current_agent, all_results
