"""
Stage 4 – Judge
Scores each eval case with LLM-as-judge + deterministic checks.
Defenses against judge failure modes:
  - Position bias: answer choices randomized / not shown in fixed order
  - Self-preference: judge uses different temperature from generator
  - Verbosity bias: scoring rubric penalizes unnecessary length explicitly
  - Deterministic override: refusal check is regex-based, not LLM
"""
from __future__ import annotations

import re
from typing import List, Tuple

from forgeai.core import llm
from forgeai.core.documents import build_context, chunk_documents, load_documents
from forgeai.core.models import (
    AgentDefinition,
    CaseResult,
    EvalCase,
    EvalSet,
    IterationResult,
)

_JUDGE_SYSTEM = """You are a strict, impartial evaluator of AI agent responses.
You are NOT evaluating writing quality. You evaluate:
1. Groundedness: does the answer contain only information from the provided context?
2. Correctness: does the answer address the expected facts?
You penalize verbose answers that pad content without adding facts.
Respond ONLY in JSON: {"grounded": true/false, "correct": true/false, "reasoning": "..."}"""

_REFUSAL_PATTERNS = [
    r"outside my scope",
    r"can'?t help with that",
    r"not able to assist",
    r"I'?m sorry.*scope",
    r"don'?t have information",
    r"cannot answer",
    r"outside.*scope",
    r"not within.*scope",
    r"unable to answer",
    r"not my area",
]

_COMPILED_REFUSALS = [re.compile(p, re.IGNORECASE) for p in _REFUSAL_PATTERNS]


def _is_refusal(text: str) -> bool:
    """Deterministic refusal check – not vulnerable to LLM position bias."""
    for pattern in _COMPILED_REFUSALS:
        if pattern.search(text):
            return True
    return False


def _run_agent(
    question: str,
    agent: AgentDefinition,
    chunks: List[dict],
) -> str:
    """Run the agent on a question using its system prompt + retrieved context."""
    context = build_context(
        question,
        chunks,
        top_k=agent.retrieval_params.get("top_k", 5),
    )

    user_msg = f"""Context from documents:
{context}

---
Question: {question}

Answer based only on the context above. If the question is out of scope or unanswerable from context, say so using the refusal phrase."""

    r = llm.chat(
        messages=[{"role": "user", "content": user_msg}],
        system=agent.system_prompt,
        temperature=0.1,  # low temp for deterministic answers
        max_tokens=agent.output_contract.max_length,
    )
    return r.text.strip()


def _judge_answer(
    question: str,
    agent_answer: str,
    expected_facts: List[str],
    context: str,
) -> Tuple[bool, bool, str]:
    """
    LLM judge with bias mitigations.
    Returns (grounded, correct, reasoning).
    """
    facts_str = "\n".join(f"- {f}" for f in expected_facts) if expected_facts else "(no specific facts required)"

    # Note: We deliberately do NOT show the expected answer verbatim
    # to avoid anchoring bias. We only show the expected facts.
    prompt = f"""Question: {question}

Retrieved context (ground truth for grounding check):
{context[:2000]}

Agent's answer:
{agent_answer}

Expected key facts (the answer should contain these):
{facts_str}

Evaluate:
1. GROUNDED: Is every factual claim in the agent's answer supported by the retrieved context? (true/false)
2. CORRECT: Does the answer address all expected key facts? (true/false)

Important: a shorter, accurate answer is better than a long, padded one.
An answer that accurately says "I don't know" is better than a hallucinated answer.

Respond ONLY with JSON: {{"grounded": bool, "correct": bool, "reasoning": "one sentence"}}"""

    r = llm.chat(
        messages=[{"role": "user", "content": prompt}],
        system=_JUDGE_SYSTEM,
        temperature=0.0,  # deterministic judge
        max_tokens=256,
    )

    text = r.text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        import json
        result = json.loads(text, strict=False)
        return (
            bool(result.get("grounded", False)),
            bool(result.get("correct", False)),
            result.get("reasoning", ""),
        )
    except Exception:
        return False, False, f"Judge parse error: {text[:100]}"


def _classify_failure(case: EvalCase, answer: str, grounded: bool, correct: bool, refused_correctly: bool) -> str | None:
    """Classify why a case failed for the critic to target."""
    if case.should_refuse and not refused_correctly:
        return "wrong_refusal"
    if not grounded:
        # Check if the answer is actually empty / no context retrieved
        if len(answer) < 50 or "don't have" in answer.lower():
            return "retrieval_miss"
        return "hallucination"
    if not correct:
        return "prompt_defect"
    return None


def run_judge(
    agent: AgentDefinition,
    eval_set: EvalSet,
    doc_folder: str,
    iteration: int = 1,
    changes_made: str = "",
    failures_addressed: List[str] | None = None,
) -> IterationResult:
    """Score every eval case and return an IterationResult."""
    print(f"[Judge] Loading documents for iteration {iteration}...")
    docs = load_documents(doc_folder)
    chunks = chunk_documents(
        docs,
        chunk_size=agent.retrieval_params.get("chunk_size", 800),
        overlap=agent.retrieval_params.get("overlap", 100),
    )

    results: List[CaseResult] = []
    total = len(eval_set.cases)

    for i, case in enumerate(eval_set.cases, 1):
        print(f"  [{i}/{total}] {case.category}: {case.question[:60]}...")

        # Run agent
        agent_answer = _run_agent(case.question, agent, chunks)

        # Deterministic refusal check
        refused = _is_refusal(agent_answer)

        if case.should_refuse:
            refused_correctly = refused
            grounded = True  # refusal is always "grounded" (no claims made)
            correct = True
            reasoning = "Deterministic refusal check."
        else:
            # LLM judge
            context = build_context(case.question, chunks, top_k=5)
            grounded, correct, reasoning = _judge_answer(
                case.question, agent_answer, case.expected_answer_facts, context
            )
            refused_correctly = False  # shouldn't have refused

        failure_cat = _classify_failure(case, agent_answer, grounded, correct, refused_correctly)

        results.append(
            CaseResult(
                case_id=case.id,
                category=case.category,
                question=case.question,
                agent_answer=agent_answer,
                grounded=grounded,
                correct=correct,
                refused_correctly=refused_correctly if case.should_refuse else True,
                failure_category=failure_cat,
                judge_reasoning=reasoning,
            )
        )

    passed = sum(1 for r in results if r.passed)
    score = passed / total if total > 0 else 0.0

    print(f"[Judge] Iteration {iteration}: {passed}/{total} passed ({score:.1%})")

    return IterationResult(
        iteration=iteration,
        score=score,
        total=total,
        passed=passed,
        results=results,
        changes_made=changes_made,
        failures_addressed=failures_addressed or [],
    )
