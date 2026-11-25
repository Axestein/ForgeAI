"""
Stage 1 – Intake Interviewer
Reads the brief + doc folder, asks ≤5 clarifying questions,
produces a RequirementsDoc and runs the groundability guard.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

from forgeai.core import llm
from forgeai.core.documents import Document, load_documents
from forgeai.core.models import RequirementsDoc

_INTAKE_SYSTEM = """You are ForgeAI's requirements analyst.
Your job: given a brief and document summaries, produce a structured requirements document.
Always respond in valid JSON matching the schema provided.
Be conservative about groundability: only mark groundability_ok=true if the documents
genuinely contain enough information to answer the must-answer topics."""

_QUESTIONS_SYSTEM = """You are ForgeAI's requirements analyst.
Given a brief and document summaries, generate at most 5 short, essential clarifying questions
that would most improve the agent specification. Respond ONLY as a JSON array of strings.
If the brief is fully clear, return an empty array."""


def _doc_summary(docs: List[Document]) -> str:
    lines = []
    for d in docs:
        preview = d.content[:300].replace("\n", " ")
        lines.append(f"- [{d.doc_type.upper()}] {d.source}: {preview}...")
    return "\n".join(lines) if lines else "(no documents)"


def _extract_json(text: str) -> dict | list:
    """Strip markdown fences and parse JSON."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _ask_clarifying_questions(brief: str, doc_summary: str) -> List[str]:
    prompt = f"""Brief: {brief}

Documents available:
{doc_summary}

Generate at most 5 clarifying questions as a JSON array."""
    r = llm.chat(
        messages=[{"role": "user", "content": prompt}],
        system=_QUESTIONS_SYSTEM,
        temperature=0.2,
    )
    try:
        questions = _extract_json(r.text)
        if isinstance(questions, list):
            return [str(q) for q in questions[:5]]
    except Exception:
        pass
    return []


def _get_cli_answers(questions: List[str]) -> List[dict]:
    """Interactive CLI dialogue – returns list of {question, answer}."""
    qa = []
    if not questions:
        return qa
    print("\n" + "="*60)
    print("ForgeAI Intake: Clarifying Questions")
    print("="*60)
    for i, q in enumerate(questions, 1):
        print(f"\nQ{i}: {q}")
        answer = input("Your answer (press Enter to skip): ").strip()
        qa.append({"question": q, "answer": answer or "(no answer provided)"})
    print("="*60 + "\n")
    return qa


def _build_requirements(
    brief: str,
    doc_summary: str,
    qa_pairs: List[dict],
) -> RequirementsDoc:
    qa_text = ""
    for item in qa_pairs:
        qa_text += f"\nQ: {item['question']}\nA: {item['answer']}\n"

    schema = RequirementsDoc.model_json_schema()

    prompt = f"""Brief: {brief}

Documents available:
{doc_summary}

Clarifying Q&A:{qa_text if qa_text else ' (none)'}

Produce a requirements document as JSON matching this schema:
{json.dumps(schema, indent=2)}

Rules:
- scope: 1-2 sentence description of what the agent will and won't do
- audience: who will use the agent
- tone: formal | friendly | technical | neutral
- must_answer_topics: 5-10 topics the agent must be able to answer
- must_refuse_topics: 3-6 topics the agent must refuse
- groundability_ok: true only if documents support the must_answer_topics
- groundability_issues: list any gaps between what's needed and what's in documents

Respond ONLY with the JSON object."""

    r = llm.chat(
        messages=[{"role": "user", "content": prompt}],
        system=_INTAKE_SYSTEM,
        temperature=0.2,
        max_tokens=2048,
    )
    data = _extract_json(r.text)
    data["brief"] = brief
    data["clarifying_qa"] = qa_pairs
    return RequirementsDoc(**data)


def run_intake(
    brief: str,
    doc_folder: str,
    interactive: bool = True,
    output_path: str | None = None,
) -> RequirementsDoc:
    """
    Full intake pipeline:
    1. Load documents
    2. Ask clarifying questions (interactive)
    3. Build requirements doc
    4. Groundability guard
    """
    print("[Intake] Loading documents...")
    docs = load_documents(doc_folder)
    if not docs:
        raise ValueError(f"No supported documents found in: {doc_folder}")
    print(f"[Intake] Loaded {len(docs)} document(s).")

    doc_sum = _doc_summary(docs)

    questions: List[str] = []
    if interactive:
        print("[Intake] Generating clarifying questions...")
        questions = _ask_clarifying_questions(brief, doc_sum)

    qa_pairs = _get_cli_answers(questions) if interactive and questions else []

    print("[Intake] Building requirements document...")
    req = _build_requirements(brief, doc_sum, qa_pairs)

    # Groundability guard report
    if not req.groundability_ok:
        print("\n⚠️  GROUNDABILITY GUARD TRIGGERED")
        print("The documents may not fully support all required topics:")
        for issue in req.groundability_issues:
            print(f"  • {issue}")
        print("Proceeding with generation, but the agent may have coverage gaps.\n")

    # Save to disk
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(req.model_dump_json(indent=2))
        print(f"[Intake] Requirements saved → {output_path}")

    return req
