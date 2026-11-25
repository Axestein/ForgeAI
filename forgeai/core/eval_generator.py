"""
Stage 3 – Eval Generator
Auto-creates ≥20 test cases from the documents themselves:
  - answerable questions (with grounding facts + sources)
  - out-of-scope questions (must be refused)
  - edge cases (ambiguous, multi-document)
"""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import List

from forgeai.core import llm
from forgeai.core.documents import Document, load_documents
from forgeai.core.models import EvalCase, EvalSet, RequirementsDoc

_EVAL_SYSTEM = """You are ForgeAI's evaluation engineer.
Your job: generate diverse, challenging test cases for an AI agent from document content.
Return ONLY valid JSON arrays. Never hallucinate document content - base questions strictly on the provided text."""

# This is the generation prompt – committed per assignment requirements
EVAL_GENERATION_PROMPT_TEMPLATE = """You are generating evaluation test cases for an AI agent.

Agent scope: {scope}
Must-answer topics: {must_answer}
Must-refuse topics: {must_refuse}

Document excerpts (the agent's knowledge base):
{doc_excerpts}

Generate exactly {n_answerable} ANSWERABLE test cases, {n_refusable} OUT-OF-SCOPE test cases,
and {n_edge} EDGE CASE test cases.

Return a JSON array where each element is:
{{
  "id": "<unique string>",
  "category": "answerable" | "out_of_scope" | "edge_case",
  "question": "<the question>",
  "expected_answer_facts": ["<key fact 1>", "<key fact 2>"],
  "expected_sources": ["<source filename>"],
  "should_refuse": true | false
}}

Rules:
- answerable: question must be answerable from the documents; list 2-4 key facts the answer should contain; should_refuse=false
- out_of_scope: question is clearly outside the agent's domain; expected_answer_facts=[]; should_refuse=true
- edge_case: ambiguous phrasing, or answer spans two documents; should_refuse=false
- Vary difficulty: some easy, some hard, some tricky
- Do NOT repeat questions
- Ensure at least some questions span multiple documents
- Return ONLY the JSON array, no other text"""


def _build_doc_excerpts(docs: List[Document], max_chars: int = 8000) -> str:
    parts = []
    per_doc = max_chars // max(len(docs), 1)
    for d in docs:
        excerpt = d.content[:per_doc].replace("\n", " ")
        parts.append(f"[{d.source}]\n{excerpt}")
    return "\n\n".join(parts)


def _generate_eval_batch(
    req: RequirementsDoc,
    docs: List[Document],
    n_answerable: int = 12,
    n_refusable: int = 5,
    n_edge: int = 5,
) -> List[EvalCase]:
    excerpts = _build_doc_excerpts(docs)

    prompt = EVAL_GENERATION_PROMPT_TEMPLATE.format(
        scope=req.scope,
        must_answer=", ".join(req.must_answer_topics),
        must_refuse=", ".join(req.must_refuse_topics),
        doc_excerpts=excerpts,
        n_answerable=n_answerable,
        n_refusable=n_refusable,
        n_edge=n_edge,
    )

    r = llm.chat(
        messages=[{"role": "user", "content": prompt}],
        system=_EVAL_SYSTEM,
        temperature=0.5,
        max_tokens=4096,
    )

    text = r.text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    raw_cases = json.loads(text)
    cases = []
    seen_ids = set()
    for rc in raw_cases:
        # Ensure unique IDs
        cid = rc.get("id", str(uuid.uuid4())[:8])
        if cid in seen_ids:
            cid = cid + "_" + str(uuid.uuid4())[:4]
        seen_ids.add(cid)
        cases.append(
            EvalCase(
                id=cid,
                category=rc.get("category", "answerable"),
                question=rc.get("question", ""),
                expected_answer_facts=rc.get("expected_answer_facts", []),
                expected_sources=rc.get("expected_sources", []),
                should_refuse=rc.get("should_refuse", False),
            )
        )
    return cases


def run_eval_generator(
    req: RequirementsDoc,
    doc_folder: str,
    agent_name: str,
    output_path: str | None = None,
    target_total: int = 22,
) -> EvalSet:
    """Generate ≥20 eval cases from the documents."""
    print("[EvalGen] Loading documents...")
    docs = load_documents(doc_folder)

    # Distribute cases: aim for target_total
    n_ans = max(12, target_total - 8)
    n_ref = 5
    n_edge = max(3, target_total - n_ans - n_ref)

    print(f"[EvalGen] Generating {n_ans} answerable + {n_ref} out-of-scope + {n_edge} edge cases...")
    cases = _generate_eval_batch(req, docs, n_ans, n_ref, n_edge)

    # Pad if LLM returned fewer than expected
    if len(cases) < 20:
        print(f"[EvalGen] Got {len(cases)} cases, requesting more...")
        extra = _generate_eval_batch(req, docs, 5, 2, 3)
        seen = {c.id for c in cases}
        for c in extra:
            if c.id not in seen:
                cases.append(c)
                seen.add(c.id)

    eval_set = EvalSet(agent_name=agent_name, cases=cases)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(eval_set.model_dump_json(indent=2))
        print(f"[EvalGen] Eval set ({len(cases)} cases) saved → {output_path}")

    return eval_set
