"""
Stage 2 – Agent Generator
Produces a complete AgentDefinition from requirements + documents.
Optionally runs a prompt tournament (stretch goal).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

from forgeai.core import llm
from forgeai.core.documents import Document, load_documents, chunk_documents
from forgeai.core.models import (
    AgentDefinition,
    OutputContract,
    RequirementsDoc,
    TOOL_REGISTRY,
    ToolSpec,
)

_GENERATOR_SYSTEM = """You are ForgeAI's agent architect.
Given a requirements document and document summaries, produce a complete agent definition.
Respond ONLY in valid JSON matching the schema provided.
Be precise: the system_prompt must cite only what the documents actually contain.
Only select tools that the brief genuinely needs."""


def _extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text, strict=False)


def _select_tools(req: RequirementsDoc) -> List[str]:
    """Ask the LLM which tools from TOOL_REGISTRY the agent actually needs."""
    tool_list = "\n".join(
        f"- {t.name}: {t.description} | Use when: {t.when_to_use}"
        for t in TOOL_REGISTRY
    )
    prompt = f"""Requirements scope: {req.scope}
Brief: {req.brief}
Must-answer topics: {req.must_answer_topics}

Available tools:
{tool_list}

Which tools does this agent genuinely need?
Return ONLY a JSON array of tool names from the list above.
Only include tools that are clearly needed. An HR policy agent does NOT need weather_api."""

    r = llm.chat(
        messages=[{"role": "user", "content": prompt}],
        system="You select tools conservatively. Return JSON array only.",
        temperature=0.1,
    )
    try:
        tools = _extract_json(r.text)
        valid = {t.name for t in TOOL_REGISTRY}
        return [t for t in tools if t in valid]
    except Exception:
        return []


def _generate_system_prompt(
    req: RequirementsDoc,
    docs: List[Document],
    selected_tools: List[str],
    version_note: str = "",
) -> str:
    doc_summary = "\n".join(
        f"- {d.source}: {d.content[:400].replace(chr(10),' ')}..."
        for d in docs
    )
    tools_desc = ""
    if selected_tools:
        t_map = {t.name: t for t in TOOL_REGISTRY}
        tools_desc = "\n".join(
            f"- {name}: {t_map[name].description}"
            for name in selected_tools
            if name in t_map
        )

    refuse_topics = "\n".join(f"- {t}" for t in req.must_refuse_topics)

    prompt = f"""Create a system prompt for an AI agent with these specifications:

Name: {req.brief[:60]}
Scope: {req.scope}
Audience: {audience_str(req.audience)}
Tone: {req.tone}

Must-answer topics:
{chr(10).join('- '+t for t in req.must_answer_topics)}

Must-refuse topics:
{refuse_topics}

Available documents:
{doc_summary}

Tools available to the agent:
{tools_desc if tools_desc else '(none – answer only from documents)'}

Output format: {req.tone} tone, cite source document for every factual claim.
Refusal phrase to use: "I'm sorry, that topic is outside my scope."

{version_note}

Write a comprehensive system prompt (200-400 words) that:
1. States the agent's role and purpose clearly
2. Instructs it to answer ONLY from provided documents
3. Lists what it must refuse
4. Explains how to cite sources
5. Sets the tone

Return ONLY the system prompt text (no JSON, no markdown fences)."""

    r = llm.chat(
        messages=[{"role": "user", "content": prompt}],
        system="You write precise, grounded system prompts for AI agents.",
        temperature=0.4,
        max_tokens=1024,
    )
    return r.text.strip()


def audience_str(audience: str) -> str:
    return audience or "general users"


def _generate_agent_definition(
    req: RequirementsDoc,
    docs: List[Document],
    selected_tools: List[str],
    version: int = 1,
    version_note: str = "",
) -> AgentDefinition:
    system_prompt = _generate_system_prompt(req, docs, selected_tools, version_note)

    # Determine chunking params
    avg_doc_len = sum(len(d.content) for d in docs) / max(len(docs), 1)
    chunk_size = 600 if avg_doc_len < 5000 else 800

    retrieval_params = {
        "chunk_size": chunk_size,
        "overlap": 100,
        "top_k": 5,
        "strategy": "keyword",
    }

    output_contract = OutputContract(
        format="markdown" if req.tone == "technical" else "plain_text",
        max_length=600,
        must_cite_sources=True,
        refusal_phrase="I'm sorry, that topic is outside my scope.",
    )

    # Build a name from the brief
    name_words = req.brief.lower().split()[:4]
    name = "_".join(re.sub(r"[^a-z0-9]", "", w) for w in name_words if w)
    name = name or "forgeai_agent"

    return AgentDefinition(
        name=name,
        system_prompt=system_prompt,
        retrieval_params=retrieval_params,
        selected_tools=selected_tools,
        output_contract=output_contract,
        version=version,
    )


def run_generator(
    req: RequirementsDoc,
    doc_folder: str,
    output_path: str | None = None,
    tournament: bool = False,
) -> Tuple[AgentDefinition, List[AgentDefinition]]:
    """
    Generate agent definition(s).
    If tournament=True, generates 3 candidates and returns all.
    Returns (best_or_only, all_candidates).
    """
    print("[Generator] Loading documents for generation...")
    docs = load_documents(doc_folder)

    print("[Generator] Selecting tools...")
    selected_tools = _select_tools(req)
    print(f"[Generator] Selected tools: {selected_tools or '(none)'}")

    if tournament:
        print("[Generator] Running prompt tournament (3 candidates)...")
        candidates = []
        notes = [
            "Style: concise and direct",
            "Style: detailed with examples",
            "Style: structured with numbered rules",
        ]
        for i, note in enumerate(notes, 1):
            print(f"  Generating candidate {i}/3...")
            agent = _generate_agent_definition(req, docs, selected_tools, version=i, version_note=note)
            agent.name = f"{agent.name}_v{i}"
            candidates.append(agent)
        best = candidates[0]  # eval stage will pick winner
    else:
        best = _generate_agent_definition(req, docs, selected_tools, version=1)
        candidates = [best]

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(best.model_dump_json(indent=2), encoding="utf-8")
        print(f"[Generator] Agent definition saved → {output_path}")

    return best, candidates
