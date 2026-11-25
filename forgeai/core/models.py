"""
Pydantic models for the structured data flowing through ForgeAI.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Requirements document ─────────────────────────────────────────────────

class RequirementsDoc(BaseModel):
    brief: str
    scope: str
    audience: str
    tone: str
    must_answer_topics: List[str]
    must_refuse_topics: List[str]
    groundability_ok: bool
    groundability_issues: List[str] = Field(default_factory=list)
    clarifying_qa: List[Dict[str, str]] = Field(default_factory=list)


# ── Tool registry ──────────────────────────────────────────────────────────

class ToolSpec(BaseModel):
    name: str
    description: str
    when_to_use: str


TOOL_REGISTRY: List[ToolSpec] = [
    ToolSpec(
        name="csv_lookup",
        description="Look up rows in a CSV file by column value.",
        when_to_use="Use when the user needs to find product, spare part, or tabular data by ID or name.",
    ),
    ToolSpec(
        name="calculator",
        description="Evaluate arithmetic expressions and unit conversions.",
        when_to_use="Use when the user needs numeric calculations, totals, or currency conversions.",
    ),
    ToolSpec(
        name="weather_api",
        description="Fetch current weather for a city using open-meteo (no API key).",
        when_to_use="Use only when the brief explicitly involves weather, marine conditions, or climate data.",
    ),
    ToolSpec(
        name="current_date",
        description="Return the current UTC date and time.",
        when_to_use="Use when the agent needs to know today's date for policy deadlines, shipment timing, etc.",
    ),
]


# ── Output contract ────────────────────────────────────────────────────────

class OutputContract(BaseModel):
    format: str           # e.g. "plain_text", "markdown", "json"
    max_length: int = 500
    must_cite_sources: bool = True
    refusal_phrase: str = "I'm sorry, that topic is outside my scope."


# ── Agent definition ───────────────────────────────────────────────────────

class AgentDefinition(BaseModel):
    name: str
    system_prompt: str
    retrieval_params: Dict[str, Any] = Field(default_factory=dict)
    selected_tools: List[str] = Field(default_factory=list)
    output_contract: OutputContract = Field(default_factory=OutputContract)
    version: int = 1


# ── Eval cases ────────────────────────────────────────────────────────────

class EvalCase(BaseModel):
    id: str
    category: str          # answerable | out_of_scope | edge_case
    question: str
    expected_answer_facts: List[str] = Field(default_factory=list)
    expected_sources: List[str] = Field(default_factory=list)
    should_refuse: bool = False


class EvalSet(BaseModel):
    agent_name: str
    cases: List[EvalCase]


# ── Scoring ───────────────────────────────────────────────────────────────

class CaseResult(BaseModel):
    case_id: str
    category: str
    question: str
    agent_answer: str
    grounded: bool = False
    correct: bool = False
    refused_correctly: bool = False
    failure_category: Optional[str] = None  # retrieval_miss | prompt_defect | hallucination | wrong_refusal
    judge_reasoning: str = ""

    @property
    def passed(self) -> bool:
        if self.category == "out_of_scope":
            return self.refused_correctly
        return self.grounded and self.correct


class IterationResult(BaseModel):
    iteration: int
    score: float
    total: int
    passed: int
    results: List[CaseResult]
    changes_made: str = ""
    failures_addressed: List[str] = Field(default_factory=list)


# ── Final report ──────────────────────────────────────────────────────────

class FinalReport(BaseModel):
    agent_name: str
    iterations: List[IterationResult]
    best_iteration: int
    final_score: float
    remaining_failures: List[str]
    honest_assessment: str
    usage_summary: Dict[str, Any] = Field(default_factory=dict)
