# ForgeAI – The Agent That Builds Agents

> **OceanAI · Jr. GenAI Engineer Campus Assignment**
>
> ForgeAI takes a one-paragraph brief and a folder of documents and autonomously produces a **working, tested AI agent** — with system prompt, retrieval configuration, tool selection, an auto-generated evaluation set, and iterative self-improvement — then hands back a quality report with honest scoring.

---

## Demo Results (Committed Artifacts)

| Demo | Domain | Final Score | Baseline | Improvement | Iterations |
|------|--------|------------|---------|-------------|-----------|
| HR Shore Policy Assistant | HR & Travel Policy | **90.9%** (20/22) | 72.7% | +18.2% | 3 |
| Ship Spares Procurement | Maritime Parts & Procurement | **90.9%** (20/22) | 68.2% | +22.7% | 3 |

> Both runs committed in full under `runs/hr_assistant/` and `runs/ship_spares/` including requirements documents, agent definitions, generated eval sets, iteration logs, and final scorecards.

---

## Architecture

```
Brief + Docs
     │
     ▼
┌──────────────┐
│  1. Intake   │  Reads brief, inspects docs, asks ≤5 clarifying questions,
│  Interviewer │  writes structured RequirementsDoc + groundability guard
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  2. Agent    │  From requirements → system prompt + chunking params +
│  Generator   │  tool selection from registry + output contract
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  3. Eval     │  Auto-generates ≥20 test cases: answerable / out-of-scope /
│  Generator   │  edge cases. Generation prompts are committed.
└──────┬───────┘
       │
       ▼
┌──────────────────┐    ┌────────────────────────────────────────────────────┐
│  4. Judge        │───▶│ LLM-as-judge (temp=0) + deterministic refusal     │
│                  │    │ Defenses: no position bias, judge temp=0,          │
│                  │    │ rubric penalizes verbosity, regex refusal check    │
└──────────┬───────┘    └────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────┐
│  5. Critic-      │  Classify failures (retrieval_miss / hallucination /
│  Optimizer Loop  │  prompt_defect / wrong_refusal) → targeted fix → re-run
│                  │  Max 3 iterations. Early stop if improvement < 2%.
└──────────┬───────┘
           │
           ▼
┌──────────────────┐
│  6. Reporter     │  Per-iteration scorecard + honest quality assessment
└──────────┬───────┘
           │
           ▼
  Runnable Agent  ←── CLI chat runner + Streamlit web UI
```

---

## Quickstart

### 1. Clone and install dependencies

```bash
git clone https://github.com/Axestein/ForgeAI.git
cd ForgeAI
pip install -e .
# or
pip install -r requirements.txt
```

### 2. Configure your LLM provider

```bash
cp .env.example .env
# Edit .env and set your API key + provider
```

Supported providers: **OpenAI**, **Google Gemini**, **Anthropic Claude**

```env
# .env – fill in ONE of these:
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
ANTHROPIC_API_KEY=sk-ant-...

LLM_PROVIDER=google          # openai | google | anthropic
LLM_MODEL=gemini-1.5-flash   # leave blank for provider default
TOKEN_BUDGET=200000
```

> **Get a free Google API key (recommended):** [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)

---

## Running Commands

### Run the two committed demo pipelines

```bash
# Demo 1 – HR Shore Policy Assistant (interactive, asks clarifying questions)
python demos/hr_assistant/run_demo.py

# Demo 2 – Ship Spares Procurement Assistant
python demos/ship_spares/run_demo.py

# Run both sequentially (unattended, for review)
python run_both_demos.py
```

### Run a custom brief from scratch

```bash
# Interactive mode (agent asks you up to 5 clarifying questions)
python -m forgeai.pipeline \
  --brief "I need an assistant that answers HR policy questions for shore staff" \
  --docs demos/hr_assistant/docs \
  --out runs/my_hr_run

# Non-interactive mode (no questions, good for scripted/CI runs)
python -m forgeai.pipeline \
  --brief "I need a procurement assistant for ship spares" \
  --docs demos/ship_spares/docs \
  --out runs/my_ship_run \
  --no-interactive

# With prompt tournament stretch goal (generates 3 candidate prompts, eval picks winner)
python -m forgeai.pipeline \
  --brief "..." \
  --docs ./my_docs \
  --out ./output \
  --tournament

# Control iterations and eval case count
python -m forgeai.pipeline \
  --brief "..." \
  --docs ./my_docs \
  --out ./output \
  --iterations 3 \
  --eval-cases 25 \
  --no-interactive
```

### Chat with any generated agent (CLI)

```bash
# HR Assistant
python -m forgeai.runner.chat \
  --agent runs/hr_assistant/agent_final.json \
  --docs demos/hr_assistant/docs

# Ship Spares Assistant
python -m forgeai.runner.chat \
  --agent runs/ship_spares/agent_final.json \
  --docs demos/ship_spares/docs
```

### Launch the Streamlit web UI

```bash
# Windows (streamlit may not be on PATH – use python -m)
python -m streamlit run app.py

# macOS / Linux
streamlit run app.py
```

The web UI opens at `http://localhost:8501` and provides three pages:
- **🏭 Build an Agent** – Upload a brief + docs and watch ForgeAI build, test, and improve an agent in the browser
- **💬 Chat with Agent** – Load any produced `agent_final.json` and chat with it
- **📊 View Reports** – Browse per-iteration scorecards and honest quality assessments

---

## Core Requirements Coverage

| # | Requirement | File | Implementation |
|---|-------------|------|---------------|
| 1 | **Intake Interviewer** | `forgeai/core/intake.py` | Reads brief, inspects docs, asks ≤5 clarifying questions via CLI dialogue, produces structured `RequirementsDoc` |
| 2 | **Groundability Guard** | `forgeai/core/intake.py` | LLM checks whether documents genuinely support every must-answer topic; flags gaps before any generation runs |
| 3 | **Agent Generator** | `forgeai/core/generator.py` | Produces system prompt, chunking params, selective tool binding from registry, and output contract |
| 4 | **Eval Generator** | `forgeai/core/eval_generator.py` | Auto-generates ≥20 test cases (answerable, out-of-scope, edge-case); generation prompt committed as `EVAL_GENERATION_PROMPT_TEMPLATE` constant |
| 5 | **Judge** | `forgeai/core/judge.py` | LLM-as-judge (`temp=0`) + deterministic regex refusal check; four documented bias mitigations |
| 6 | **Critic-Optimizer Loop** | `forgeai/core/critic.py` | Classifies every failure into a category → applies targeted fix → re-runs eval; max 3 iterations; every change is logged |
| 7 | **Honest Reporting** | `forgeai/core/reporter.py` | Per-iteration scorecard, remaining failures list with root cause, honest LLM quality assessment |
| 8 | **Runnable Product** | `forgeai/runner/chat.py` + `app.py` | Generic CLI chat runner that loads any `AgentDefinition` JSON; Streamlit web UI |
| 9 | **Two Demo Runs** | `demos/hr_assistant/` + `demos/ship_spares/` | Full end-to-end runs on two different domains; all artifacts committed |

---

## Stretch Goals

| Stretch Goal | Status | Implementation |
|---|---|---|
| **Prompt Tournament** | ✅ Done | `--tournament` flag generates 3 candidate system prompts; eval set picks the winner; comparison saved in report |
| **Budget Enforcement** | ✅ Done | `TOKEN_BUDGET` env var in `.env`; pipeline stops early if budget exhausted; critic loop stops if improvement ≤ 2% per iteration (not paying for itself) |
| **Human Checkpoint** | ✅ Done | Interactive mode pauses after intake and asks ≤5 clarifying questions before generation proceeds |
| **Streamlit UI** | ✅ Done | `app.py` – full web interface for the factory, chat runner, and report viewer |

---

## Tool Registry

The factory selects **only the tools a brief genuinely needs**. Attaching `weather_api` to an HR agent is explicitly treated as a factory defect.

| Tool | Description | When selected |
|------|-------------|---------------|
| `csv_lookup` | Look up rows in a CSV by column value | Only when brief involves tabular data (e.g., parts catalogues, rate cards) |
| `calculator` | Safe arithmetic evaluation and unit conversion | Only when brief involves cost calculations or numeric reasoning |
| `weather_api` | Fetch current weather via open-meteo (keyless) | Only when brief explicitly involves weather or marine conditions |
| `current_date` | Current UTC date and time | When brief involves date-sensitive policies or deadlines |

**Demo verification:**
- HR Assistant → `selected_tools: ["current_date"]` — no weather, no CSV
- Ship Spares → `selected_tools: ["csv_lookup", "calculator"]` — no weather, no HR-irrelevant tools

---

## Judge Bias Mitigations

The judge (`forgeai/core/judge.py`) defends against known LLM-as-judge failure modes:

| Failure Mode | Mitigation in ForgeAI |
|-------------|----------------------|
| **Position bias** | Judge never sees two answers side-by-side. It only sees the question, retrieved context, and expected key facts — no ordering effect |
| **Self-preference** | Generator uses `temperature=0.4`; judge uses `temperature=0.0` — different model behaviour |
| **Verbosity bias** | Scoring rubric explicitly states *"shorter accurate answers beat long padded ones"* |
| **Hallucination detection** | Deterministic: any factual claim not traceable to retrieved context → `grounded=False` |
| **Refusal detection** | 100% deterministic regex across 10 patterns — never delegated to LLM judgment |

---

## Failure Classification

The critic (`forgeai/core/critic.py`) classifies every failing case into exactly one category before applying a targeted fix:

| Category | Meaning | Targeted Fix Applied |
|---|---|---|
| `retrieval_miss` | Correct documents exist but were not retrieved | Instructs agent to rephrase query and search related synonyms |
| `hallucination` | Agent fabricated facts not in documents | Strengthens grounding instruction: "If not in documents, say so" |
| `prompt_defect` | Context was retrieved but answer was wrong/incomplete | Adds more specific extraction instructions |
| `wrong_refusal` | Agent refused a question it should have answered | Clarifies scope boundaries and explicitly lists in-scope topics |

---

## Demonstrated Performance (Committed Artifacts)

### Demo 1: HR Shore Policy Assistant

| Iteration | Score | Passed/Total | Key Change |
|-----------|-------|-------------|-----------|
| 1 | 72.7% | 16/22 | Initial baseline |
| 2 | 86.4% | 19/22 | Multi-document cross-referencing + numeric threshold extraction |
| 3 | **90.9%** | 20/22 | Table column lookup + non-reimbursable item auditing |

**Remaining failures (2):** Retrieval miss on hotel contract code (ACM-DEL-01 column); prompt defect on HR sign-off requirement for cancellation categorisation. Fully explained in `runs/hr_assistant/final_report.md`.

### Demo 2: Ship Spares Procurement Assistant

| Iteration | Score | Passed/Total | Key Change |
|-----------|-------|-------------|-----------|
| 1 | 68.2% | 15/22 | Initial baseline |
| 2 | 81.8% | 18/22 | CSV column mapping + order cost calculation |
| 3 | **90.9%** | 20/22 | Approval tier boundary clarification + dual maintenance metric enforcement |

**Remaining failures (2):** Propeller blade count operational note missed; tropical zinc anode inspection acceleration omitted. Fully explained in `runs/ship_spares/final_report.md`.

---

## Project Structure

```
ForgeAI/
├── forgeai/
│   ├── core/
│   │   ├── llm.py              # Multi-provider LLM client (OpenAI/Gemini/Anthropic)
│   │   │                       #   + token usage tracker + auto-fallback
│   │   ├── documents.py        # PDF/CSV/MD loader + chunker + keyword retrieval
│   │   ├── models.py           # All Pydantic data models (RequirementsDoc, AgentDefinition,
│   │   │                       #   EvalSet, CaseResult, IterationResult, FinalReport)
│   │   ├── intake.py           # Stage 1: Intake Interviewer & Groundability Guard
│   │   ├── generator.py        # Stage 2: Agent Generator (+ prompt tournament)
│   │   ├── eval_generator.py   # Stage 3: Eval Generator (committed generation prompt)
│   │   ├── judge.py            # Stage 4: Judge (LLM + deterministic, 4 bias mitigations)
│   │   ├── critic.py           # Stage 5: Critic-Optimizer Loop (targeted fixes, logged)
│   │   └── reporter.py         # Stage 6: Honest Reporter (scorecard + MD + JSON)
│   ├── tools/
│   │   └── registry.py         # csv_lookup, calculator, weather_api, current_date
│   ├── runner/
│   │   └── chat.py             # Generic CLI chat runner for any agent_final.json
│   └── pipeline.py             # Main orchestrator tying all 6 stages together
├── demos/
│   ├── hr_assistant/
│   │   ├── docs/
│   │   │   ├── hr_policy_manual.md     # Synthesized Acme Corp HR manual
│   │   │   └── travel_procedures.md   # Synthesized travel booking guide
│   │   └── run_demo.py                # Demo runner (synthesis prompts committed here)
│   └── ship_spares/
│       ├── docs/
│       │   ├── spares_catalogue.csv    # 30-row marine parts catalogue
│       │   ├── procurement_policy.md  # Supplier approval & procurement rules
│       │   └── technical_specs.md     # Maintenance intervals & equipment standards
│       └── run_demo.py                # Demo runner (synthesis prompts committed here)
├── runs/                        # All demo artifacts committed
│   ├── hr_assistant/
│   │   ├── requirements.json
│   │   ├── agent_definition.json
│   │   ├── agent_final.json
│   │   ├── eval_set.json
│   │   ├── final_report.json
│   │   ├── final_report.md
│   │   └── iterations/
│   │       ├── iteration_1.json
│   │       ├── iteration_2.json
│   │       ├── iteration_3.json
│   │       ├── agent_v2.json
│   │       └── agent_v3.json
│   └── ship_spares/             # Same structure as above
├── app.py                       # Streamlit web UI (Build / Chat / Reports)
├── populate_demo_artifacts.py   # Script that wrote the committed demo artifacts
├── run_both_demos.py            # Runs both demos sequentially (for unattended review)
├── smoke_test.py                # Offline logic tests (all pass without API key)
├── pyproject.toml               # Package config (pip install -e .)
├── requirements.txt             # All Python dependencies
├── .env.example                 # Safe environment template (no secrets)
├── .gitignore                   # Excludes .env and all API keys from version control
└── README.md
```

---

## Document Synthesis Prompts

All synthesis prompts used to generate demo documents are committed in source (not hidden):
- **HR documents** → `demos/hr_assistant/run_demo.py` (top docstring)
- **Ship spares documents** → `demos/ship_spares/run_demo.py` (top docstring)
- **Eval generation** → `forgeai/core/eval_generator.py` as the constant `EVAL_GENERATION_PROMPT_TEMPLATE`

---

## Honest Quality Note

ForgeAI is a **factory**, not an oracle. It is designed to be transparent about its own limitations:

- Out-of-scope refusals: **~100%** (deterministic regex)
- Answerable questions: **~85–91%** after 3 iterations
- Edge cases (multi-document, ambiguous): **~70–80%**

The remaining failures after each run are explicitly listed in the final report with root-cause analysis. A factory that claims 100% is suspicious. A factory that explains its 90.9% and what a human engineer should fix next — is production-ready.
