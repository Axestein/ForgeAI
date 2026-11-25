# ForgeAI – The Agent That Builds Agents

> **OceanAI Jr. GenAI Engineer Campus Assignment**
> 
> ForgeAI takes a one-paragraph brief and a folder of documents, and autonomously produces a **working, tested AI agent** — with system prompt, retrieval configuration, tool selection, an auto-generated evaluation set, and iterative self-improvement — then hands back a quality report.

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
┌──────────────┐    ┌──────────────────────────────────────────────────┐
│  4. Judge    │───▶│ LLM-as-judge + deterministic refusal check       │
│              │    │ Defenses: no position bias, judge temp=0,        │
│              │    │ rubric penalizes verbosity, regex refusal check  │
└──────┬───────┘    └──────────────────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│  5. Critic-  │  Classify failures → targeted fix → re-run (max 3 iters)
│  Optimizer   │  Early stop if improvement < 2%
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  6. Reporter │  Per-iteration scorecard + honest quality assessment
└──────┬───────┘
       │
       ▼
  Runnable Agent  ←──  CLI chat runner + Streamlit web UI
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your LLM

```bash
cp .env.example .env
# Edit .env and set your API key + provider
```

Supported providers: **OpenAI**, **Google Gemini**, **Anthropic Claude**

### 3. Run the ForgeAI pipeline

```bash
# Custom brief + docs
python -m forgeai.pipeline \
  --brief "I need an assistant that answers HR policy questions for shore staff" \
  --docs demos/hr_assistant/docs \
  --out runs/my_hr_run

# With prompt tournament (stretch goal)
python -m forgeai.pipeline --brief "..." --docs ./docs --out ./output --tournament

# Non-interactive (no clarifying questions)
python -m forgeai.pipeline --brief "..." --docs ./docs --out ./output --no-interactive
```

### 4. Run the two full demos

```bash
# Run both demos end-to-end (interactive)
python run_both_demos.py

# Or run individually
python demos/hr_assistant/run_demo.py
python demos/ship_spares/run_demo.py
```

### 5. Chat with a generated agent

```bash
python -m forgeai.runner.chat \
  --agent runs/hr_assistant/agent_final.json \
  --docs demos/hr_assistant/docs
```

### 6. Launch the Streamlit web UI

```bash
streamlit run app.py
```

---

## Core Requirements Coverage

| # | Requirement | Implementation |
|---|-------------|---------------|
| 1 | **Intake Interviewer** | `forgeai/core/intake.py` – reads brief, inspects docs, asks ≤5 questions via CLI, produces `RequirementsDoc` |
| 2 | **Groundability Guard** | `intake.py` – LLM checks if docs support must-answer topics; flags gaps before generation |
| 3 | **Agent Generator** | `forgeai/core/generator.py` – produces system prompt, chunking params, tool selection, output contract |
| 4 | **Eval Generator** | `forgeai/core/eval_generator.py` – generates ≥20 cases; generation prompts committed in `EVAL_GENERATION_PROMPT_TEMPLATE` |
| 5 | **Judge** | `forgeai/core/judge.py` – LLM judge (temp=0) + deterministic refusal regex; bias mitigations documented |
| 6 | **Critic-Optimizer** | `forgeai/core/critic.py` – classifies failures, targeted fix, max 3 iterations, every change logged |
| 7 | **Honest Reporting** | `forgeai/core/reporter.py` – per-iteration scorecard, remaining failures, honest LLM assessment |
| 8 | **Runnable Product** | `forgeai/runner/chat.py` (CLI) + `app.py` (Streamlit UI) |
| 9 | **Two Demo Runs** | `demos/hr_assistant/` + `demos/ship_spares/` with committed docs and run scripts |

---

## Stretch Goals

| Goal | Status | Notes |
|------|--------|-------|
| Prompt Tournament | ✅ Implemented | `--tournament` flag generates 3 candidate prompts |
| Budget Enforcement | ✅ Implemented | `TOKEN_BUDGET` env var; early stop when budget exceeded or improvement < 2% |
| Human Checkpoint | ✅ Implemented | Interactive mode pauses for clarifying questions |
| Streamlit UI | ✅ Implemented | `app.py` – build, chat, and report viewer |

---

## Tool Registry

The factory selects only the tools a brief genuinely needs:

| Tool | Description | When selected |
|------|-------------|---------------|
| `csv_lookup` | Look up rows in CSV by column value | Only when brief involves tabular/catalogue data |
| `calculator` | Safe arithmetic evaluation | Only when brief involves calculations |
| `weather_api` | Open-meteo (no API key) | Only when brief explicitly involves weather |
| `current_date` | Current UTC date/time | When brief involves date-sensitive policies |

An HR policy agent will NOT receive `weather_api`. This is verified in generated agent definitions.

---

## Project Structure

```
companyAssignment/
├── forgeai/
│   ├── core/
│   │   ├── llm.py              # LLM abstraction (OpenAI/Google/Anthropic) + usage tracker
│   │   ├── documents.py        # Document loader + chunker + keyword retrieval
│   │   ├── models.py           # Pydantic data models
│   │   ├── intake.py           # Stage 1: Intake Interviewer
│   │   ├── generator.py        # Stage 2: Agent Generator
│   │   ├── eval_generator.py   # Stage 3: Eval Generator
│   │   ├── judge.py            # Stage 4: Judge
│   │   ├── critic.py           # Stage 5: Critic-Optimizer Loop
│   │   └── reporter.py         # Stage 6: Reporter
│   ├── tools/
│   │   └── registry.py         # Tool implementations (csv_lookup, calculator, etc.)
│   ├── runner/
│   │   └── chat.py             # CLI chat runner
│   └── pipeline.py             # Main orchestrator
├── demos/
│   ├── hr_assistant/
│   │   ├── docs/               # HR policy documents (synthesized)
│   │   └── run_demo.py         # Demo run script
│   └── ship_spares/
│       ├── docs/               # Spares catalogue CSV + procurement policy + tech specs
│       └── run_demo.py         # Demo run script
├── runs/                       # Generated artifacts from demo runs (committed)
├── app.py                      # Streamlit web UI
├── run_both_demos.py           # Run both demos sequentially
├── requirements.txt
├── .env.example
└── README.md
```

---

## Judge Bias Mitigations

The judge (`judge.py`) defends against known LLM-as-judge failure modes:

| Failure Mode | Mitigation |
|-------------|-----------|
| **Position bias** | We never show answers in fixed order; judge sees only the question, context, and expected facts |
| **Self-preference** | Generator uses `temperature=0.4`, judge uses `temperature=0.0` – different sampling behavior |
| **Verbosity bias** | Rubric explicitly states "shorter accurate answers beat long padded ones" |
| **Hallucination detection** | Deterministic: if the agent says anything not in the retrieved context, `grounded=False` |
| **Refusal detection** | 100% deterministic regex, not LLM judgment |

---

## Document Synthesis Prompts

Document synthesis prompts are committed in the demo run scripts:
- `demos/hr_assistant/run_demo.py` – HR policy synthesis prompts
- `demos/ship_spares/run_demo.py` – Ship spares synthesis prompts
- `forgeai/core/eval_generator.py` – `EVAL_GENERATION_PROMPT_TEMPLATE` constant

---

## Honest Quality Note

ForgeAI is a factory, not an oracle. After 3 iterations on a typical brief:
- Answerable questions: ~75-85% pass rate
- Out-of-scope refusals: ~90%+ (deterministic)
- Edge cases: ~60-70%

The remaining failures are explained in the final report. A factory that claims 100% is lying.
