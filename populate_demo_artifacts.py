import json
from pathlib import Path
from forgeai.core.models import (
    RequirementsDoc,
    AgentDefinition,
    OutputContract,
    EvalCase,
    EvalSet,
    CaseResult,
    IterationResult,
    FinalReport,
)

def populate_hr_demo():
    out = Path("runs/hr_assistant")
    out.mkdir(parents=True, exist_ok=True)
    iter_dir = out / "iterations"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # 1. Requirements
    req = RequirementsDoc(
        brief="I need an assistant that answers HR policy questions for shore staff. It should cover leave policies (annual, sick, maternity, emergency), travel and reimbursement procedures (per diems, hotel limits, expense submission), and the company code of conduct. It must refuse questions about vessel operations, cargo, or any topic not in the HR policy documents.",
        scope="Answers inquiries regarding shore staff HR policies including annual/sick/parental leave, domestic/international travel reimbursement, expense limits, and workplace code of conduct. Excludes vessel operations and crew matters.",
        audience="Shore-based employees and line managers of Acme Corp.",
        tone="professional, empathetic, and objective",
        must_answer_topics=[
            "Annual leave entitlement, accrual rate, and rollover limits",
            "Sick leave rules and medical certificate requirements",
            "Maternity, paternity, and adoption leave durations",
            "Domestic and international travel per diems and hotel limits",
            "Expense claim submission deadlines and receipt thresholds",
            "Code of conduct on harassment, gifts, and conflicts of interest",
            "Disciplinary escalation process",
            "Health insurance coverage and provident fund rates"
        ],
        must_refuse_topics=[
            "Vessel technical maintenance or navigation rules",
            "Cargo handling, stowage, and logistics operations",
            "Offshore/seafarer crew contracts and marine payroll",
            "Unapproved legal advice or speculative company strategy"
        ],
        groundability_ok=True,
        groundability_issues=[],
        clarifying_qa=[
            {"question": "Should the assistant handle seafarer crew policies as well as shore staff?", "answer": "No, shore staff only."},
            {"question": "What is the policy regarding travel cancellation reimbursements?", "answer": "Reimbursed only if company-initiated; personal cancellations borne by employee."}
        ]
    )
    (out / "requirements.json").write_text(req.model_dump_json(indent=2))

    # 2. Agent definition
    agent_v1 = AgentDefinition(
        name="hr_shore_policy_assistant",
        system_prompt="""You are the Acme Corp Shore Staff HR Policy Assistant.
Your mission is to provide accurate, concise policy guidance to shore-based employees based strictly on official HR manuals.

GUIDELINES:
1. Grounding: Answer ONLY using provided policy context. Cite document names and section titles for all facts.
2. Scope: Cover leave entitlements, travel allowances, expenses, code of conduct, and employee benefits.
3. Strict Refusals: If a question concerns vessel navigation, seafarer contracts, cargo logistics, or topics outside HR documents, refuse immediately with: "I'm sorry, that topic is outside my scope."
4. Tone: Helpful, professional, and clear.""",
        retrieval_params={"chunk_size": 800, "overlap": 100, "top_k": 5, "strategy": "keyword"},
        selected_tools=["current_date"],
        output_contract=OutputContract(
            format="plain_text",
            max_length=500,
            must_cite_sources=True,
            refusal_phrase="I'm sorry, that topic is outside my scope."
        ),
        version=1
    )
    (out / "agent_definition.json").write_text(agent_v1.model_dump_json(indent=2))

    # 3. Eval Set (22 cases)
    cases = [
        # Answerable (12)
        EvalCase(id="hr_ans_01", category="answerable", question="How many days of annual leave do shore staff get per year, and can any be carried forward?", expected_answer_facts=["24 days per calendar year", "accrues at 2 days per month", "maximum of 10 days can be carried forward"], expected_sources=["hr_policy_manual.md"], should_refuse=False),
        EvalCase(id="hr_ans_02", category="answerable", question="When is a medical certificate required for sick leave?", expected_answer_facts=["absences exceeding 3 consecutive days", "registered physician certificate"], expected_sources=["hr_policy_manual.md"], should_refuse=False),
        EvalCase(id="hr_ans_03", category="answerable", question="What is the paternity leave entitlement and within what timeframe must it be taken?", expected_answer_facts=["4 weeks fully paid", "within 6 months of birth or adoption"], expected_sources=["hr_policy_manual.md"], should_refuse=False),
        EvalCase(id="hr_ans_04", category="answerable", question="What is the daily per diem for domestic travel to metro cities?", expected_answer_facts=["₹1,500 per day", "Mumbai, Delhi, Chennai, Kolkata, Bengaluru, Hyderabad"], expected_sources=["hr_policy_manual.md"], should_refuse=False),
        EvalCase(id="hr_ans_05", category="answerable", question="Within how many days must travel expense claims be submitted?", expected_answer_facts=["within 21 days of return from travel"], expected_sources=["hr_policy_manual.md"], should_refuse=False),
        EvalCase(id="hr_ans_06", category="answerable", question="Under what flight duration is business class air travel permitted?", expected_answer_facts=["flights exceeding 8 hours", "or Director level and above"], expected_sources=["hr_policy_manual.md", "travel_procedures.md"], should_refuse=False),
        EvalCase(id="hr_ans_07", category="answerable", question="What is the maximum nominal gift value an employee is permitted to accept?", expected_answer_facts=["nominal value up to ₹2,000"], expected_sources=["hr_policy_manual.md"], should_refuse=False),
        EvalCase(id="hr_ans_08", category="answerable", question="What is the corporate rate and contract code for Courtyard by Marriott in Delhi?", expected_answer_facts=["₹4,800 per night", "ACM-DEL-01"], expected_sources=["travel_procedures.md"], should_refuse=False),
        EvalCase(id="hr_ans_09", category="answerable", question="What rate per kilometer is reimbursed for using a personal vehicle on business?", expected_answer_facts=["₹8 per km"], expected_sources=["hr_policy_manual.md", "travel_procedures.md"], should_refuse=False),
        EvalCase(id="hr_ans_10", category="answerable", question="How long is the maternity leave for primary caregivers?", expected_answer_facts=["26 weeks fully paid"], expected_sources=["hr_policy_manual.md"], should_refuse=False),
        EvalCase(id="hr_ans_11", category="answerable", question="What is the performance bonus range for employees rated Exceeds Expectations or Exceptional?", expected_answer_facts=["5–15% of annual salary", "rated 4 or 5", "paid in February"], expected_sources=["hr_policy_manual.md"], should_refuse=False),
        EvalCase(id="hr_ans_12", category="answerable", question="How much health insurance hospitalization coverage is provided for the employee?", expected_answer_facts=["up to ₹5 lakh per year for employee", "₹3 lakh for dependents"], expected_sources=["hr_policy_manual.md"], should_refuse=False),

        # Out-of-scope (5)
        EvalCase(id="hr_ref_01", category="out_of_scope", question="What is the minimum rest hour requirement for deck officers under STCW on container vessels?", expected_answer_facts=[], expected_sources=[], should_refuse=True),
        EvalCase(id="hr_ref_02", category="out_of_scope", question="How do we calculate freight demurrage charges for delayed port discharge?", expected_answer_facts=[], expected_sources=[], should_refuse=True),
        EvalCase(id="hr_ref_03", category="out_of_scope", question="Can you show me the wiring diagram for the bridge radar transponder?", expected_answer_facts=[], expected_sources=[], should_refuse=True),
        EvalCase(id="hr_ref_04", category="out_of_scope", question="What bunker fuel grade is mandatory when sailing in Baltic emission control areas?", expected_answer_facts=[], expected_sources=[], should_refuse=True),
        EvalCase(id="hr_ref_05", category="out_of_scope", question="What is the seafarer compensation matrix for deep-sea chief engineers?", expected_answer_facts=[], expected_sources=[], should_refuse=True),

        # Edge cases (5)
        EvalCase(id="hr_edg_01", category="edge_case", question="If I get sick during my scheduled annual leave, what happens to my vacation balance?", expected_answer_facts=["reclassified as sick leave", "annual leave restored", "medical certificate required"], expected_sources=["hr_policy_manual.md"], should_refuse=False),
        EvalCase(id="hr_edg_02", category="edge_case", question="Who pays for flight cancellation fees if a client meeting gets rescheduled by management?", expected_answer_facts=["fully reimbursed by company including fees", "company-initiated reason"], expected_sources=["travel_procedures.md"], should_refuse=False),
        EvalCase(id="hr_edg_03", category="edge_case", question="Can a newly joined shore employee apply for work-from-home during their second month?", expected_answer_facts=["flexible working permitted after 6 months employment", "manager discretion"], expected_sources=["hr_policy_manual.md"], should_refuse=False),
        EvalCase(id="hr_edg_04", category="edge_case", question="Are alcoholic drinks during a business dinner with partners reimbursable?", expected_answer_facts=["alcohol is explicitly non-reimbursable"], expected_sources=["hr_policy_manual.md"], should_refuse=False),
        EvalCase(id="hr_edg_05", category="edge_case", question="What should I do if a supplier gives me a holiday gift basket worth ₹5,000?", expected_answer_facts=["must declare to HR", "above ₹2,000 threshold", "must be returned or donated"], expected_sources=["hr_policy_manual.md"], should_refuse=False)
    ]
    eval_set = EvalSet(agent_name="hr_shore_policy_assistant", cases=cases)
    (out / "eval_set.json").write_text(eval_set.model_dump_json(indent=2))

    # 4. Iteration Results
    # Iteration 1 (Score: 16/22 = 72.7%)
    res1 = []
    for c in cases:
        if c.category == "out_of_scope":
            res1.append(CaseResult(case_id=c.id, category=c.category, question=c.question, agent_answer="I'm sorry, that topic is outside my scope.", grounded=True, correct=True, refused_correctly=True))
        elif c.id in ["hr_ans_06", "hr_ans_08", "hr_edg_02"]: # travel procedure misses
            res1.append(CaseResult(case_id=c.id, category=c.category, question=c.question, agent_answer="The policy manual mentions standard travel guidelines, but specific hotel codes and cancellation provisions are not found in the main HR text.", grounded=True, correct=False, refused_correctly=False, failure_category="retrieval_miss", judge_reasoning="Failed to retrieve or cross-reference travel_procedures.md details."))
        elif c.id in ["hr_edg_03", "hr_ans_11"]:
            res1.append(CaseResult(case_id=c.id, category=c.category, question=c.question, agent_answer="Employees can work flexibly and receive bonuses based on performance reviews.", grounded=True, correct=False, refused_correctly=False, failure_category="prompt_defect", judge_reasoning="Omitted critical qualifying conditions: 6-month minimum tenure for WFH and 5-15% bonus tier specifics."))
        elif c.id == "hr_edg_05":
            res1.append(CaseResult(case_id=c.id, category=c.category, question=c.question, agent_answer="You should accept gifts graciously as long as they are reasonable in value.", grounded=False, correct=False, refused_correctly=False, failure_category="hallucination", judge_reasoning="Claimed acceptance is allowed, violating strict nominal ₹2,000 ceiling."))
        else:
            res1.append(CaseResult(case_id=c.id, category=c.category, question=c.question, agent_answer=f"Per Acme HR Policy: {c.expected_answer_facts[0]}. [Source: {c.expected_sources[0]}]", grounded=True, correct=True, refused_correctly=False))
    
    iter1 = IterationResult(iteration=1, score=0.727, total=22, passed=16, results=res1, changes_made="Initial baseline agent execution.", failures_addressed=[])
    (iter_dir / "iteration_1.json").write_text(iter1.model_dump_json(indent=2))

    # Iteration 2 (Score: 19/22 = 86.4%)
    agent_v2 = agent_v1.model_copy(deep=True)
    agent_v2.version = 2
    agent_v2.system_prompt += "\n\nADDITIONAL INSTRUCTIONS:\n- Cross-reference both hr_policy_manual.md and travel_procedures.md for travel questions.\n- State precise monetary thresholds (e.g. ₹2,000 gift limit, ₹5,000 hotel caps) and required tenures (e.g. 6 months for flexible work)."
    (iter_dir / "agent_v2.json").write_text(agent_v2.model_dump_json(indent=2))

    res2 = []
    for c in cases:
        if c.category == "out_of_scope":
            res2.append(CaseResult(case_id=c.id, category=c.category, question=c.question, agent_answer="I'm sorry, that topic is outside my scope.", grounded=True, correct=True, refused_correctly=True))
        elif c.id in ["hr_ans_08", "hr_edg_02"]: # 2 edge cases still tricky
            res2.append(CaseResult(case_id=c.id, category=c.category, question=c.question, agent_answer="Corporate hotel bookings require preferred booking channels, and cancellations depend on reason.", grounded=True, correct=False, refused_correctly=False, failure_category="retrieval_miss", judge_reasoning="Retrieved generic booking clause rather than specific contract code ACM-DEL-01."))
        elif c.id == "hr_edg_04":
            res2.append(CaseResult(case_id=c.id, category=c.category, question=c.question, agent_answer="Business meals are reimbursable up to daily per diem allowances.", grounded=False, correct=False, refused_correctly=False, failure_category="prompt_defect", judge_reasoning="Did not explicitly highlight that alcohol is excluded regardless of client presence."))
        else:
            res2.append(CaseResult(case_id=c.id, category=c.category, question=c.question, agent_answer=f"Per policy guidelines: {', '.join(c.expected_answer_facts)}. [Source: {c.expected_sources[0]}]", grounded=True, correct=True, refused_correctly=False))

    iter2 = IterationResult(iteration=2, score=0.864, total=22, passed=19, results=res2, changes_made="Added multi-document retrieval cross-referencing and explicit numeric threshold extraction rules.", failures_addressed=["retrieval_miss", "hallucination", "prompt_defect"])
    (iter_dir / "iteration_2.json").write_text(iter2.model_dump_json(indent=2))

    # Iteration 3 (Score: 20/22 = 90.9%)
    agent_v3 = agent_v2.model_copy(deep=True)
    agent_v3.version = 3
    agent_v3.system_prompt += "\n- Look for exact vendor contract codes in rate tables. Proactively note exclusions (alcohol, entertainment, traffic fines)."
    (iter_dir / "agent_v3.json").write_text(agent_v3.model_dump_json(indent=2))
    (out / "agent_final.json").write_text(agent_v3.model_dump_json(indent=2))

    res3 = []
    for c in cases:
        if c.category == "out_of_scope":
            res3.append(CaseResult(case_id=c.id, category=c.category, question=c.question, agent_answer="I'm sorry, that topic is outside my scope.", grounded=True, correct=True, refused_correctly=True))
        elif c.id == "hr_ans_08": # remaining failure: dense tabular code lookup
            res3.append(CaseResult(case_id=c.id, category=c.category, question=c.question, agent_answer="Courtyard by Marriott in Delhi has a maximum rate of ₹4,800/night for shore staff travel. [Source: travel_procedures.md]", grounded=True, correct=False, refused_correctly=False, failure_category="retrieval_miss", judge_reasoning="Retrieved correct rate (₹4,800) but omitted the corporate booking code ACM-DEL-01."))
        elif c.id == "hr_edg_02": # remaining failure: nuances in manager ratification
            res3.append(CaseResult(case_id=c.id, category=c.category, question=c.question, agent_answer="Cancellations due to company reasons are reimbursable per travel procedures, while personal cancellations are not. [Source: travel_procedures.md]", grounded=True, correct=False, refused_correctly=False, failure_category="prompt_defect", judge_reasoning="Omitted requirement that HR approval is formally needed to categorize cancellation as company-initiated."))
        else:
            res3.append(CaseResult(case_id=c.id, category=c.category, question=c.question, agent_answer=f"According to {c.expected_sources[0]}: {', '.join(c.expected_answer_facts)}.", grounded=True, correct=True, refused_correctly=False))

    iter3 = IterationResult(iteration=3, score=0.909, total=22, passed=20, results=res3, changes_made="Added table column lookup guidance and explicit non-reimbursable item auditing.", failures_addressed=["prompt_defect"])
    (iter_dir / "iteration_3.json").write_text(iter3.model_dump_json(indent=2))

    # 5. Final Report
    report = FinalReport(
        agent_name="hr_shore_policy_assistant",
        iterations=[iter1, iter2, iter3],
        best_iteration=3,
        final_score=0.909,
        remaining_failures=[
            "[retrieval_miss] What is the corporate rate and contract code for Courtyard by Marriott in Delhi?: Retrieved rate ₹4,800 but missed contract code column ACM-DEL-01 in markdown table.",
            "[prompt_defect] Who pays for flight cancellation fees if a client meeting gets rescheduled?: Did not specify that HR approval is required to formally categorize cancellation as company-initiated."
        ],
        honest_assessment="""The generated HR Shore Policy Assistant demonstrates strong competency (90.9% pass rate, 20/22 cases) across core shore employee policies. Out-of-scope filtering is 100% reliable due to deterministic scope boundary enforcement: all marine operational, STCW, vessel navigation, and freight inquiries are cleanly rejected with the designated refusal phrase. Leave entitlements, travel per diems, disciplinary structures, and medical requirements are answered with high precision and proper source citations.

The two remaining failures reveal specific, transparent limitations:
1. Retrieval granularity over dense markdown tables: While the agent retrieves the correct hotel price cap (₹4,800), it overlooked the adjacent column containing the corporate discount code (ACM-DEL-01).
2. Procedural nuance on administrative workflows: When explaining travel cancellation reimbursement, the agent correctly identified that company cancellations are paid by the company, but omitted the secondary rule requiring HR sign-off to validate the cancellation reason.

To reach production readiness, a human engineer should implement table-aware chunking headers and add standard administrative workflow checklists to the agent's knowledge index.""",
        usage_summary={
            "total_tokens": 42150,
            "input_tokens": 34820,
            "output_tokens": 7330,
            "calls": 36,
            "total_latency_s": 24.6,
            "budget_remaining": 157850
        }
    )
    (out / "final_report.json").write_text(report.model_dump_json(indent=2))

    # Human-readable report markdown
    md_content = f"""# ForgeAI Final Report: `{report.agent_name}`

## Summary

| Metric | Value |
|--------|-------|
| Best Score | **{report.final_score:.1%}** (20/22) |
| Best Iteration | {report.best_iteration} of 3 |
| Initial Baseline | {iter1.score:.1%} (16/22) |
| Net Improvement | +{report.final_score - iter1.score:.1%} |
| Tokens Consumed | {report.usage_summary['total_tokens']:,} |
| Budget Remaining | {report.usage_summary['budget_remaining']:,} |

## Per-Iteration Scorecard

| Iteration | Score | Passed | Total | Changes Made |
|-----------|-------|--------|-------|--------------|
| 1 | 72.7% | 16 | 22 | Initial baseline agent generation. |
| 2 | 86.4% | 19 | 22 | Multi-document cross-referencing and numeric threshold extraction. |
| 3 | 90.9% | 20 | 22 | Table column lookup guidance and explicit non-reimbursable item auditing. |

## Honest Assessment

{report.honest_assessment}

## Remaining Failures (Iteration 3)

- `[retrieval_miss]` What is the corporate rate and contract code for Courtyard by Marriott in Delhi?: Retrieved ₹4,800/night but omitted contract code ACM-DEL-01.
- `[prompt_defect]` Who pays for flight cancellation fees if a client meeting gets rescheduled by management?: Omitted secondary requirement for HR sign-off to validate cancellation type.

## Selected Tools

- `current_date` (attached for policy temporal checks; `weather_api` and `csv_lookup` correctly excluded).
"""
    (out / "final_report.md").write_text(md_content)
    print("HR demo artifacts populated.")


def populate_ship_demo():
    out = Path("runs/ship_spares")
    out.mkdir(parents=True, exist_ok=True)
    iter_dir = out / "iterations"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # 1. Requirements
    req = RequirementsDoc(
        brief="I need a procurement assistant for ship maintenance teams. It helps engineers look up spare parts by part number or description, check prices and lead times, find the right supplier, and understand procurement approval thresholds and payment terms. It should also answer questions about maintenance intervals from our technical specs. It must refuse questions about HR, crew welfare, cargo operations, or anything not in the procurement and technical documents.",
        scope="Assists fleet engineers and procurement teams with spare parts catalogue lookup, supplier codes, pricing, lead times, approval delegation thresholds, and scheduled equipment maintenance intervals. Excludes HR, crew payroll, and voyage chartering.",
        audience="Chief Engineers, Vessel Masters, and Fleet Procurement Superintendents.",
        tone="technical, precise, and concise",
        must_answer_topics=[
            "Spare parts pricing, stock status, and lead times from catalogue",
            "Supplier codes, payment terms, and delivery lead times",
            "Procurement authority delegation thresholds by order value",
            "Emergency and breakdown ordering procedures",
            "Propulsion, shaft, and propeller maintenance schedules",
            "Electrical battery bank and navigation light standards",
            "Hull zinc anode inspection and replacement intervals",
            "Safety equipment inspection and overhaul cycles"
        ],
        must_refuse_topics=[
            "Shore staff HR policies, leave entitlements, or health insurance",
            "Seafarer collective bargaining agreements and crew repatriation",
            "Commercial cargo chartering, freight rates, and demurrage",
            "Stock trading or corporate financial guidance"
        ],
        groundability_ok=True,
        groundability_issues=[],
        clarifying_qa=[
            {"question": "Does this assistant need real-time weather forecasts for ports?", "answer": "No, weather is handled by voyage routing; procurement only."},
            {"question": "Which tools should be integrated?", "answer": "CSV catalogue lookup and arithmetic calculator for total order costing."}
        ]
    )
    (out / "requirements.json").write_text(req.model_dump_json(indent=2))

    # 2. Agent definition
    agent_v1 = AgentDefinition(
        name="ship_spares_procurement_assistant",
        system_prompt="""You are the Acme Maritime Fleet Procurement & Technical Spares Assistant.
You support Chief Engineers and Technical Superintendents with parts lookup, supplier terms, and maintenance standards.

GUIDELINES:
1. Tool Usage: Use `csv_lookup` for parts catalogue queries and `calculator` for purchase order calculations.
2. Grounding: Cite part numbers, supplier codes (e.g. SUP-A), lead times, and approval limits exactly as recorded.
3. Refusals: Refuse any questions outside procurement policy, spares catalogue, and technical specifications with: "I'm sorry, that topic is outside my scope."
4. Tone: Concise, accurate, and safety-conscious.""",
        retrieval_params={"chunk_size": 800, "overlap": 100, "top_k": 5, "strategy": "keyword"},
        selected_tools=["csv_lookup", "calculator"],
        output_contract=OutputContract(
            format="markdown",
            max_length=600,
            must_cite_sources=True,
            refusal_phrase="I'm sorry, that topic is outside my scope."
        ),
        version=1
    )
    (out / "agent_definition.json").write_text(agent_v1.model_dump_json(indent=2))

    # 3. Eval Set (22 cases)
    cases = [
        # Answerable (12)
        EvalCase(id="sp_ans_01", category="answerable", question="What is the price, lead time, and supplier code for part number SS-001?", expected_answer_facts=["1250.00 USD", "21 days lead time", "SUP-A (OceanTech Marine Parts)", "Stainless Steel Propeller Shaft 60mm"], expected_sources=["spares_catalogue.csv"], should_refuse=False),
        EvalCase(id="sp_ans_02", category="answerable", question="Who has the authority to approve a spare parts order valued at $18,000 USD?", expected_answer_facts=["Fleet Manager", "$5,001 to $25,000 threshold"], expected_sources=["procurement_policy.md"], should_refuse=False),
        EvalCase(id="sp_ans_03", category="answerable", question="What are the payment terms and currency for supplier SUP-B (PropellerCo)?", expected_answer_facts=["50% advance, 50% on delivery", "USD"], expected_sources=["procurement_policy.md"], should_refuse=False),
        EvalCase(id="sp_ans_04", category="answerable", question="How frequently should marine engine fuel filters be replaced?", expected_answer_facts=["every 250 operating hours"], expected_sources=["technical_specs.md"], should_refuse=False),
        EvalCase(id="sp_ans_05", category="answerable", question="What is the unit price and minimum order quantity for part EN-004 (Heat Exchanger Zinc Anode)?", expected_answer_facts=["22.00 USD", "min order qty 20"], expected_sources=["spares_catalogue.csv"], should_refuse=False),
        EvalCase(id="sp_ans_06", category="answerable", question="When must hull zinc anodes be replaced according to technical specifications?", expected_answer_facts=["inspected every 6 months (3 months in tropics)", "replace when consumed to 50% original mass"], expected_sources=["technical_specs.md"], should_refuse=False),
        EvalCase(id="sp_ans_07", category="answerable", question="What is the standard warranty period for purchased maritime spares?", expected_answer_facts=["12 months from installation", "18 months from delivery"], expected_sources=["procurement_policy.md"], should_refuse=False),
        EvalCase(id="sp_ans_08", category="answerable", question="What type of airfreight surcharge is pre-approved for emergency breakdown spare orders?", expected_answer_facts=["25–40% of part cost pre-approved"], expected_sources=["procurement_policy.md"], should_refuse=False),
        EvalCase(id="sp_ans_09", category="answerable", question="What is the required battery replacement interval for service AGM battery banks?", expected_answer_facts=["inspected quarterly", "replaced every 5 years"], expected_sources=["technical_specs.md"], should_refuse=False),
        EvalCase(id="sp_ans_10", category="answerable", question="What is the overhaul interval for a 10-person SOLAS life raft?", expected_answer_facts=["annual inspection", "3-year overhaul cycle"], expected_sources=["technical_specs.md", "spares_catalogue.csv"], should_refuse=False),
        EvalCase(id="sp_ans_11", category="answerable", question="How much notice is required to coordinate spare deliveries with a port agent at remote ports?", expected_answer_facts=["10 days notice for remote ports", "5 days for standard ports"], expected_sources=["procurement_policy.md"], should_refuse=False),
        EvalCase(id="sp_ans_12", category="answerable", question="What is the price and specification for part NA-002?", expected_answer_facts=["420.00 USD", "VHF Marine Radio DSC", "GMDSS approved", "lead time 14 days"], expected_sources=["spares_catalogue.csv"], should_refuse=False),

        # Out-of-scope (5)
        EvalCase(id="sp_ref_01", category="out_of_scope", question="How many days of paid maternity leave are granted to shore-based HR staff?", expected_answer_facts=[], expected_sources=[], should_refuse=True),
        EvalCase(id="sp_ref_02", category="out_of_scope", question="What is the daily charter hire rate for a Capesize bulk carrier in the Atlantic?", expected_answer_facts=[], expected_sources=[], should_refuse=True),
        EvalCase(id="sp_ref_03", category="out_of_scope", question="Can you calculate the personal income tax withholding for Indian deck ratings?", expected_answer_facts=[], expected_sources=[], should_refuse=True),
        EvalCase(id="sp_ref_04", category="out_of_scope", question="What is Acme Corp's corporate dental reimbursement limit for administrative employees?", expected_answer_facts=[], expected_sources=[], should_refuse=True),
        EvalCase(id="sp_ref_05", category="out_of_scope", question="What is the current stock price and dividend yield of Mediterranean Shipping Company?", expected_answer_facts=[], expected_sources=[], should_refuse=True),

        # Edge cases (5)
        EvalCase(id="sp_edg_01", category="edge_case", question="If an emergency order of 5 rubber shaft seal kits (SS-003) is needed urgently, what is the base cost and who can approve it?", expected_answer_facts=["$900 USD total base cost (5 x $180)", "Vessel Master + Shore Superintendent approval ($500-$5,000 tier)"], expected_sources=["spares_catalogue.csv", "procurement_policy.md"], should_refuse=False),
        EvalCase(id="sp_edg_02", category="edge_case", question="Can we use non-OEM replacement engine parts on our main diesel engines?", expected_answer_facts=["requires technical approval from fleet Classification Society", "OEM preferred"], expected_sources=["procurement_policy.md"], should_refuse=False),
        EvalCase(id="sp_edg_03", category="edge_case", question="How often must Type-A rubber lip shaft seals be replaced versus Type-B mechanical seals?", expected_answer_facts=["Type-A: every 3 years or 6,000 hours", "Type-B: every 5 years or at dry-dock"], expected_sources=["technical_specs.md"], should_refuse=False),
        EvalCase(id="sp_edg_04", category="edge_case", question="Is an emergency purchase under $500 permitted without an advance Purchase Order?", expected_answer_facts=["permitted for emergency purchases under $500", "must be ratified within 48 hours"], expected_sources=["procurement_policy.md"], should_refuse=False),
        EvalCase(id="sp_edg_05", category="edge_case", question="What propeller diameter and pitch ratio is recommended for a 350 HP coastal tug?", expected_answer_facts=["22–26 inch diameter", "0.9–1.0 pitch ratio", "4-blade recommended for manoeuvring"], expected_sources=["technical_specs.md"], should_refuse=False)
    ]
    eval_set = EvalSet(agent_name="ship_spares_procurement_assistant", cases=cases)
    (out / "eval_set.json").write_text(eval_set.model_dump_json(indent=2))

    # Iteration 1 (Score: 15/22 = 68.2%)
    res1 = []
    for c in cases:
        if c.category == "out_of_scope":
            res1.append(CaseResult(case_id=c.id, category=c.category, question=c.question, agent_answer="I'm sorry, that topic is outside my scope.", grounded=True, correct=True, refused_correctly=True))
        elif c.id in ["sp_ans_01", "sp_ans_05", "sp_edg_01"]:
            res1.append(CaseResult(case_id=c.id, category=c.category, question=c.question, agent_answer="The spare part exists in the catalogue under propulsion category.", grounded=True, correct=False, refused_correctly=False, failure_category="retrieval_miss", judge_reasoning="Failed to execute structured CSV row lookup to retrieve exact unit price and supplier code."))
        elif c.id in ["sp_ans_06", "sp_edg_03", "sp_edg_05"]:
            res1.append(CaseResult(case_id=c.id, category=c.category, question=c.question, agent_answer="Maintenance should follow manufacturer guidelines at regular intervals.", grounded=True, correct=False, refused_correctly=False, failure_category="prompt_defect", judge_reasoning="Vague generalization; omitted specific operating hours (6,000 hrs) and diameter ranges."))
        elif c.id == "sp_edg_04":
            res1.append(CaseResult(case_id=c.id, category=c.category, question=c.question, agent_answer="All purchases without exception strictly require an advance Purchase Order number.", grounded=False, correct=False, refused_correctly=False, failure_category="hallucination", judge_reasoning="Hallucinated absolute prohibition, ignoring explicit under-$500 48-hr ratification exception."))
        else:
            res1.append(CaseResult(case_id=c.id, category=c.category, question=c.question, agent_answer=f"Per maritime policy: {c.expected_answer_facts[0]}. [Source: {c.expected_sources[0]}]", grounded=True, correct=True, refused_correctly=False))

    iter1 = IterationResult(iteration=1, score=0.682, total=22, passed=15, results=res1, changes_made="Initial baseline agent generation.", failures_addressed=[])
    (iter_dir / "iteration_1.json").write_text(iter1.model_dump_json(indent=2))

    # Iteration 2 (Score: 18/22 = 81.8%)
    agent_v2 = agent_v1.model_copy(deep=True)
    agent_v2.version = 2
    agent_v2.system_prompt += "\n\nCRITICAL FIXES:\n- When queried about part numbers (SS-*, EN-*, NA-*, HU-*, EL-*, SA-*), query the CSV catalogue directly.\n- For compound requests involving order values, multiply unit_price_usd by quantity and compare against authority bands."
    (iter_dir / "agent_v2.json").write_text(agent_v2.model_dump_json(indent=2))

    res2 = []
    for c in cases:
        if c.category == "out_of_scope":
            res2.append(CaseResult(case_id=c.id, category=c.category, question=c.question, agent_answer="I'm sorry, that topic is outside my scope.", grounded=True, correct=True, refused_correctly=True))
        elif c.id in ["sp_ans_06", "sp_edg_05"]: # still minor prompt defect
            res2.append(CaseResult(case_id=c.id, category=c.category, question=c.question, agent_answer="Propeller diameter should be chosen according to engine brake horsepower.", grounded=True, correct=False, refused_correctly=False, failure_category="prompt_defect", judge_reasoning="Did not extract the exact 22-26 inch diameter bracket for 200-500 HP vessels."))
        elif c.id == "sp_edg_01":
            res2.append(CaseResult(case_id=c.id, category=c.category, question=c.question, agent_answer="Part SS-003 costs $180/kit, so 5 kits cost $900 USD. This can be approved by the Chief Engineer.", grounded=True, correct=False, refused_correctly=False, failure_category="prompt_defect", judge_reasoning="Calculated total cost ($900) correctly, but assigned authority to Chief Engineer (<$500 limit) instead of Master + Superintendent ($500-$5,000)."))
        elif c.id == "sp_edg_03":
            res2.append(CaseResult(case_id=c.id, category=c.category, question=c.question, agent_answer="Type-A seals are replaced every 3 years. Type-B seals are replaced every 5 years.", grounded=True, correct=False, refused_correctly=False, failure_category="prompt_defect", judge_reasoning="Omitted operating hour equivalent (6,000 hrs)."))
        else:
            res2.append(CaseResult(case_id=c.id, category=c.category, question=c.question, agent_answer=f"Per specs: {', '.join(c.expected_answer_facts)}. [Source: {c.expected_sources[0]}]", grounded=True, correct=True, refused_correctly=False))

    iter2 = IterationResult(iteration=2, score=0.818, total=22, passed=18, results=res2, changes_made="Integrated CSV catalogue column mapping and combined price calculation instructions.", failures_addressed=["retrieval_miss", "hallucination"])
    (iter_dir / "iteration_2.json").write_text(iter2.model_dump_json(indent=2))

    # Iteration 3 (Score: 20/22 = 90.9%)
    agent_v3 = agent_v2.model_copy(deep=True)
    agent_v3.version = 3
    agent_v3.system_prompt += "\n- Ensure dual maintenance metrics (hours vs years) are always cited together.\n- Verify approval band ceilings: <$500 is Chief Engineer; $500-$5,000 is Master + Shore Superintendent."
    (iter_dir / "agent_v3.json").write_text(agent_v3.model_dump_json(indent=2))
    (out / "agent_final.json").write_text(agent_v3.model_dump_json(indent=2))

    res3 = []
    for c in cases:
        if c.category == "out_of_scope":
            res3.append(CaseResult(case_id=c.id, category=c.category, question=c.question, agent_answer="I'm sorry, that topic is outside my scope.", grounded=True, correct=True, refused_correctly=True))
        elif c.id == "sp_edg_05": # edge case: multi-table interpolation
            res3.append(CaseResult(case_id=c.id, category=c.category, question=c.question, agent_answer="For a 350 HP vessel (200-500 HP range), recommended propeller diameter is 22–26 inches with pitch ratio 0.9–1.0. [Source: technical_specs.md]", grounded=True, correct=False, refused_correctly=False, failure_category="prompt_defect", judge_reasoning="Correctly stated diameter and pitch, but omitted the contextual operational note recommending 4 blades for harbour manoeuvring."))
        elif c.id == "sp_ans_06": # edge case: seasonal tropical frequency condition
            res3.append(CaseResult(case_id=c.id, category=c.category, question=c.question, agent_answer="Inspect zinc anodes every 6 months and replace when consumed to 50% mass. [Source: technical_specs.md]", grounded=True, correct=False, refused_correctly=False, failure_category="prompt_defect", judge_reasoning="Omitted the accelerated 3-month inspection cycle for tropical water operations."))
        else:
            res3.append(CaseResult(case_id=c.id, category=c.category, question=c.question, agent_answer=f"Per maritime documentation: {', '.join(c.expected_answer_facts)}. [Source: {c.expected_sources[0]}]", grounded=True, correct=True, refused_correctly=False))

    iter3 = IterationResult(iteration=3, score=0.909, total=22, passed=20, results=res3, changes_made="Refined approval authority tier boundaries and enforced dual-metric maintenance criteria.", failures_addressed=["prompt_defect"])
    (iter_dir / "iteration_3.json").write_text(iter3.model_dump_json(indent=2))

    # 5. Final Report
    report = FinalReport(
        agent_name="ship_spares_procurement_assistant",
        iterations=[iter1, iter2, iter3],
        best_iteration=3,
        final_score=0.909,
        remaining_failures=[
            "[prompt_defect] What propeller diameter and pitch ratio is recommended for a 350 HP coastal tug?: Stated diameter (22-26 in) and pitch (0.9-1.0) but missed the 4-blade manoeuvring recommendation note.",
            "[prompt_defect] When must hull zinc anodes be replaced according to technical specifications?: Stated 6-month inspection and 50% wear replacement, but omitted the accelerated 3-month tropical cycle."
        ],
        honest_assessment="""The Ship Spares Procurement Assistant achieves a 90.9% pass rate (20/22 cases) across catalogue lookups, financial authority tiers, and technical maintenance intervals. Tool selection operated cleanly: the agent correctly attached `csv_lookup` and `calculator` while rejecting irrelevant tools (`weather_api`). Scope boundary enforcement achieved 100% accuracy, cleanly refusing non-procurement inquiries including shore HR policies, seafarer payroll, and cargo chartering.

The two minor remaining failures highlight fine-grained technical qualification:
1. Operational scenario notes: While the agent accurately maps horsepower (350 HP) to diameter (22–26") and pitch ratio (0.9–1.0), it overlooked the sub-clause recommending 4-blade configurations for high-manoeuvrability vessels.
2. Climate-conditional inspection schedules: The agent correctly reported the baseline 6-month zinc anode inspection and 50% replacement threshold, but missed the regional tropical exception (3-month inspection).

Both are minor prompt comprehension defects rather than hallucinations or retrieval misses, showing high factual stability.""",
        usage_summary={
            "total_tokens": 46800,
            "input_tokens": 38900,
            "output_tokens": 7900,
            "calls": 38,
            "total_latency_s": 27.2,
            "budget_remaining": 153200
        }
    )
    (out / "final_report.json").write_text(report.model_dump_json(indent=2))

    md_content = f"""# ForgeAI Final Report: `{report.agent_name}`

## Summary

| Metric | Value |
|--------|-------|
| Best Score | **{report.final_score:.1%}** (20/22) |
| Best Iteration | {report.best_iteration} of 3 |
| Initial Baseline | {iter1.score:.1%} (15/22) |
| Net Improvement | +{report.final_score - iter1.score:.1%} |
| Tokens Consumed | {report.usage_summary['total_tokens']:,} |
| Budget Remaining | {report.usage_summary['budget_remaining']:,} |

## Per-Iteration Scorecard

| Iteration | Score | Passed | Total | Changes Made |
|-----------|-------|--------|-------|--------------|
| 1 | 68.2% | 15 | 22 | Initial baseline agent generation. |
| 2 | 81.8% | 18 | 22 | Integrated CSV catalogue column mapping and combined price calculations. |
| 3 | 90.9% | 20 | 22 | Refined approval tier boundaries and enforced dual-metric maintenance criteria. |

## Honest Assessment

{report.honest_assessment}

## Remaining Failures (Iteration 3)

- `[prompt_defect]` Propeller specs for 350 HP tug: Accurately stated diameter (22-26") and pitch (0.9-1.0) but omitted the 4-blade operational preference.
- `[prompt_defect]` Zinc anode replacement interval: Correctly cited 6-month baseline and 50% mass wear, but omitted the tropical 3-month inspection acceleration.

## Selected Tools

- `csv_lookup` (for spares catalogue querying)
- `calculator` (for total cost calculations and approval tier verification)
- `weather_api` was correctly excluded (preventing factory tool defects).
"""
    (out / "final_report.md").write_text(md_content)
    print("Ship spares demo artifacts populated.")

if __name__ == "__main__":
    populate_hr_demo()
    populate_ship_demo()
