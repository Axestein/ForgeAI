import sys
sys.path.insert(0, '.')

from forgeai.core.models import RequirementsDoc, AgentDefinition, EvalCase, EvalSet, CaseResult, IterationResult, TOOL_REGISTRY, OutputContract
print('models OK -', len(TOOL_REGISTRY), 'tools in registry')

from forgeai.core.documents import load_documents, chunk_documents, keyword_search, build_context
docs = load_documents('demos/hr_assistant/docs')
print(f'HR docs loaded: {len(docs)} files')
chunks = chunk_documents(docs)
print(f'Chunks created: {len(chunks)}')
hits = keyword_search('annual leave policy', chunks, top_k=3)
print(f'Retrieval test: got {len(hits)} hits')
print(f'  Top hit: {hits[0]["source"]} - {hits[0]["text"][:80]}...')

docs2 = load_documents('demos/ship_spares/docs')
print(f'Ship spares docs loaded: {len(docs2)} files')
chunks2 = chunk_documents(docs2)
hits2 = keyword_search('propeller shaft price', chunks2, top_k=2)
print(f'Ship retrieval: {len(hits2)} hits')

from forgeai.tools.registry import calculator, current_date
calc_result = calculator('2 * 500 + 100')
print(f'Calculator: 2*500+100 = {calc_result["result"]}')
date_result = current_date()
print(f'Current date: {date_result["utc_date"]}')

from forgeai.core.judge import _is_refusal
assert _is_refusal("I'm sorry, that topic is outside my scope.") == True
assert _is_refusal("The annual leave is 24 days.") == False
print("Refusal regex: OK")

r = CaseResult(case_id='t1', category='answerable', question='q', agent_answer='a', grounded=True, correct=True)
assert r.passed == True
r2 = CaseResult(case_id='t2', category='out_of_scope', question='q', agent_answer='a', refused_correctly=True)
assert r2.passed == True
r3 = CaseResult(case_id='t3', category='answerable', question='q', agent_answer='a', grounded=False, correct=True)
assert r3.passed == False
print("CaseResult.passed logic: OK")

print()
print("ALL SMOKE TESTS PASSED")
