"""
ForgeAI Streamlit Web UI
Run with: streamlit run app.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from pathlib import Path

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ForgeAI – The Agent That Builds Agents",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar nav ───────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ ForgeAI")
    st.caption("The Agent That Builds Agents")
    page = st.radio(
        "Navigation",
        ["🏭 Build an Agent", "💬 Chat with Agent", "📊 View Reports"],
        index=0,
    )
    st.divider()
    st.markdown("**Configuration**")
    provider = st.selectbox("LLM Provider", ["openai", "google", "anthropic"])
    api_key = st.text_input(
        f"{provider.capitalize()} API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", "") if provider == "openai" else "",
    )
    model_name = st.text_input(
        "Model (leave blank for default)",
        value="",
        placeholder="e.g. gpt-4o-mini",
    )
    if st.button("Apply Config"):
        os.environ["LLM_PROVIDER"] = provider
        if api_key:
            key_map = {
                "openai": "OPENAI_API_KEY",
                "google": "GOOGLE_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
            }
            os.environ[key_map[provider]] = api_key
        if model_name:
            os.environ["LLM_MODEL"] = model_name
        st.success("Config applied!")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 – BUILD AN AGENT
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏭 Build an Agent":
    st.title("🏭 ForgeAI – Build an Agent")
    st.markdown("Provide a brief and upload your documents. ForgeAI will autonomously build, test, and improve an agent.")

    col1, col2 = st.columns([2, 1])

    with col1:
        brief = st.text_area(
            "Agent Brief",
            height=120,
            placeholder="e.g. I need an assistant that answers HR policy questions for shore staff, covering leave policies, travel reimbursements, and code of conduct.",
            help="One paragraph describing what the agent should do.",
        )

    with col2:
        uploaded_files = st.file_uploader(
            "Upload Documents",
            accept_multiple_files=True,
            type=["pdf", "txt", "md", "csv"],
            help="PDF, TXT, Markdown, or CSV files the agent should know about.",
        )

    adv_col1, adv_col2, adv_col3 = st.columns(3)
    with adv_col1:
        max_iter = st.slider("Max Iterations", 1, 3, 3)
    with adv_col2:
        eval_cases = st.slider("Eval Cases", 20, 30, 22)
    with adv_col3:
        tournament = st.checkbox("Prompt Tournament (stretch)", value=False)

    output_name = st.text_input("Run Name", value="my_agent_run", help="Output folder name under runs/")

    run_btn = st.button("🚀 Build Agent", type="primary", disabled=not (brief and uploaded_files))

    if run_btn:
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("GOOGLE_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
            st.error("Please set an API key in the sidebar first.")
        else:
            # Save uploaded files to temp directory
            with tempfile.TemporaryDirectory() as tmp_docs:
                for uf in uploaded_files:
                    dest = Path(tmp_docs) / uf.name
                    dest.write_bytes(uf.read())

                output_dir = str(Path("runs") / output_name)

                # Stream logs
                log_placeholder = st.empty()
                progress_placeholder = st.empty()

                log_lines = []

                class StreamCapture:
                    def write(self, text):
                        if text.strip():
                            log_lines.append(text.strip())
                            log_placeholder.code("\n".join(log_lines[-30:]), language="")
                    def flush(self): pass

                old_stdout = sys.stdout
                sys.stdout = StreamCapture()

                try:
                    from forgeai.pipeline import run_pipeline
                    with st.spinner("ForgeAI is building your agent..."):
                        result = run_pipeline(
                            brief=brief,
                            doc_folder=tmp_docs,
                            output_dir=output_dir,
                            interactive=False,  # No CLI prompts in Streamlit
                            max_iterations=max_iter,
                            tournament=tournament,
                            target_eval_cases=eval_cases,
                        )
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    result = None
                finally:
                    sys.stdout = old_stdout

                if result and "error" not in result:
                    st.success(f"✅ Agent built! Best score: {result['best_score']:.1%}")

                    # Show report
                    report_md_path = Path(result["artifacts"]["final_report_md"])
                    if report_md_path.exists():
                        with st.expander("📊 Final Report", expanded=True):
                            st.markdown(report_md_path.read_text(encoding="utf-8"))

                    st.info(f"All artifacts saved to: `{output_dir}`")

                    # Show download links
                    artifacts = result.get("artifacts", {})
                    if "agent_final" in artifacts and Path(artifacts["agent_final"]).exists():
                        agent_data = Path(artifacts["agent_final"]).read_bytes()
                        st.download_button("⬇ Download Agent Definition", agent_data, file_name="agent_final.json")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 – CHAT WITH AGENT
# ═══════════════════════════════════════════════════════════════════════════
elif page == "💬 Chat with Agent":
    st.title("💬 Chat with a ForgeAI Agent")

    col1, col2 = st.columns([1, 1])
    with col1:
        agent_file = st.file_uploader("Upload Agent Definition (.json)", type=["json"])
    with col2:
        doc_files = st.file_uploader("Upload Documents", accept_multiple_files=True, type=["pdf", "txt", "md", "csv"])

    # Also allow selecting from runs/
    runs_dir = Path("runs")
    existing_runs = [d for d in runs_dir.iterdir() if d.is_dir()] if runs_dir.exists() else []
    if existing_runs:
        run_names = ["(none)"] + [d.name for d in existing_runs]
        selected_run = st.selectbox("Or load from existing run", run_names)
    else:
        selected_run = "(none)"

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent_loaded" not in st.session_state:
        st.session_state.agent_loaded = False
    if "chat_chunks" not in st.session_state:
        st.session_state.chat_chunks = []
    if "chat_agent" not in st.session_state:
        st.session_state.chat_agent = None

    load_btn = st.button("Load Agent")
    if load_btn:
        try:
            from forgeai.core.documents import chunk_documents, load_documents
            from forgeai.core.models import AgentDefinition

            if selected_run != "(none)":
                run_path = runs_dir / selected_run
                agent_path = run_path / "agent_final.json"
                if not agent_path.exists():
                    agent_path = run_path / "agent_definition.json"
                agent_data = json.loads(agent_path.read_text(encoding="utf-8"))
                st.session_state.chat_agent = AgentDefinition(**agent_data)

                # Load docs from run folder (use a placeholder if none)
                doc_folder = str(run_path / "docs") if (run_path / "docs").exists() else str(run_path)
                docs = load_documents(doc_folder) if Path(doc_folder).exists() else []
            elif agent_file and doc_files:
                agent_data = json.loads(agent_file.read())
                st.session_state.chat_agent = AgentDefinition(**agent_data)

                with tempfile.TemporaryDirectory() as tmp:
                    for f in doc_files:
                        (Path(tmp) / f.name).write_bytes(f.read())
                    docs = load_documents(tmp)
            else:
                st.error("Please upload agent + docs, or select an existing run.")
                docs = []

            if docs:
                agent = st.session_state.chat_agent
                st.session_state.chat_chunks = chunk_documents(
                    docs,
                    chunk_size=agent.retrieval_params.get("chunk_size", 800),
                    overlap=agent.retrieval_params.get("overlap", 100),
                )
                st.session_state.agent_loaded = True
                st.session_state.chat_history = []
                st.success(f"✅ Loaded agent: {agent.name}")

        except Exception as e:
            st.error(f"Load error: {e}")

    if st.session_state.agent_loaded:
        agent = st.session_state.chat_agent
        st.info(f"**Agent:** {agent.name} | **Tools:** {', '.join(agent.selected_tools) or 'none'}")

        # Render history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("Ask the agent anything...")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    from forgeai.core.judge import _run_agent
                    answer = _run_agent(user_input, agent, st.session_state.chat_chunks)
                st.markdown(answer)

            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 – VIEW REPORTS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 View Reports":
    st.title("📊 ForgeAI Run Reports")

    runs_dir = Path("runs")
    if not runs_dir.exists() or not any(runs_dir.iterdir()):
        st.info("No runs found yet. Build an agent first!")
    else:
        runs = [d for d in runs_dir.iterdir() if d.is_dir()]
        run_names = [d.name for d in runs]
        selected = st.selectbox("Select Run", run_names)

        if selected:
            run_path = runs_dir / selected

            # Report markdown
            report_md = run_path / "final_report.md"
            if report_md.exists():
                st.markdown(report_md.read_text(encoding="utf-8"))
            else:
                st.warning("No final_report.md found for this run.")

            # Show iteration scorecards
            iter_dir = run_path / "iterations"
            if iter_dir.exists():
                st.subheader("Iteration Details")
                iter_files = sorted(iter_dir.glob("iteration_*.json"))
                for f in iter_files:
                    iter_data = json.loads(f.read_text(encoding="utf-8"))
                    with st.expander(f"Iteration {iter_data['iteration']} – Score: {iter_data['score']:.1%}"):
                        st.write(f"**Passed:** {iter_data['passed']}/{iter_data['total']}")
                        st.write(f"**Changes:** {iter_data.get('changes_made', '(initial run)')}")
                        # Show failures
                        failures = [r for r in iter_data["results"] if not (
                            (r["grounded"] and r["correct"]) if r["category"] != "out_of_scope"
                            else r["refused_correctly"]
                        )]
                        if failures:
                            st.markdown("**Failures:**")
                            for fail in failures[:5]:
                                st.markdown(f"- `[{fail['failure_category']}]` {fail['question'][:70]}")

            # Raw files
            with st.expander("📁 Raw Artifacts"):
                for f in sorted(run_path.rglob("*.json")):
                    rel = f.relative_to(run_path)
                    if st.button(f"📄 {rel}", key=str(rel)):
                        st.code(f.read_text(encoding="utf-8"), language="json")
