"""
Chat Runner – loads any produced AgentDefinition and runs it as a CLI chat.
Usage:
    python -m forgeai.runner.chat --agent path/to/agent_definition.json --docs path/to/docs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()


def load_agent(path: str):
    """Load an AgentDefinition from a JSON file."""
    # Import here to avoid circular imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from forgeai.core.models import AgentDefinition

    data = json.loads(Path(path).read_text())
    return AgentDefinition(**data)


def run_chat_cli(agent_path: str, doc_folder: str) -> None:
    """Interactive CLI chat runner."""
    from forgeai.core.documents import chunk_documents, load_documents
    from forgeai.core.judge import _run_agent
    from forgeai.core.models import AgentDefinition

    console.print(Panel.fit("[bold green]ForgeAI Chat Runner[/bold green]", border_style="green"))

    agent = load_agent(agent_path)
    console.print(f"[dim]Agent:[/dim] {agent.name} (v{agent.version})")
    console.print(f"[dim]Tools:[/dim] {', '.join(agent.selected_tools) or 'none'}")
    console.print(f"[dim]Docs:[/dim] {doc_folder}")
    console.print("[dim]Type 'quit' or 'exit' to end session.[/dim]\n")

    # Pre-load documents and chunks
    console.print("[yellow]Loading documents...[/yellow]")
    from forgeai.core.documents import load_documents, chunk_documents
    docs = load_documents(doc_folder)
    chunks = chunk_documents(
        docs,
        chunk_size=agent.retrieval_params.get("chunk_size", 800),
        overlap=agent.retrieval_params.get("overlap", 100),
    )
    console.print(f"[green]✓ Loaded {len(docs)} document(s), {len(chunks)} chunks.[/green]\n")

    history = []
    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Session ended.[/dim]")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        console.print("[yellow]Thinking...[/yellow]")
        answer = _run_agent(user_input, agent, chunks)

        console.print(Panel(
            Markdown(answer),
            title=f"[bold]{agent.name}[/bold]",
            border_style="blue",
        ))


def main():
    parser = argparse.ArgumentParser(description="ForgeAI Chat Runner")
    parser.add_argument("--agent", required=True, help="Path to agent_definition.json")
    parser.add_argument("--docs", required=True, help="Path to documents folder")
    args = parser.parse_args()

    if not Path(args.agent).exists():
        print(f"Error: agent file not found: {args.agent}")
        sys.exit(1)
    if not Path(args.docs).exists():
        print(f"Error: docs folder not found: {args.docs}")
        sys.exit(1)

    run_chat_cli(args.agent, args.docs)


if __name__ == "__main__":
    main()
