import argparse
import json
import urllib.request
import urllib.error
from retriever import retrieve, get_topic
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console()

OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = """You are a research assistant with access to a curated set of Arxiv research papers.
Your job is to answer the user's questions accurately, grounding every claim in the provided context.

Rules:
- Only use information from the provided context chunks. Do not use prior knowledge.
- Always cite the source paper for each claim using [Paper Title] inline.
- If the context doesn't contain enough information to answer, say so clearly.
- Be concise but precise. Prefer depth over breadth.
- For follow-up questions, use the conversation history to understand what the user is referring to."""


def ollama_chat_stream(messages, model):
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": True,
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as response:
            for line in response:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
    except urllib.error.URLError:
        console.print("\n[red]could not connect to ollama[/red]")
        console.print("[dim]start ollama with: ollama serve[/dim]")
        raise


def format_context(results: list) -> str:
    parts = []
    for i, r in enumerate(results):
        parts.append(
            f"--- SOURCE {i+1}: {r['title']} ({r['arxiv_id']}) ---\n{r['chunk']}"
        )
    return "\n\n".join(parts)


def print_sources(results: list):
    console.print("\n[dim]─── Sources ────[/dim]")
    seen = set()
    for r in results:
        if r["title"] not in seen:
            seen.add(r["title"])
            authors = ", ".join(r["authors"])
            console.print(f"[dim]  * [bold]{r['title'][:65]}[/bold][/dim]")
            console.print(f"[dim]    {authors} — {r['url']}[/dim]")
    console.print()


def chat(top_k: int, model: str):
    topic = get_topic()

    console.print(Panel(
        f"[bold cyan]arxiv chat[bold cyan]\n"
        f"[dim]Topic:[/dim]  [yellow]{topic}[/yellow]\n"
        f"[dim]Model:[/dim]  [yellow]{model} (local via Ollama)[/yellow]\n"
        f"[dim]Commands:[/dim] [bold]!quit[/bold] to exit",
        border_style="cyan",
    ))

    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    last_results = []

    while True:
        console.print()
        try:
            query = console.input("[bold green]You ->[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]goodbye[dim]")
            break

        if not query:
            continue
        if query.lower() in ("!quit", "!exit", "quit", "exit"):
            console.print("[dim]goodbye[/dim]")
            break
        if query.lower() == "!sources":
            if last_results:
                print_sources(last_results)
            else:
                console.print("[dim]no sources.[/dim]")
            continue

        results = retrieve(query, top_k=top_k)
        last_results = results

        if not results:
            console.print("[red]no relevant chunks found in the index.[/red]")
            continue

        context = format_context(results)
        user_message = (
            f"Context from research papers:\n\n{context}\n\n"
            f"Question: {query}"
        )

        messages = history + [{"role": "user", "content": user_message}]

        console.print()
        console.print(Rule(style="dim"))
        console.print("[bold cyan]Assistant ->[/bold cyan]")

        full_response = ""
        try:
            for token in ollama_chat_stream(messages, model):
                console.print(token, end="", highlight=False)
                full_response += token
        except urllib.error.URLError:
            continue

        console.print("\n")
        print_sources(results)

        # up historu
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": full_response})

        # keeping last 6 turns to avoid contxt overflow
        system_msg = history[0]
        turns = history[1:]
        if len(turns) > 12:
            turns = turns[-12:]
        history = [system_msg] + turns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with arxiv.")
    parser.add_argument("--top-k", type=int, default=6, help="chunks to retrieve per query (default: 6)")
    parser.add_argument("--model", type=str, default="llama3")
    args = parser.parse_args()
    chat(top_k=args.top_k, model=args.model)
