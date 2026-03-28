#!/usr/bin/env python3

import arxiv
import fitz
import faiss
import numpy as np
import json
import os
import argparse
import urllib.request
import tempfile
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

CHUNK_SIZE = 400 # tokens
CHUNK_OVERLAP = 50 # words of overlap b/w chunks
INDEX_DIR = "index"
EMBED_MODEL = "all-MiniLM-L6-v2"


def fetch_papers(topic: str, n: int) -> list[arxiv.Result]:
    console.print(f"\n[bold cyan] Searching Arxiv for: [/bold cyan] [yellow]{topic}[/yellow]")
    search = arxiv.Search(
        query=topic,
        max_results=n,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers = list(search.results())
    console.print(f"[green]found {len(papers)} papers[/green]")
    return papers


def download_and_extract(paper: arxiv.Result) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        urllib.request.urlretrieve(paper.pdf_url, tmp_path)
        doc = fitz.open(tmp_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    finally:
        os.unlink(tmp_path)

def chunk_text(text, chunk_size = CHUNK_SIZE, overlap = CHUNK_OVERLAP) -> list[str]:
    #split text into chunks
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks

def build_index(topic, n):
    papers = fetch_papers(topic, n)
    model = SentenceTransformer(EMBED_MODEL)

    all_chunks = []
    all_metadata = []

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
    ) as progress:
        task = progress.add_task("[cyan]processing: processing papers...", total=len(papers))

        for paper in papers:
            progress.update(task, description=f"[cyan]Processing: [yellow]{paper.title[:50]}...")
            try:
                text = download_and_extract(paper)
                chunks = chunk_text(text)
                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadata.append({
                        "title": paper.title,
                        "authors": [a.name for a in paper.authors[:3]],
                        "arxiv_id": paper.entry_id.split("/")[-1],
                        "url": paper.entry_id,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    })
            except Exception as e:
                console.print(f"[red]failed to process {paper.title[:40]: {e}[/red]}")
            progress.advance(task)

    console.print(f"\n[green]Total chunks to embed: {len(all_chunks)}[/green]")
    console.print("[cyan]embedding chunks...[/cyan]")

    embeddings = model.encode(all_chunks, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings).astype("float32")


    # norm for cosine sim
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # store index and metadata
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))
    with open(os.path.join(INDEX_DIR, "metadata.json"), "w") as f:
        json.dump(all_metadata, f, indent=2)
    with open(os.path.join(INDEX_DIR, "chunks.json"), "w") as f:
        json.dump(all_chunks, f, indent=2)
    with open(os.path.join(INDEX_DIR, "topic.txt"), "w") as f:
        f.write(topic)

    console.print(f"\n[green]index built. {len(all_chunks)} chunks from {len(papers)} papers saved to ./{INDEX_DIR}/[/green]")
    console.print(f"[dim]Papers indexed: [/dim]")
    for p in papers:
        console.print(f"  [dim]• {p.title[:70]}[/dim]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ingest arxiv papers into a local rag index")
    parser.add_argument("--topic", required=True, help="Research topic to search on arxiv")
    parser.add_argument("--n", type=int, default=8, help="no. of papers to fetch (def:8)")
    args = parser.parse_args()
    build_index(args.topic, args.n)
