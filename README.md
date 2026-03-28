# Arxiv search 

A CLI tool that fetches papers from Arxiv on any topic, builds a local vector index, and lets you have a multi-turn conversation with source-cited answers grounded entirely in the papers.

## How it works

```
arxiv API → PDF download → text extraction → chunking → embedding → FAISS index
                                                                          ↓
                                              user query → retrieve top-k chunks → LLM → cited answer
```

**Stack:** Python · FAISS · sentence-transformers · PyMuPDF · Anthropic API · Rich(for stylized command line prompts)

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Step 1 — Ingest papers on a topic:**
```bash
python ingest.py --topic "LLM agents memory" --n 10
```

**Step 2 — Chat:**
```bash
python chat.py
```

### Commands
| Command | Description |
|---------|-------------|
| `!sources` | Show full source list from last answer |
| `!quit` | Exit |

### Options
```bash
python ingest.py --topic "topic" --n 15      # fetch 15 papers
python chat.py --top-k 8                     # retrieve 8 chunks per query
```

## Project structure

```
arxiv-rag/
├── ingest.py        # fetch, extract, chunk, embed, index
├── retriever.py     # load index + retrieve top-k chunks
├── chat.py          # multi-turn CLI chat with streaming
├── index/           # auto-created: FAISS index + metadata
└── requirements.txt
```
